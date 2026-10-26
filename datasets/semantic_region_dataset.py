"""
Semantic Region Mask Matching PyTorch Dataset.

Loads cross-view scene pairs where:
  - Each view has a collection of semantic region binary masks (walls, floors, ceilings, furniture).
  - Ground truth provides 1-to-1 matching pairs and out-of-view / occluded dustbin regions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def extract_region_patch(bgr_img: np.ndarray, mask: np.ndarray, patch_size: int = 112) -> np.ndarray:
    """
    Directly operates image with mask:
    1. Multiplies image with binary mask (background pixels become pure black [0, 0, 0]).
    2. Crops to bounding box of the mask.
    3. Letterbox pads to (patch_size, patch_size) square, preserving true aspect ratio.
    4. Normalizes with ImageNet statistics to (3, patch_size, patch_size) float32 tensor.
    """
    masked = (bgr_img * (mask[:, :, None] > 0)).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((3, patch_size, patch_size), dtype=np.float32)

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    crop = masked[y1 : y2 + 1, x1 : x2 + 1]

    h_c, w_c = crop.shape[:2]
    scale = patch_size / max(h_c, w_c)
    new_w = max(1, min(patch_size, int(round(w_c * scale))))
    new_h = max(1, min(patch_size, int(round(h_c * scale))))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    square = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    pad_y = (patch_size - new_h) // 2
    pad_x = (patch_size - new_w) // 2
    square[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = ((square[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
    return rgb


_SHARED_LSD = None


def get_shared_lsd():
    global _SHARED_LSD
    if _SHARED_LSD is None:
        _SHARED_LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _SHARED_LSD


def extract_patch_lines_from_square(
    square_bgr: np.ndarray,
    max_lines: int = 16,
    min_length: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ultra-fast LSD line extraction directly on the 112x112 square patch.
    Takes ~1.5ms per patch instead of ~25ms on large crops.
    """
    import math

    lines_5d = np.zeros((max_lines, 5), dtype=np.float32)
    mask = np.zeros((max_lines,), dtype=bool)

    if square_bgr is None or square_bgr.size == 0:
        return lines_5d, mask

    if square_bgr.ndim == 3:
        gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = square_bgr

    H_p, W_p = gray.shape[:2]
    lsd = get_shared_lsd()
    raw_lines, _, _, _ = lsd.detect(gray)

    if raw_lines is not None:
        valid_lines = []
        for l_arr in raw_lines:
            x1, y1, x2, y2 = [float(v) for v in l_arr.flatten()]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_length:
                continue
            theta = math.atan2(y2 - y1, x2 - x1)
            cu = (x1 + x2) / (2.0 * float(W_p))
            cv = (y1 + y2) / (2.0 * float(H_p))
            l_norm = length / float(max(H_p, W_p))
            sin2 = math.sin(2.0 * theta)
            cos2 = math.cos(2.0 * theta)
            valid_lines.append(([cu, cv, l_norm, sin2, cos2], length))

        valid_lines.sort(key=lambda x: x[1], reverse=True)
        for k, (feat, _) in enumerate(valid_lines[:max_lines]):
            lines_5d[k] = feat
            mask[k] = True

    return lines_5d, mask


def extract_region_patch_from_crop(
    crop_bgra: np.ndarray,
    patch_size: int = 112,
    return_square: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Directly pads and normalizes an already-segmented small image crop (BGRA or BGR)
    to a (3, patch_size, patch_size) float32 tensor.
    If return_square=True, returns (rgb_tensor, square_uint8_bgr).
    """
    if crop_bgra is None or crop_bgra.size == 0:
        empty = np.zeros((3, patch_size, patch_size), dtype=np.float32)
        return (empty, np.zeros((patch_size, patch_size, 3), dtype=np.uint8)) if return_square else empty

    if crop_bgra.ndim == 3 and crop_bgra.shape[2] == 4:
        crop_bgr = crop_bgra[..., :3] * (crop_bgra[..., 3:] > 0)
    else:
        crop_bgr = crop_bgra

    h_c, w_c = crop_bgr.shape[:2]
    scale = patch_size / max(h_c, w_c)
    new_w = max(1, min(patch_size, int(round(w_c * scale))))
    new_h = max(1, min(patch_size, int(round(h_c * scale))))

    resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    square = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    pad_y = (patch_size - new_h) // 2
    pad_x = (patch_size - new_w) // 2
    square[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = ((square[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
    return (rgb, square) if return_square else rgb


def extract_patch_lines_from_crop(
    crop_bgr: np.ndarray,
    max_lines: int = 16,
    min_length: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts straight lines inside a region crop (compatibility wrapper)."""
    res = extract_region_patch_from_crop(crop_bgr, patch_size=112, return_square=True)
    _, square = res if isinstance(res, tuple) else (res, np.zeros((112, 112, 3), dtype=np.uint8))
    return extract_patch_lines_from_square(square, max_lines=max_lines, min_length=min_length)


class SemanticRegionDataset(Dataset):
    """
    PyTorch Dataset for Cross-View Semantic Region Mask Matching.
    Directly extracts small region patches (112x112) from image & masks for network matching.
    """

    def __init__(
        self,
        dataset_dir: str | Path = "data/structured3d_semantic_regions",
        split: str = "train",
        patch_size: int = 112,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        min_area_ratio: float = 0.50,
        load_masks: bool = False,
        load_lines: bool = False,
        max_lines_per_patch: int = 16,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_dir).resolve()
        self.split = split
        self.patch_size = patch_size
        self.transform = transform
        self.min_area_ratio = min_area_ratio
        self.load_masks = load_masks
        self.load_lines = load_lines
        self.max_lines_per_patch = max_lines_per_patch

        split_file = self.root / f"{split}_split.json"
        if not split_file.is_file():
            # If split file not found, list all pair directories
            pair_dirs = sorted([d.name for d in self.root.iterdir() if d.is_dir() and (d / "correspondence.json").is_file()])
            if not pair_dirs:
                raise FileNotFoundError(f"No valid pair directories found in {self.root}")
            n_total = len(pair_dirs)
            n_tr = int(n_total * 0.8)
            n_va = int(n_total * 0.1)
            if split == "train":
                raw_pair_ids = pair_dirs[:n_tr]
            elif split == "val":
                raw_pair_ids = pair_dirs[n_tr:n_tr+n_va]
            else:
                raw_pair_ids = pair_dirs[n_tr+n_va:]
        else:
            with open(split_file, "r", encoding="utf-8") as f:
                raw_pair_ids = json.load(f)

        # In-memory metadata cache for parsed JSON/NPZ structures
        # Avoids repeated disk opens, json parsing, and OS file descriptor pressure across epochs
        self._meta_cache: dict[str, tuple[list[dict], list[dict], dict, Any]] = {}

        # Filter pairs by minimum mutually matched area ratio
        filtered_ids = []
        for pid in raw_pair_ids:
            p_dir = self.root / pid
            corr_file = p_dir / "correspondence.json"
            r0_file = p_dir / "regions_0.json"
            r1_file = p_dir / "regions_1.json"
            if not (corr_file.is_file() and r0_file.is_file() and r1_file.is_file()):
                continue
            try:
                with open(corr_file, "r", encoding="utf-8") as f:
                    corr = json.load(f)
                with open(r0_file, "r", encoding="utf-8") as f:
                    regs0 = json.load(f)
                with open(r1_file, "r", encoding="utf-8") as f:
                    regs1 = json.load(f)

                if self.min_area_ratio > 0.0:
                    area0_map = {r["idx"]: r.get("pixel_area", 0) for r in regs0}
                    area1_map = {r["idx"]: r.get("pixel_area", 0) for r in regs1}
                    tot0 = sum(area0_map.values())
                    tot1 = sum(area1_map.values())
                    if tot0 <= 0 or tot1 <= 0:
                        continue
                    m0 = sum(area0_map.get(m["mask0_idx"], 0) for m in corr["matches"])
                    m1 = sum(area1_map.get(m["mask1_idx"], 0) for m in corr["matches"])
                    ratio0 = m0 / tot0
                    ratio1 = m1 / tot1
                    if min(ratio0, ratio1) < self.min_area_ratio:
                        continue

                filtered_ids.append(pid)
                # Store parsed metadata directly in shared memory cache
                self._meta_cache[pid] = (regs0, regs1, corr, None)
            except Exception:
                continue

        self.pair_ids = filtered_ids
        if self.min_area_ratio > 0.0:
            print(f"[*] SemanticRegionDataset ({split}): Filtered {len(self.pair_ids)}/{len(raw_pair_ids)} pairs with matched area ratio >= {self.min_area_ratio*100:.0f}%.")
        else:
            print(f"[*] SemanticRegionDataset ({split}): Loaded {len(self.pair_ids)} cross-view region pairs.")

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_id = self.pair_ids[idx]
        p_dir = self.root / pair_id

        # 1. Optionally Load Full RGB Images (only if load_masks=True, e.g. for visualization)
        H, W = 480, 640
        bgr0, bgr1 = None, None
        if self.load_masks:
            bgr0 = cv2.imread(str(p_dir / "image_0.png"))
            bgr1 = cv2.imread(str(p_dir / "image_1.png"))
            if bgr0 is not None:
                H, W = bgr0.shape[:2]

        # 2. Retrieve In-Memory Metadata (regions & correspondence)
        if pair_id in self._meta_cache:
            regs0, regs1, corr, cached_lines = self._meta_cache[pair_id]
        else:
            with open(p_dir / "regions_0.json", "r", encoding="utf-8") as f:
                regs0 = json.load(f)
            with open(p_dir / "regions_1.json", "r", encoding="utf-8") as f:
                regs1 = json.load(f)
            with open(p_dir / "correspondence.json", "r", encoding="utf-8") as f:
                corr = json.load(f)
            cached_lines = None
            self._meta_cache[pair_id] = (regs0, regs1, corr, cached_lines)

        # Check line cache if load_lines is True
        cache_file = p_dir / "patch_lines.npz"
        has_line_cache = False
        if self.load_lines:
            if cached_lines is not None:
                has_line_cache = True
            elif cache_file.is_file():
                try:
                    npz = np.load(cache_file)
                    t_patch_lines0 = torch.from_numpy(npz["lines0"])
                    t_line_mask0 = torch.from_numpy(npz["mask0"])
                    t_patch_lines1 = torch.from_numpy(npz["lines1"])
                    t_line_mask1 = torch.from_numpy(npz["mask1"])
                    cached_lines = (t_patch_lines0, t_line_mask0, t_patch_lines1, t_line_mask1)
                    has_line_cache = True
                    self._meta_cache[pair_id] = (regs0, regs1, corr, cached_lines)
                except Exception:
                    has_line_cache = False

        # 3. Extract Small Region Patches (112x112) directly from small crops
        masks0_list = []
        bboxes0_list = []
        patches0_list = []
        patch_lines0_list = []
        line_mask0_list = []
        for r in regs0:
            bboxes0_list.append(r["bbox"])
            if self.load_masks:
                m_path = p_dir / "masks_0" / f"mask_{r['idx']:02d}.png"
                m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    masks0_list.append((m > 127).astype(np.float32))

            crop_path = p_dir / "image_0" / f"region_{r['idx']:02d}.png"
            if crop_path.is_file():
                crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                patch, square = extract_region_patch_from_crop(crop_img, patch_size=self.patch_size, return_square=True)
                patches0_list.append(patch)
            else:
                if bgr0 is None:
                    bgr0 = cv2.imread(str(p_dir / "image_0.png"))
                m_path = p_dir / "masks_0" / f"mask_{r['idx']:02d}.png"
                m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                patch = extract_region_patch(bgr0, m, patch_size=self.patch_size)
                patches0_list.append(patch)
                square = None

            if self.load_lines and not has_line_cache:
                if square is not None:
                    lines_5d, l_mask = extract_patch_lines_from_square(square, max_lines=self.max_lines_per_patch)
                else:
                    lines_5d, l_mask = extract_patch_lines_from_crop(crop_img if crop_path.is_file() else bgr0, max_lines=self.max_lines_per_patch)
                patch_lines0_list.append(lines_5d)
                line_mask0_list.append(l_mask)

        masks1_list = []
        bboxes1_list = []
        patches1_list = []
        patch_lines1_list = []
        line_mask1_list = []
        for r in regs1:
            bboxes1_list.append(r["bbox"])
            if self.load_masks:
                m_path = p_dir / "masks_1" / f"mask_{r['idx']:02d}.png"
                m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    masks1_list.append((m > 127).astype(np.float32))

            crop_path = p_dir / "image_1" / f"region_{r['idx']:02d}.png"
            if crop_path.is_file():
                crop_img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
                patch, square = extract_region_patch_from_crop(crop_img, patch_size=self.patch_size, return_square=True)
                patches1_list.append(patch)
            else:
                if bgr1 is None:
                    bgr1 = cv2.imread(str(p_dir / "image_1.png"))
                m_path = p_dir / "masks_1" / f"mask_{r['idx']:02d}.png"
                m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                patch = extract_region_patch(bgr1, m, patch_size=self.patch_size)
                patches1_list.append(patch)
                square = None

            if self.load_lines and not has_line_cache:
                if square is not None:
                    lines_5d, l_mask = extract_patch_lines_from_square(square, max_lines=self.max_lines_per_patch)
                else:
                    lines_5d, l_mask = extract_patch_lines_from_crop(crop_img if crop_path.is_file() else bgr1, max_lines=self.max_lines_per_patch)
                patch_lines1_list.append(lines_5d)
                line_mask1_list.append(l_mask)

        # 4. Correspondence Ground Truth (loaded from metadata cache)
        gt_matches = [(m["mask0_idx"], m["mask1_idx"]) for m in corr["matches"]]
        unmatched_0 = corr.get("unmatched_in_view0", [])
        unmatched_1 = corr.get("unmatched_in_view1", [])

        # 5. Image Normalization & Tensor Conversion
        if self.load_masks and bgr0 is not None and bgr1 is not None:
            rgb0 = ((bgr0[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
            rgb1 = ((bgr1[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
            t_img0 = torch.from_numpy(rgb0)
            t_img1 = torch.from_numpy(rgb1)
            t_masks0 = torch.from_numpy(np.stack(masks0_list, axis=0)) if masks0_list else torch.zeros((0, H, W), dtype=torch.float32)
            t_masks1 = torch.from_numpy(np.stack(masks1_list, axis=0)) if masks1_list else torch.zeros((0, H, W), dtype=torch.float32)
        else:
            t_img0 = None
            t_img1 = None
            t_masks0 = None
            t_masks1 = None

        M0 = len(patches0_list)
        M1 = len(patches1_list)
        t_patches0 = torch.from_numpy(np.stack(patches0_list, axis=0)) if M0 > 0 else torch.zeros((0, 3, self.patch_size, self.patch_size), dtype=torch.float32)
        t_patches1 = torch.from_numpy(np.stack(patches1_list, axis=0)) if M1 > 0 else torch.zeros((0, 3, self.patch_size, self.patch_size), dtype=torch.float32)

        t_bboxes0 = torch.from_numpy(np.array(bboxes0_list, dtype=np.float32)) if M0 > 0 else torch.zeros((0, 4), dtype=torch.float32)
        t_bboxes1 = torch.from_numpy(np.array(bboxes1_list, dtype=np.float32)) if M1 > 0 else torch.zeros((0, 4), dtype=torch.float32)

        # Normalized 2D centroids in [-1, 1] for motion & transformation consistency
        centroids0_list = []
        for r in regs0:
            c = r.get("centroid_2d", [(r["bbox"][0] + r["bbox"][2]) / 2.0, (r["bbox"][1] + r["bbox"][3]) / 2.0])
            centroids0_list.append([c[0] / float(W) * 2.0 - 1.0, c[1] / float(H) * 2.0 - 1.0])

        centroids1_list = []
        for r in regs1:
            c = r.get("centroid_2d", [(r["bbox"][0] + r["bbox"][2]) / 2.0, (r["bbox"][1] + r["bbox"][3]) / 2.0])
            centroids1_list.append([c[0] / float(W) * 2.0 - 1.0, c[1] / float(H) * 2.0 - 1.0])

        t_centroids0 = torch.from_numpy(np.array(centroids0_list, dtype=np.float32)) if M0 > 0 else torch.zeros((0, 2), dtype=torch.float32)
        t_centroids1 = torch.from_numpy(np.array(centroids1_list, dtype=np.float32)) if M1 > 0 else torch.zeros((0, 2), dtype=torch.float32)

        if self.load_lines:
            if has_line_cache and cached_lines is not None:
                t_patch_lines0 = cached_lines[0].clone()
                t_line_mask0 = cached_lines[1].clone()
                t_patch_lines1 = cached_lines[2].clone()
                t_line_mask1 = cached_lines[3].clone()
            else:
                t_patch_lines0 = torch.from_numpy(np.stack(patch_lines0_list, axis=0)) if M0 > 0 else torch.zeros((0, self.max_lines_per_patch, 5), dtype=torch.float32)
                t_line_mask0 = torch.from_numpy(np.stack(line_mask0_list, axis=0)) if M0 > 0 else torch.zeros((0, self.max_lines_per_patch), dtype=torch.bool)
                t_patch_lines1 = torch.from_numpy(np.stack(patch_lines1_list, axis=0)) if M1 > 0 else torch.zeros((0, self.max_lines_per_patch, 5), dtype=torch.float32)
                t_line_mask1 = torch.from_numpy(np.stack(line_mask1_list, axis=0)) if M1 > 0 else torch.zeros((0, self.max_lines_per_patch), dtype=torch.bool)
                cached_lines = (t_patch_lines0, t_line_mask0, t_patch_lines1, t_line_mask1)
                self._meta_cache[pair_id] = (regs0, regs1, corr, cached_lines)
                try:
                    np.savez_compressed(
                        cache_file,
                        lines0=t_patch_lines0.numpy(),
                        mask0=t_line_mask0.numpy(),
                        lines1=t_patch_lines1.numpy(),
                        mask1=t_line_mask1.numpy(),
                    )
                except Exception:
                    pass
        else:
            t_patch_lines0 = None
            t_line_mask0 = None
            t_patch_lines1 = None
            t_line_mask1 = None

        # 6. Target Assignment Matrix: (M0+1, M1+1)
        gt_matrix = torch.zeros((M0 + 1, M1 + 1), dtype=torch.float32)
        for i, j in gt_matches:
            if i < M0 and j < M1:
                gt_matrix[i, j] = 1.0

        for i in unmatched_0:
            if i < M0:
                gt_matrix[i, M1] = 1.0

        for j in unmatched_1:
            if j < M1:
                gt_matrix[M0, j] = 1.0

        # Region pixel areas and matched area ratios
        area0_map = {r["idx"]: r.get("pixel_area", 0) for r in regs0}
        area1_map = {r["idx"]: r.get("pixel_area", 0) for r in regs1}
        tot0 = sum(area0_map.values())
        tot1 = sum(area1_map.values())
        matched_area0 = sum(area0_map.get(m[0], 0) for m in gt_matches)
        matched_area1 = sum(area1_map.get(m[1], 0) for m in gt_matches)
        ratio0 = matched_area0 / tot0 if tot0 > 0 else 0.0
        ratio1 = matched_area1 / tot1 if tot1 > 0 else 0.0
        areas0 = [r.get("pixel_area", 0) for r in regs0]
        areas1 = [r.get("pixel_area", 0) for r in regs1]

        return {
            "pair_id": pair_id,
            "patches0": t_patches0,
            "patches1": t_patches1,
            "img0": t_img0,
            "img1": t_img1,
            "masks0": t_masks0,
            "masks1": t_masks1,
            "bboxes0": t_bboxes0,
            "bboxes1": t_bboxes1,
            "patch_lines0": t_patch_lines0,
            "patch_lines1": t_patch_lines1,
            "line_mask0": t_line_mask0,
            "line_mask1": t_line_mask1,
            "centroids0": t_centroids0,
            "centroids1": t_centroids1,
            "gt_matches": gt_matches,
            "unmatched_0": unmatched_0,
            "unmatched_1": unmatched_1,
            "gt_matrix": gt_matrix,
            "num_regions0": M0,
            "num_regions1": M1,
            "rotation_deg": corr.get("rotation_deg", 0.0),
            "baseline_m": corr.get("baseline_m", 0.0),
            "bgr0": bgr0,
            "bgr1": bgr1,
            "regs0": regs0,
            "regs1": regs1,
            "areas0": areas0,
            "areas1": areas1,
            "matched_area_ratio_0": ratio0,
            "matched_area_ratio_1": ratio1,
            "line_matches": corr.get("line_matches", []),
            "unmatched_lines_0": corr.get("unmatched_lines_0", []),
            "unmatched_lines_1": corr.get("unmatched_lines_1", []),
        }


def collate_region_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a single pair into batched tensors."""
    item = batch[0]
    objs0 = [
        {
            "category": r.get("category", r["label"]),
            "label": r["label"],
            "bbox": r["bbox"],
            "centroid": r["centroid_2d"],
        }
        for r in item.get("regs0", [])
    ]
    objs1 = [
        {
            "category": r.get("category", r["label"]),
            "label": r["label"],
            "bbox": r["bbox"],
            "centroid": r["centroid_2d"],
        }
        for r in item.get("regs1", [])
    ]

    return {
        "pair_id": item["pair_id"],
        "scene_name": item["pair_id"],
        "view_i": 0,
        "view_j": 1,
        "patches0": item["patches0"].unsqueeze(0),
        "patches1": item["patches1"].unsqueeze(0),
        "img0": item["img0"].unsqueeze(0) if item.get("img0") is not None else None,
        "img1": item["img1"].unsqueeze(0) if item.get("img1") is not None else None,
        "masks0": item["masks0"].unsqueeze(0) if item.get("masks0") is not None else None,
        "masks1": item["masks1"].unsqueeze(0) if item.get("masks1") is not None else None,
        "bboxes0": item["bboxes0"].unsqueeze(0),
        "bboxes1": item["bboxes1"].unsqueeze(0),
        "patch_lines0": item["patch_lines0"].unsqueeze(0) if item.get("patch_lines0") is not None else None,
        "patch_lines1": item["patch_lines1"].unsqueeze(0) if item.get("patch_lines1") is not None else None,
        "line_mask0": item["line_mask0"].unsqueeze(0) if item.get("line_mask0") is not None else None,
        "line_mask1": item["line_mask1"].unsqueeze(0) if item.get("line_mask1") is not None else None,
        "centroids0": item["centroids0"].unsqueeze(0) if "centroids0" in item else None,
        "centroids1": item["centroids1"].unsqueeze(0) if "centroids1" in item else None,
        "gt_matches": [item["gt_matches"]],
        "unmatched_0": [item["unmatched_0"]],
        "unmatched_1": [item["unmatched_1"]],
        "gt_matrix": item["gt_matrix"].unsqueeze(0),
        "rotation_deg": item["rotation_deg"],
        "baseline_m": item["baseline_m"],
        "raw_img0": item.get("bgr0"),
        "raw_img1": item.get("bgr1"),
        "objs0": objs0,
        "objs1": objs1,
        "regs0": item.get("regs0", []),
        "regs1": item.get("regs1", []),
        "areas0": item.get("areas0", []),
        "areas1": item.get("areas1", []),
        "matched_area_ratio_0": item.get("matched_area_ratio_0", 0.0),
        "matched_area_ratio_1": item.get("matched_area_ratio_1", 0.0),
    }


DISTINCT_COLORS = [
    (40, 220, 40),    # Bright Green
    (0, 180, 255),    # Vibrant Orange
    (255, 200, 0),    # Vivid Cyan
    (220, 60, 220),   # Bright Magenta
    (0, 235, 235),    # Lemon Yellow
    (50, 100, 255),   # Coral
    (200, 100, 255),  # Lavender Pink
    (255, 140, 40),   # Sky Blue
    (100, 240, 140),  # Mint Green
    (180, 220, 50),   # Teal
    (240, 140, 180),  # Violet
    (80, 180, 240),   # Gold Amber
]


def visualize_semantic_region_matches(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    masks0: np.ndarray | torch.Tensor,
    masks1: np.ndarray | torch.Tensor,
    regs0: list[dict[str, Any]],
    regs1: list[dict[str, Any]],
    pred_matches: list[tuple[int, int]],
    gt_matches: list[tuple[int, int]],
    pred_un0: list[int],
    pred_un1: list[int],
    out_path: Path | str,
    title: str = "Semantic Region Mask Matching",
    connect_tp: bool = False,
) -> None:
    """
    Renders a unified dual-pair comparison canvas containing TWO PAIRS of images stacked vertically:
      - Top Pair (Row 1): Model Prediction & Error Diagnostics
          * Green mask: overlay for correctly matched regions (TP).
          * Red mask: overlay for incorrectly matched regions (FP).
          * Red lines: connect centroids of incorrectly matched regions to diagnose errors.
          * Yellow dashed lines: connect missed ground truth pairs (FN).
          * Blue lines: region boundaries across all semantic segments.
      - Bottom Pair (Row 2): Ground Truth Target Matching
          * Matching regions highlighted with identical distinct colors per GT pair.
          * Solid colored lines: connect corresponding centroids across views.
          * Blue lines: region boundaries across all semantic segments.
    """
    if isinstance(masks0, torch.Tensor):
        m0_arr = masks0.detach().cpu().numpy()
    else:
        m0_arr = np.asarray(masks0)
    if m0_arr.ndim == 4:
        m0_arr = m0_arr[0]

    if isinstance(masks1, torch.Tensor):
        m1_arr = masks1.detach().cpu().numpy()
    else:
        m1_arr = np.asarray(masks1)
    if m1_arr.ndim == 4:
        m1_arr = m1_arr[0]

    H0, W0 = bgr0.shape[:2]
    H1, W1 = bgr1.shape[:2]
    H = max(H0, H1)
    header_h = 50
    gap = 30
    W_total = W0 + W1 + gap

    # Layout geometry: Top row (Model Pred) + Middle banner + Bottom row (Ground Truth)
    row1_y = header_h
    mid_header_y = row1_y + H
    row2_y = mid_header_y + header_h
    total_h = row2_y + H + 12

    canvas = np.full((total_h, W_total, 3), 26, dtype=np.uint8)

    gt_set = set(gt_matches)
    pred_set = set(pred_matches)
    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    GREEN = (40, 220, 40)      # BGR Vivid Green
    RED = (30, 30, 240)        # BGR Vivid Red
    BLUE = (255, 110, 0)       # BGR Vivid Cyan-Blue

    # =========================================================================
    # ROW 1: MODEL PREDICTION & ERROR DIAGNOSTICS (上层：模型预测与诊断分析)
    # =========================================================================
    overlay0_p = bgr0.copy()
    overlay1_p = bgr1.copy()

    # 1. 绿色掩膜一个匹配正确 (Green mask for correct matches - TP)
    for i, j in tp_set:
        if i < len(m0_arr):
            mask_i = m0_arr[i] > 0.5
            overlay0_p[mask_i] = (overlay0_p[mask_i] * 0.40 + np.array(GREEN) * 0.60).astype(np.uint8)
        if j < len(m1_arr):
            mask_j = m1_arr[j] > 0.5
            overlay1_p[mask_j] = (overlay1_p[mask_j] * 0.40 + np.array(GREEN) * 0.60).astype(np.uint8)

    # 2. 红色表示一个匹配不正确 (Red mask for incorrect matches - FP)
    for i, j in fp_set:
        if i < len(m0_arr):
            mask_i = m0_arr[i] > 0.5
            overlay0_p[mask_i] = (overlay0_p[mask_i] * 0.35 + np.array(RED) * 0.65).astype(np.uint8)
        if j < len(m1_arr):
            mask_j = m1_arr[j] > 0.5
            overlay1_p[mask_j] = (overlay1_p[mask_j] * 0.35 + np.array(RED) * 0.65).astype(np.uint8)

    # 3. 区域之间使用蓝色的线划分出来 (Blue boundary lines for all regions)
    for i in range(len(m0_arr)):
        mask_i = (m0_arr[i] > 0.5).astype(np.uint8)
        cnts0, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay0_p, cnts0, -1, BLUE, 2)

    for j in range(len(m1_arr)):
        mask_j = (m1_arr[j] > 0.5).astype(np.uint8)
        cnts1, _ = cv2.findContours(mask_j, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay1_p, cnts1, -1, BLUE, 2)

    canvas[row1_y : row1_y + H0, :W0] = overlay0_p
    canvas[row1_y : row1_y + H1, W0 + gap :] = overlay1_p

    # Correct match markers (Green centroid dots)
    for i, j in tp_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row1_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row1_y))
        cv2.circle(canvas, pt0, 5, GREEN, -1)
        cv2.circle(canvas, pt1, 5, GREEN, -1)
        if connect_tp:
            cv2.line(canvas, pt0, pt1, (60, 200, 60), 1, cv2.LINE_AA)

    # 红色的区域用线连起来表示如何匹配的 (Red lines connecting incorrect matches - FP)
    for i, j in fp_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row1_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row1_y))

        cv2.circle(canvas, pt0, 6, RED, -1)
        cv2.circle(canvas, pt1, 6, RED, -1)
        cv2.line(canvas, pt0, pt1, RED, 2, cv2.LINE_AA)

        cat0 = regs0[i].get("category", regs0[i].get("label", "")) if i < len(regs0) else ""
        cat1 = regs1[j].get("category", regs1[j].get("label", "")) if j < len(regs1) else ""
        tag = f"{cat0}->{cat1} [FP WRONG]"
        cv2.putText(canvas, tag, (pt0[0] + 6, pt0[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # False Negatives (Missed GT matches - yellow dashed lines)
    for i, j in fn_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row1_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row1_y))
        yellow = (0, 215, 235)
        dist = np.hypot(pt1[0] - pt0[0], pt1[1] - pt0[1])
        steps = max(int(dist / 16), 1)
        for s in range(0, steps, 2):
            p_a = (int(pt0[0] + (pt1[0] - pt0[0]) * (s / steps)),
                   int(pt0[1] + (pt1[1] - pt0[1]) * (s / steps)))
            p_b = (int(pt0[0] + (pt1[0] - pt0[0]) * (min(s + 1, steps) / steps)),
                   int(pt0[1] + (pt1[1] - pt0[1]) * (min(s + 1, steps) / steps)))
            cv2.line(canvas, p_a, p_b, yellow, 1, cv2.LINE_AA)

    # Row 1 Banner
    n_tp, n_fp, n_fn = len(tp_set), len(fp_set), len(fn_set)
    pair_prec = n_tp / (n_tp + n_fp) * 100.0 if (n_tp + n_fp) > 0 else 0.0
    pair_rec = n_tp / (n_tp + n_fn) * 100.0 if (n_tp + n_fn) > 0 else 0.0
    status_str = f"TP={n_tp} (Green Mask) | FP={n_fp} (Red Mask & Red Line) | FN={n_fn} (Yellow Dashed) | Prec={pair_prec:.1f}% | Rec={pair_rec:.1f}%"
    cv2.rectangle(canvas, (0, 0), (W_total, header_h), (34, 34, 34), -1)
    cv2.putText(canvas, f"[TOP: MODEL PREDICTION & DIAGNOSIS] {title} [{status_str}]", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    # =========================================================================
    # ROW 2: GROUND TRUTH TARGET CORRESPONDENCES (下层：真值匹配对)
    # =========================================================================
    overlay0_gt = bgr0.copy()
    overlay1_gt = bgr1.copy()

    # 1. 对应区域染上相同的高对比度辨识色 (Identical distinct color per GT pair)
    for m_idx, (i, j) in enumerate(gt_matches):
        col = DISTINCT_COLORS[m_idx % len(DISTINCT_COLORS)]
        if i < len(m0_arr):
            mask_i = m0_arr[i] > 0.5
            overlay0_gt[mask_i] = (overlay0_gt[mask_i] * 0.42 + np.array(col) * 0.58).astype(np.uint8)
        if j < len(m1_arr):
            mask_j = m1_arr[j] > 0.5
            overlay1_gt[mask_j] = (overlay1_gt[mask_j] * 0.42 + np.array(col) * 0.58).astype(np.uint8)

    # 2. 区域之间使用蓝色的线划分出来 (Blue boundary lines for all regions)
    for i in range(len(m0_arr)):
        mask_i = (m0_arr[i] > 0.5).astype(np.uint8)
        cnts0, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay0_gt, cnts0, -1, BLUE, 2)

    for j in range(len(m1_arr)):
        mask_j = (m1_arr[j] > 0.5).astype(np.uint8)
        cnts1, _ = cv2.findContours(mask_j, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay1_gt, cnts1, -1, BLUE, 2)

    canvas[row2_y : row2_y + H0, :W0] = overlay0_gt
    canvas[row2_y : row2_y + H1, W0 + gap :] = overlay1_gt

    # 3. 绘制真值连接线与目标名称 (Solid matching lines between centroids)
    matched_0_set = {i for i, _ in gt_matches}
    matched_1_set = {j for _, j in gt_matches}
    un_0_count = len(regs0) - len(matched_0_set)
    un_1_count = len(regs1) - len(matched_1_set)

    for m_idx, (i, j) in enumerate(gt_matches):
        col = DISTINCT_COLORS[m_idx % len(DISTINCT_COLORS)]
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row2_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row2_y))

        cv2.circle(canvas, pt0, 5, col, -1)
        cv2.circle(canvas, pt1, 5, col, -1)
        cv2.line(canvas, pt0, pt1, col, 2, cv2.LINE_AA)

        lbl = regs0[i].get("label", regs0[i].get("category", f"match_{m_idx}")) if i < len(regs0) else ""
        cv2.putText(canvas, lbl, (pt0[0] + 5, pt0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Row 2 Middle Banner
    gt_banner = f"[BOTTOM: GROUND TRUTH CORRESPONDENCES] Target Matches: {len(gt_matches)} pairs | Unmatched: View0={un_0_count}, View1={un_1_count} | Matching regions share identical color & connecting lines"
    cv2.rectangle(canvas, (0, mid_header_y), (W_total, mid_header_y + header_h), (34, 34, 34), -1)
    cv2.putText(canvas, gt_banner, (20, mid_header_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas)

