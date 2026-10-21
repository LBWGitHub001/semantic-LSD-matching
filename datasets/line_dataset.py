"""
LineFeat PyTorch Dataset for Cross-Image Line Segment Matching.

Loads image pairs, ground-truth 3D-to-2D projected line matches, camera intrinsics/extrinsics,
and semantic segmentation maps from the precomputed dataset index.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.read_structured3d import Camera, STRUCTURED3D_CLASSES


class LineFeatPairDataset(Dataset):
    """
    PyTorch Dataset yielding view pairs with verified ground-truth line segment matches.

    Each item contains:
      - 'image0', 'image1': Normalized RGB image tensors (3, H, W)
      - 'semantic0', 'semantic1': Semantic class label maps (H, W)
      - 'gt_lines0', 'gt_lines1': Matched line coordinates (M, 4) [x1, y1, x2, y2]
      - 'gt_line_ids': Global 3D line IDs (M,)
      - 'gt_semantics': Semantic class name for each match (list of str)
      - 'K0', 'K1': Pinhole intrinsics (3, 3)
      - 'R_0to1', 't_0to1': Relative pose from view0 to view1
      - 'rotation_deg', 'baseline_meters': Geometric baseline metrics
    """

    def __init__(
        self,
        index_path: str | Path = "data/line_gt_dataset/index.json",
        dataset_root: str | Path | None = None,
        min_matches: int = 1,
        max_rotation_deg: float | None = 85.0,
        scenes: Sequence[str] | None = None,
        split: str | None = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        load_images: bool = True,
        load_semantics: bool = True,
        load_depths: bool = False,
        load_cameras: bool = True,
        image_size: tuple[int, int] | None = None,  # (H, W) or None for original
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        p = Path(index_path)
        if not p.is_file():
            cand = Path(__file__).resolve().parent.parent / index_path
            if cand.is_file():
                p = cand
        self.index_path = p.resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self.gt_dir = self.index_path.parent

        with open(self.index_path, "r", encoding="utf-8") as handle:
            raw_index = json.load(handle)

        # Dataset root resolution
        if dataset_root is not None:
            self.dataset_root = Path(dataset_root).resolve()
        else:
            cand = Path(raw_index.get("dataset_root", ""))
            if cand.is_dir():
                self.dataset_root = cand
            else:
                # Fallback to local workspace search
                cand2 = Path("Structured3D_perspective_full_00/Structured3D").resolve()
                if cand2.is_dir():
                    self.dataset_root = cand2
                else:
                    raise FileNotFoundError("Could not locate Structured3D dataset directory.")

        self.load_images = load_images
        self.load_semantics = load_semantics
        self.load_depths = load_depths
        self.load_cameras = load_cameras
        self.image_size = image_size
        self.transform = transform

        # Filter pairs
        pairs = raw_index["pairs"]
        all_scenes = sorted(list({p["scene_id"] for p in pairs}))

        # Handle split / scene filtering
        if scenes is not None:
            scene_set = set(scenes)
            pairs = [p for p in pairs if p["scene_id"] in scene_set]
        elif split is not None:
            num_scenes = len(all_scenes)
            n_train = int(num_scenes * train_ratio)
            n_val = int(num_scenes * val_ratio)
            if split == "train":
                active_scenes = set(all_scenes[:n_train])
            elif split == "val":
                active_scenes = set(all_scenes[n_train : n_train + n_val])
            elif split == "test":
                active_scenes = set(all_scenes[n_train + n_val :])
            else:
                raise ValueError(f"Unknown split: {split}. Choose 'train', 'val', or 'test'.")
            pairs = [p for p in pairs if p["scene_id"] in active_scenes]

        # Criteria filtering
        filtered_pairs = []
        for p in pairs:
            if p.get("num_matches", 0) < min_matches:
                continue
            if max_rotation_deg is not None and p.get("relative_rotation_deg", 0.0) > max_rotation_deg:
                continue
            filtered_pairs.append(p)

        self.pairs = filtered_pairs

        # Cache for scene JSON annotations to minimize disk reads
        self._scene_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _get_scene_data(self, scene_anno_rel: str) -> dict[str, Any]:
        if scene_anno_rel not in self._scene_cache:
            anno_path = self.gt_dir / scene_anno_rel
            with open(anno_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            # Index pairs by pair_id for O(1) lookup
            pair_map = {p["pair_id"]: p for p in data.get("pairs", [])}
            self._scene_cache[scene_anno_rel] = pair_map
        return self._scene_cache[scene_anno_rel]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_meta = self.pairs[idx]
        pair_id = pair_meta["pair_id"]
        scene_anno_rel = pair_meta["scene_annotation"]

        # Fetch detailed match info
        scene_pairs = self._get_scene_data(scene_anno_rel)
        pair_detail = scene_pairs.get(pair_id, pair_meta)

        # Paths
        v0_dir = self.dataset_root.parent.parent / pair_meta["view0_dir"]
        v1_dir = self.dataset_root.parent.parent / pair_meta["view1_dir"]

        if not v0_dir.is_dir():
            # Try direct relative to workspace
            v0_dir = Path(pair_meta["view0_dir"]).resolve()
            v1_dir = Path(pair_meta["view1_dir"]).resolve()

        item: dict[str, Any] = {
            "pair_id": pair_id,
            "scene_id": pair_meta["scene_id"],
            "room_id": pair_meta["room_id"],
            "rotation_deg": float(pair_meta.get("relative_rotation_deg", 0.0)),
            "baseline_meters": float(pair_meta.get("baseline_meters", 0.0)),
        }

        # 1. Images
        img0_path = v0_dir / "rgb_rawlight.png"
        img1_path = v1_dir / "rgb_rawlight.png"
        item["image0_path"] = str(img0_path)
        item["image1_path"] = str(img1_path)

        orig_w, orig_h = 1280, 720
        sx, sy = 1.0, 1.0

        if self.load_images:
            bgr0 = cv2.imread(str(img0_path))
            bgr1 = cv2.imread(str(img1_path))
            if bgr0 is None or bgr1 is None:
                raise FileNotFoundError(f"Could not read image files for pair {pair_id}")

            orig_h, orig_w = bgr0.shape[:2]

            if self.image_size is not None and (orig_h, orig_w) != self.image_size:
                th, tw = self.image_size
                sx = tw / orig_w
                sy = th / orig_h
                bgr0 = cv2.resize(bgr0, (tw, th), interpolation=cv2.INTER_AREA)
                bgr1 = cv2.resize(bgr1, (tw, th), interpolation=cv2.INTER_AREA)

            rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
            rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)

            # (H, W, 3) -> (3, H, W) float32 in [0, 1]
            t0 = torch.from_numpy(rgb0).permute(2, 0, 1).float() / 255.0
            t1 = torch.from_numpy(rgb1).permute(2, 0, 1).float() / 255.0

            if self.transform is not None:
                t0 = self.transform(t0)
                t1 = self.transform(t1)

            item["image0"] = t0
            item["image1"] = t1

        # 2. Semantic segmentation (preserve true 0..40 class IDs)
        if self.load_semantics:
            sem0_path = v0_dir / "semantic.png"
            sem1_path = v1_dir / "semantic.png"
            sem0_raw = np.array(Image.open(sem0_path), dtype=np.uint8)
            sem1_raw = np.array(Image.open(sem1_path), dtype=np.uint8)

            if self.image_size is not None and (orig_h, orig_w) != self.image_size:
                th, tw = self.image_size
                sem0_raw = cv2.resize(sem0_raw, (tw, th), interpolation=cv2.INTER_NEAREST)
                sem1_raw = cv2.resize(sem1_raw, (tw, th), interpolation=cv2.INTER_NEAREST)

            item["semantic0"] = torch.from_numpy(sem0_raw).long()
            item["semantic1"] = torch.from_numpy(sem1_raw).long()

        # 3. Depth maps (optional, converted to meters float32)
        if self.load_depths:
            d0_path = v0_dir / "depth.png"
            d1_path = v1_dir / "depth.png"
            d0_raw = cv2.imread(str(d0_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            d1_raw = cv2.imread(str(d1_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

            if self.image_size is not None and (orig_h, orig_w) != self.image_size:
                th, tw = self.image_size
                d0_raw = cv2.resize(d0_raw, (tw, th), interpolation=cv2.INTER_NEAREST)
                d1_raw = cv2.resize(d1_raw, (tw, th), interpolation=cv2.INTER_NEAREST)

            item["depth0"] = torch.from_numpy(d0_raw)
            item["depth1"] = torch.from_numpy(d1_raw)

        # 4. Cameras & Relative Pose
        if self.load_cameras:
            cam0 = Camera.from_file(v0_dir / "camera_pose.txt", width=orig_w, height=orig_h)
            cam1 = Camera.from_file(v1_dir / "camera_pose.txt", width=orig_w, height=orig_h)

            K0 = cam0.K.copy()
            K1 = cam1.K.copy()

            if sx != 1.0 or sy != 1.0:
                K0[0, :] *= sx
                K0[1, :] *= sy
                K1[0, :] *= sx
                K1[1, :] *= sy

            # Relative pose: R_rel = R1_w2c @ R0_c2w = R1_c2w.T @ R0_c2w
            # t_rel = R1_w2c @ (pos0 - pos1) in meters
            R_rel = cam1.R_w2c @ cam0.R_c2w
            t_rel = (cam1.R_w2c @ (cam0.position - cam1.position)) / 1000.0  # mm to meters

            item["K0"] = torch.from_numpy(K0).float()
            item["K1"] = torch.from_numpy(K1).float()
            item["R_0to1"] = torch.from_numpy(R_rel).float()
            item["t_0to1"] = torch.from_numpy(t_rel).float()

        # 5. Ground Truth Line Matches
        matches = pair_detail.get("matches", [])
        lines0_list = []
        lines1_list = []
        ids_list = []
        sem_list = []

        for m in matches:
            l0 = m["line0"]
            l1 = m["line1"]
            x1_0, y1_0 = l0["start"][0] * sx, l0["start"][1] * sy
            x2_0, y2_0 = l0["end"][0] * sx, l0["end"][1] * sy

            x1_1, y1_1 = l1["start"][0] * sx, l1["start"][1] * sy
            x2_1, y2_1 = l1["end"][0] * sx, l1["end"][1] * sy

            lines0_list.append([x1_0, y1_0, x2_0, y2_0])
            lines1_list.append([x1_1, y1_1, x2_1, y2_1])
            ids_list.append(m["line_id"])
            sem_list.append(m.get("semantic", "void"))

        if len(lines0_list) > 0:
            item["gt_lines0"] = torch.tensor(lines0_list, dtype=torch.float32)
            item["gt_lines1"] = torch.tensor(lines1_list, dtype=torch.float32)
            item["gt_line_ids"] = torch.tensor(ids_list, dtype=torch.long)
        else:
            item["gt_lines0"] = torch.zeros((0, 4), dtype=torch.float32)
            item["gt_lines1"] = torch.zeros((0, 4), dtype=torch.float32)
            item["gt_line_ids"] = torch.zeros((0,), dtype=torch.long)

        item["gt_semantics"] = sem_list

        return item


def collate_line_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function for LineFeatPairDataset DataLoader.

    Handles variable numbers of ground-truth line matches per pair.
    Fixed-size tensors (images, camera matrices) are stacked into batched tensors (B, ...),
    while variable line tensors are kept as lists of tensors.
    """
    collated: dict[str, Any] = {}
    keys = batch[0].keys()

    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            # Check if all shapes match
            shapes = [v.shape for v in vals]
            if len(set(shapes)) == 1:
                collated[k] = torch.stack(vals, dim=0)
            else:
                # Variable length (e.g., gt_lines0 of different M)
                collated[k] = vals
        else:
            collated[k] = vals

    return collated
