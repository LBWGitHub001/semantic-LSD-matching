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


class SemanticRegionDataset(Dataset):
    """
    PyTorch Dataset for Cross-View Semantic Region Mask Matching.
    """

    def __init__(
        self,
        dataset_dir: str | Path = "data/structured3d_semantic_regions",
        split: str = "train",
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_dir).resolve()
        self.split = split
        self.transform = transform

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
                self.pair_ids = pair_dirs[:n_tr]
            elif split == "val":
                self.pair_ids = pair_dirs[n_tr:n_tr+n_va]
            else:
                self.pair_ids = pair_dirs[n_tr+n_va:]
        else:
            with open(split_file, "r", encoding="utf-8") as f:
                self.pair_ids = json.load(f)

        print(f"[*] SemanticRegionDataset ({split}): Loaded {len(self.pair_ids)} cross-view region pairs.")

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_id = self.pair_ids[idx]
        p_dir = self.root / pair_id

        # 1. Load RGB Images
        bgr0 = cv2.imread(str(p_dir / "image_0.png"))
        bgr1 = cv2.imread(str(p_dir / "image_1.png"))
        if bgr0 is None or bgr1 is None:
            raise FileNotFoundError(f"Missing images in {p_dir}")

        H, W = bgr0.shape[:2]

        # 2. Load Region Metadata
        with open(p_dir / "regions_0.json", "r", encoding="utf-8") as f:
            regs0 = json.load(f)
        with open(p_dir / "regions_1.json", "r", encoding="utf-8") as f:
            regs1 = json.load(f)

        # 3. Load Binary Masks
        masks0_list = []
        bboxes0_list = []
        for r in regs0:
            m_path = p_dir / "masks_0" / f"mask_{r['idx']:02d}.png"
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            masks0_list.append((m > 127).astype(np.float32))
            bboxes0_list.append(r["bbox"])

        masks1_list = []
        bboxes1_list = []
        for r in regs1:
            m_path = p_dir / "masks_1" / f"mask_{r['idx']:02d}.png"
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            masks1_list.append((m > 127).astype(np.float32))
            bboxes1_list.append(r["bbox"])

        # 4. Load Correspondence Ground Truth
        with open(p_dir / "correspondence.json", "r", encoding="utf-8") as f:
            corr = json.load(f)

        gt_matches = [(m["mask0_idx"], m["mask1_idx"]) for m in corr["matches"]]
        unmatched_0 = corr.get("unmatched_in_view0", [])
        unmatched_1 = corr.get("unmatched_in_view1", [])

        # 5. Image Normalization
        rgb0 = ((bgr0[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        rgb1 = ((bgr1[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)

        # Stack into tensors
        M0 = len(masks0_list)
        M1 = len(masks1_list)
        t_masks0 = torch.from_numpy(np.stack(masks0_list, axis=0)) if M0 > 0 else torch.zeros((0, H, W), dtype=torch.float32)
        t_masks1 = torch.from_numpy(np.stack(masks1_list, axis=0)) if M1 > 0 else torch.zeros((0, H, W), dtype=torch.float32)

        t_bboxes0 = torch.from_numpy(np.array(bboxes0_list, dtype=np.float32)) if M0 > 0 else torch.zeros((0, 4), dtype=torch.float32)
        t_bboxes1 = torch.from_numpy(np.array(bboxes1_list, dtype=np.float32)) if M1 > 0 else torch.zeros((0, 4), dtype=torch.float32)

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

        return {
            "pair_id": pair_id,
            "img0": torch.from_numpy(rgb0),
            "img1": torch.from_numpy(rgb1),
            "masks0": t_masks0,
            "masks1": t_masks1,
            "bboxes0": t_bboxes0,
            "bboxes1": t_bboxes1,
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
        "img0": item["img0"].unsqueeze(0),
        "img1": item["img1"].unsqueeze(0),
        "masks0": item["masks0"].unsqueeze(0),
        "masks1": item["masks1"].unsqueeze(0),
        "bboxes0": item["bboxes0"].unsqueeze(0),
        "bboxes1": item["bboxes1"].unsqueeze(0),
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
    }


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
) -> None:
    """
    Renders genuine semantic mask matching visualization with alpha-blending,
    mask contours, centroid connections, and explicit TP/FP/FN error diagnostics.
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
    header_h = 60
    canvas = np.full((H + header_h, W0 + W1 + 30, 3), 32, dtype=np.uint8)

    # Place base images
    canvas[header_h : header_h + H0, :W0] = bgr0
    canvas[header_h : header_h + H1, W0 + 30 :] = bgr1

    overlay0 = canvas[header_h : header_h + H0, :W0].copy()
    overlay1 = canvas[header_h : header_h + H1, W0 + 30 :].copy()

    gt_set = set(gt_matches)
    pred_set = set(pred_matches)
    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    # 12 Distinct high-contrast palette colors (BGR)
    palette = [
        (40, 220, 40),    # Vivid Green
        (0, 165, 255),    # Orange
        (255, 210, 30),   # Cyan
        (230, 80, 230),   # Magenta
        (30, 240, 255),   # Yellow
        (80, 120, 255),   # Coral
        (200, 130, 255),  # Pink
        (255, 160, 50),   # Sky Blue
        (120, 240, 140),  # Mint
        (70, 240, 240),   # Lime Yellow
        (220, 190, 60),   # Aqua Teal
        (240, 130, 190),  # Lavender
    ]

    # 1. Neutral tint for truly unmatched regions (low opacity dark overlay, no bounding box clutter)
    for i in pred_un0:
        if i < len(m0_arr):
            mask_i = m0_arr[i] > 0.5
            overlay0[mask_i] = (overlay0[mask_i] * 0.75 + np.array([70, 70, 70]) * 0.25).astype(np.uint8)

    for j in pred_un1:
        if j < len(m1_arr):
            mask_j = m1_arr[j] > 0.5
            overlay1[mask_j] = (overlay1[mask_j] * 0.75 + np.array([70, 70, 70]) * 0.25).astype(np.uint8)

    # 2. Render TP matches (Colored mask overlay + solid contours)
    tp_list = sorted(list(tp_set))
    for idx, (i, j) in enumerate(tp_list):
        col = palette[idx % len(palette)]
        if i < len(m0_arr):
            mask_i = m0_arr[i] > 0.5
            overlay0[mask_i] = (overlay0[mask_i] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
            cnts0, _ = cv2.findContours((mask_i * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay0, cnts0, -1, col, 2)

        if j < len(m1_arr):
            mask_j = m1_arr[j] > 0.5
            overlay1[mask_j] = (overlay1[mask_j] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
            cnts1, _ = cv2.findContours((mask_j * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay1, cnts1, -1, col, 2)

    # 3. Render FP matches (Red tint overlay)
    for i, j in fp_set:
        red_col = (40, 40, 230)
        if i < len(m0_arr):
            mask_i = m0_arr[i] > 0.5
            overlay0[mask_i] = (overlay0[mask_i] * 0.55 + np.array(red_col) * 0.45).astype(np.uint8)
        if j < len(m1_arr):
            mask_j = m1_arr[j] > 0.5
            overlay1[mask_j] = (overlay1[mask_j] * 0.55 + np.array(red_col) * 0.45).astype(np.uint8)

    canvas[header_h : header_h + H0, :W0] = overlay0
    canvas[header_h : header_h + H1, W0 + 30 :] = overlay1

    # 4. Draw Lines & Diagnostics
    # (a) True Positives: Green lines between centroids
    for idx, (i, j) in enumerate(tp_list):
        col = palette[idx % len(palette)]
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + header_h))
        pt1 = (int(c1[0] + W0 + 30), int(c1[1] + header_h))

        cv2.circle(canvas, pt0, 5, col, -1)
        cv2.circle(canvas, pt1, 5, col, -1)
        cv2.line(canvas, pt0, pt1, col, 2, cv2.LINE_AA)

        label_txt = regs0[i].get("label", str(i)) if i < len(regs0) else f"m{i}"
        cv2.putText(canvas, f"{label_txt} [TP]", (pt0[0] + 6, pt0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # (b) False Positives: Red lines + Explicit class mismatch warning
    for i, j in fp_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + header_h))
        pt1 = (int(c1[0] + W0 + 30), int(c1[1] + header_h))

        red = (40, 40, 240)
        cv2.circle(canvas, pt0, 6, red, -1)
        cv2.circle(canvas, pt1, 6, red, -1)
        cv2.line(canvas, pt0, pt1, red, 2, cv2.LINE_AA)

        cat0 = regs0[i].get("category", regs0[i].get("label", "")) if i < len(regs0) else ""
        cat1 = regs1[j].get("category", regs1[j].get("label", "")) if j < len(regs1) else ""
        tag = f"{cat0}->{cat1} [FP WRONG]"
        cv2.putText(canvas, tag, (pt0[0] + 6, pt0[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    # (c) False Negatives: Yellow dashed line indicating missed GT match
    for i, j in fn_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + header_h))
        pt1 = (int(c1[0] + W0 + 30), int(c1[1] + header_h))

        yellow = (0, 220, 240)
        # Draw dashed line
        dist = np.hypot(pt1[0] - pt0[0], pt1[1] - pt0[1])
        steps = max(int(dist / 14), 1)
        for s in range(0, steps, 2):
            p_a = (int(pt0[0] + (pt1[0] - pt0[0]) * (s / steps)),
                   int(pt0[1] + (pt1[1] - pt0[1]) * (s / steps)))
            p_b = (int(pt0[0] + (pt1[0] - pt0[0]) * (min(s + 1, steps) / steps)),
                   int(pt0[1] + (pt1[1] - pt0[1]) * (min(s + 1, steps) / steps)))
            cv2.line(canvas, p_a, p_b, yellow, 1, cv2.LINE_AA)

        lbl = regs0[i].get("label", "") if i < len(regs0) else ""
        cv2.putText(canvas, f"{lbl} [MISSED]", (pt0[0] + 6, pt0[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, yellow, 1, cv2.LINE_AA)

    # 5. Header diagnostics banner
    n_tp = len(tp_set)
    n_fp = len(fp_set)
    n_fn = len(fn_set)
    pair_prec = n_tp / (n_tp + n_fp) * 100.0 if (n_tp + n_fp) > 0 else 0.0
    pair_rec = n_tp / (n_tp + n_fn) * 100.0 if (n_tp + n_fn) > 0 else 0.0
    status_str = f"TP={n_tp} | FP={n_fp} | FN={n_fn} | Precision={pair_prec:.1f}% | Recall={pair_rec:.1f}%"

    cv2.putText(canvas, f"{title} [{status_str}]", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas)

