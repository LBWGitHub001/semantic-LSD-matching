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
    objs0 = [{"category": r["label"], "bbox": r["bbox"], "centroid": r["centroid_2d"]} for r in item.get("regs0", [])]
    objs1 = [{"category": r["label"], "bbox": r["bbox"], "centroid": r["centroid_2d"]} for r in item.get("regs1", [])]

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
    }
