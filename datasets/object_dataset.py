"""
Object Matching Dataset and Visualization Utilities ("连连看").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ObjectMatchingDataset(Dataset):
    """
    Dataset for pairwise object matching.
    Loads RGB views, instance masks, and object bounding boxes for pairs of viewpoints.
    """

    def __init__(
        self,
        data_dir: str | Path = "data/synthetic_object_matching_scenes",
        split: str = "train",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split

        splits_file = self.data_dir / "dataset_splits.json"
        if not splits_file.exists():
            raise FileNotFoundError(f"Missing {splits_file}. Run 'python generate_dataset.py' first.")

        with open(splits_file, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        scene_names = splits_data.get(split, [])
        self.samples: list[dict[str, Any]] = []

        for s_name in scene_names:
            s_dir = self.data_dir / s_name
            pairs_file = s_dir / "pair_matches.json"
            if not pairs_file.exists():
                continue
            with open(pairs_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)

            for p in p_data["pairs"]:
                self.samples.append({
                    "scene_dir": str(s_dir),
                    "scene_name": s_name,
                    "view_i": p["view_i"],
                    "view_j": p["view_j"],
                    "gt_matches": [tuple(m) for m in p["gt_matches"]],
                    "unmatched_i": p["unmatched_i"],
                    "unmatched_j": p["unmatched_j"],
                })

        print(f"[{split.upper()}] Loaded {len(self.samples)} pairs from {len(scene_names)} scenes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        info = self.samples[idx]
        s_dir = Path(info["scene_dir"])
        vi, vj = info["view_i"], info["view_j"]

        img0 = cv2.imread(str(s_dir / f"view_{vi:02d}.png"))
        img1 = cv2.imread(str(s_dir / f"view_{vj:02d}.png"))
        H, W, _ = img0.shape

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        rgb0_norm = ((img0[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
        rgb1_norm = ((img1[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)

        mask0 = cv2.imread(str(s_dir / f"instance_mask_{vi:02d}.png"), cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.imread(str(s_dir / f"instance_mask_{vj:02d}.png"), cv2.IMREAD_GRAYSCALE)

        with open(s_dir / f"objects_{vi:02d}.json", "r", encoding="utf-8") as f:
            objs0 = json.load(f)
        with open(s_dir / f"objects_{vj:02d}.json", "r", encoding="utf-8") as f:
            objs1 = json.load(f)

        m0_tensors, bboxes0, patches0 = [], [], []
        for obj in objs0:
            m = (mask0 == obj["inst_id"]).astype(np.float32)
            m0_tensors.append(m)
            bboxes0.append(obj["bbox"])
            # Extract small region patch (112x112)
            masked = (img0 * (m[:, :, None] > 0)).astype(np.uint8)
            ys, xs = np.where(m > 0)
            if len(xs) > 0 and len(ys) > 0:
                crop = masked[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                scale = 112 / max(crop.shape[:2])
                nw, nh = max(1, min(112, int(round(crop.shape[1] * scale)))), max(1, min(112, int(round(crop.shape[0] * scale))))
                res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                sq = np.zeros((112, 112, 3), dtype=np.uint8)
                sq[(112 - nh) // 2 : (112 - nh) // 2 + nh, (112 - nw) // 2 : (112 - nw) // 2 + nw] = res
                rgb = ((sq[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
            else:
                rgb = np.zeros((3, 112, 112), dtype=np.float32)
            patches0.append(rgb)

        m1_tensors, bboxes1, patches1 = [], [], []
        for obj in objs1:
            m = (mask1 == obj["inst_id"]).astype(np.float32)
            m1_tensors.append(m)
            bboxes1.append(obj["bbox"])
            masked = (img1 * (m[:, :, None] > 0)).astype(np.uint8)
            ys, xs = np.where(m > 0)
            if len(xs) > 0 and len(ys) > 0:
                crop = masked[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                scale = 112 / max(crop.shape[:2])
                nw, nh = max(1, min(112, int(round(crop.shape[1] * scale)))), max(1, min(112, int(round(crop.shape[0] * scale))))
                res = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                sq = np.zeros((112, 112, 3), dtype=np.uint8)
                sq[(112 - nh) // 2 : (112 - nh) // 2 + nh, (112 - nw) // 2 : (112 - nw) // 2 + nw] = res
                rgb = ((sq[..., ::-1] / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
            else:
                rgb = np.zeros((3, 112, 112), dtype=np.float32)
            patches1.append(rgb)

        t_patches0 = torch.from_numpy(np.stack(patches0, axis=0)) if patches0 else torch.zeros((0, 3, 112, 112), dtype=torch.float32)
        t_patches1 = torch.from_numpy(np.stack(patches1, axis=0)) if patches1 else torch.zeros((0, 3, 112, 112), dtype=torch.float32)

        masks0_arr = np.stack(m0_tensors, axis=0) if m0_tensors else np.zeros((1, H, W), dtype=np.float32)
        bboxes0_arr = np.array(bboxes0, dtype=np.float32) if bboxes0 else np.zeros((1, 4), dtype=np.float32)
        masks1_arr = np.stack(m1_tensors, axis=0) if m1_tensors else np.zeros((1, H, W), dtype=np.float32)
        bboxes1_arr = np.array(bboxes1, dtype=np.float32) if bboxes1 else np.zeros((1, 4), dtype=np.float32)

        return {
            "patches0": t_patches0,
            "patches1": t_patches1,
            "img0": torch.from_numpy(rgb0_norm),
            "img1": torch.from_numpy(rgb1_norm),
            "raw_img0": img0,
            "raw_img1": img1,
            "masks0": torch.from_numpy(masks0_arr),
            "bboxes0": torch.from_numpy(bboxes0_arr),
            "masks1": torch.from_numpy(masks1_arr),
            "bboxes1": torch.from_numpy(bboxes1_arr),
            "objs0": objs0,
            "objs1": objs1,
            "gt_matches": info["gt_matches"],
            "unmatched_0": info["unmatched_i"],
            "unmatched_1": info["unmatched_j"],
            "scene_name": info["scene_name"],
            "view_i": vi,
            "view_j": vj,
        }


def collate_single_pair(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Single pair collator for variable-sized object sets."""
    item = batch[0]
    return {
        "patches0": item["patches0"].unsqueeze(0) if "patches0" in item else None,
        "patches1": item["patches1"].unsqueeze(0) if "patches1" in item else None,
        "img0": item["img0"].unsqueeze(0),
        "img1": item["img1"].unsqueeze(0),
        "raw_img0": item["raw_img0"],
        "raw_img1": item["raw_img1"],
        "masks0": item["masks0"].unsqueeze(0),
        "bboxes0": item["bboxes0"].unsqueeze(0),
        "masks1": item["masks1"].unsqueeze(0),
        "bboxes1": item["bboxes1"].unsqueeze(0),
        "objs0": item["objs0"],
        "objs1": item["objs1"],
        "gt_matches": [item["gt_matches"]],
        "unmatched_0": [item["unmatched_0"]],
        "unmatched_1": [item["unmatched_1"]],
        "scene_name": item["scene_name"],
        "view_i": item["view_i"],
        "view_j": item["view_j"],
    }


def predict_bipartite_matches(
    prob_matrix: torch.Tensor | np.ndarray,
    thresh: float = 0.2,
    categories_0: list[str] | None = None,
    categories_1: list[str] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Computes mutual nearest neighbor (MNN) assignments from doubly-stochastic Sinkhorn matrix.
    prob_matrix: (M0 + 1, M1 + 1)
    Optional categories_0 & categories_1 enforce class compatibility (masking out cross-class logits).
    """
    if isinstance(prob_matrix, torch.Tensor):
        P = prob_matrix.detach().cpu().numpy().copy()
    else:
        P = np.array(prob_matrix, copy=True)

    M0_plus_1, M1_plus_1 = P.shape
    M0 = M0_plus_1 - 1
    M1 = M1_plus_1 - 1

    # Optional hard semantic masking: zero out matches between disparate classes
    if categories_0 is not None and categories_1 is not None:
        for i in range(min(M0, len(categories_0))):
            c0 = categories_0[i]
            for j in range(min(M1, len(categories_1))):
                c1 = categories_1[j]
                if c0 != c1:
                    P[i, j] = 0.0

    matches = []
    unmatched_0 = []

    best1 = np.argmax(P[:M0, :], axis=1)
    best0 = np.argmax(P[:, :M1], axis=0)

    for i in range(M0):
        j = int(best1[i])
        if j == M1:
            unmatched_0.append(i)
        elif j < M1:
            if best0[j] == i and P[i, j] >= thresh:
                matches.append((i, j))
            else:
                unmatched_0.append(i)

    matched_j = {j for _, j in matches}
    unmatched_1 = [j for j in range(M1) if j not in matched_j]

    return matches, unmatched_0, unmatched_1


def visualize_object_matches(
    raw_img0: np.ndarray,
    raw_img1: np.ndarray,
    objs0: list[dict[str, Any]],
    objs1: list[dict[str, Any]],
    pred_matches: list[tuple[int, int]],
    gt_matches: list[tuple[int, int]],
    pred_un0: list[int],
    pred_un1: list[int],
    out_path: Path | str,
    title: str = "Object Bipartite Matching",
) -> None:
    """Draws side-by-side '连连看' visualization with green match lines and bounding boxes."""
    H0, W0, _ = raw_img0.shape
    H1, W1, _ = raw_img1.shape
    canvas_h = max(H0, H1) + 60
    canvas_w = W0 + W1 + 40
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    y_offset = 50
    x0_offset = 10
    x1_offset = W0 + 30

    canvas[y_offset : y_offset + H0, x0_offset : x0_offset + W0] = raw_img0
    canvas[y_offset : y_offset + H1, x1_offset : x1_offset + W1] = raw_img1

    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)

    gt_set = set(gt_matches)

    for i in pred_un0:
        if i < len(objs0):
            bx = objs0[i]["bbox"]
            p1 = (int(bx[0] + x0_offset), int(bx[1] + y_offset))
            p2 = (int(bx[2] + x0_offset), int(bx[3] + y_offset))
            cv2.rectangle(canvas, p1, p2, (40, 40, 230), 2)
            cv2.putText(canvas, f"{objs0[i]['category']} (unmatched)", (p1[0], max(p1[1] - 6, y_offset + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    for j in pred_un1:
        if j < len(objs1):
            bx = objs1[j]["bbox"]
            p1 = (int(bx[0] + x1_offset), int(bx[1] + y_offset))
            p2 = (int(bx[2] + x1_offset), int(bx[3] + y_offset))
            cv2.rectangle(canvas, p1, p2, (40, 40, 230), 2)
            cv2.putText(canvas, f"{objs1[j]['category']} (unmatched)", (p1[0], max(p1[1] - 6, y_offset + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    for i, j in pred_matches:
        if i >= len(objs0) or j >= len(objs1):
            continue
        b0 = objs0[i]["bbox"]
        b1 = objs1[j]["bbox"]
        c0 = objs0[i].get("centroid", [(b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0])
        c1 = objs1[j].get("centroid", [(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0])

        is_correct = (i, j) in gt_set if gt_set else True
        color = (40, 190, 40) if is_correct else (200, 30, 220)

        p0_min = (int(b0[0] + x0_offset), int(b0[1] + y_offset))
        p0_max = (int(b0[2] + x0_offset), int(b0[3] + y_offset))
        cv2.rectangle(canvas, p0_min, p0_max, color, 2)
        cv2.putText(canvas, objs0[i]["category"], (p0_min[0], max(p0_min[1] - 6, y_offset + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        p1_min = (int(b1[0] + x1_offset), int(b1[1] + y_offset))
        p1_max = (int(b1[2] + x1_offset), int(b1[3] + y_offset))
        cv2.rectangle(canvas, p1_min, p1_max, color, 2)
        cv2.putText(canvas, objs1[j]["category"], (p1_min[0], max(p1_min[1] - 6, y_offset + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        pt_from = (int(c0[0] + x0_offset), int(c0[1] + y_offset))
        pt_to = (int(c1[0] + x1_offset), int(c1[1] + y_offset))
        cv2.line(canvas, pt_from, pt_to, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_from, 5, color, -1)
        cv2.circle(canvas, pt_to, 5, color, -1)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas)
