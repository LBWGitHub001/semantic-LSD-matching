"""
Training and Evaluation Pipeline for XFeatObjectMatcher ("连连看" Object Bipartite Matcher).

Features:
  1. Dataset loader for synthetic_object_matching_scenes (40 train / 5 val / 5 test).
  2. Loss: SuperGlue-style Negative Log-Likelihood (NLL) with dustbin supervision.
  3. Mixed Precision Training (AMP) for RTX 4060 GPU acceleration.
  4. Metrics: Precision, Recall, F1-Score, and Dustbin Accuracy.
  5. Side-by-side "连连看" visualization: green lines connecting matched object bounding boxes,
     red boxes for out-of-view/dustbin objects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat.xfeat_object_matcher import ObjectMatchingLoss, XFeatObjectMatcher


# ==============================================================================
# 1. Dataset Loader
# ==============================================================================
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
            raise FileNotFoundError(f"Missing {splits_file}. Run scripts/generate_object_matching_dataset.py first.")

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

        # 1. Load RGB images
        img0 = cv2.imread(str(s_dir / f"view_{vi:02d}.png"))
        img1 = cv2.imread(str(s_dir / f"view_{vj:02d}.png"))
        H, W, _ = img0.shape

        # Normalize to ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        rgb0_norm = ((img0[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)
        rgb1_norm = ((img1[..., ::-1] / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32)

        # 2. Load Instance Masks
        mask0 = cv2.imread(str(s_dir / f"instance_mask_{vi:02d}.png"), cv2.IMREAD_GRAYSCALE)
        mask1 = cv2.imread(str(s_dir / f"instance_mask_{vj:02d}.png"), cv2.IMREAD_GRAYSCALE)

        # 3. Load Object Metadata
        with open(s_dir / f"objects_{vi:02d}.json", "r", encoding="utf-8") as f:
            objs0 = json.load(f)
        with open(s_dir / f"objects_{vj:02d}.json", "r", encoding="utf-8") as f:
            objs1 = json.load(f)

        # Create binary masks tensor: (M, H, W)
        m0_tensors = []
        bboxes0 = []
        for obj in objs0:
            b_mask = (mask0 == obj["inst_id"]).astype(np.float32)
            m0_tensors.append(b_mask)
            bboxes0.append(obj["bbox"])

        m1_tensors = []
        bboxes1 = []
        for obj in objs1:
            b_mask = (mask1 == obj["inst_id"]).astype(np.float32)
            m1_tensors.append(b_mask)
            bboxes1.append(obj["bbox"])

        masks0_arr = np.stack(m0_tensors, axis=0) if m0_tensors else np.zeros((1, H, W), dtype=np.float32)
        bboxes0_arr = np.array(bboxes0, dtype=np.float32) if bboxes0 else np.zeros((1, 4), dtype=np.float32)
        masks1_arr = np.stack(m1_tensors, axis=0) if m1_tensors else np.zeros((1, H, W), dtype=np.float32)
        bboxes1_arr = np.array(bboxes1, dtype=np.float32) if bboxes1 else np.zeros((1, 4), dtype=np.float32)

        return {
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


# ==============================================================================
# 2. Evaluation & Prediction
# ==============================================================================
def predict_bipartite_matches(
    prob_matrix: torch.Tensor,
    thresh: float = 0.2,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Computes mutual nearest neighbor (MNN) assignments from doubly-stochastic Sinkhorn matrix.
    prob_matrix: (M0 + 1, M1 + 1)
    Returns:
      predicted_matches: list of (idx0, idx1)
      predicted_unmatched_0: list of idx0 assigned to dustbin
      predicted_unmatched_1: list of idx1 assigned to dustbin
    """
    P = prob_matrix.detach().cpu().numpy()
    M0_plus_1, M1_plus_1 = P.shape
    M0 = M0_plus_1 - 1
    M1 = M1_plus_1 - 1

    matches = []
    unmatched_0 = []

    best1 = np.argmax(P[:M0, :], axis=1)  # (M0,) best col for each row
    best0 = np.argmax(P[:, :M1], axis=0)  # (M1,) best row for each col

    for i in range(M0):
        j = int(best1[i])
        if j == M1:
            # Explicitly assigned to dustbin column
            unmatched_0.append(i)
        elif j < M1:
            # Check mutual consistency
            if best0[j] == i and P[i, j] >= thresh:
                matches.append((i, j))
            else:
                unmatched_0.append(i)

    # Check View 1 unmatched
    matched_j = {j for _, j in matches}
    unmatched_1 = [j for j in range(M1) if j not in matched_j]

    return matches, unmatched_0, unmatched_1


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Computes Precision, Recall, F1, and Dustbin Accuracy over a dataset loader."""
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_dustbin_correct = 0
    total_dustbin_items = 0

    with torch.no_grad():
        for batch in loader:
            img0 = batch["img0"].to(device)
            img1 = batch["img1"].to(device)
            masks0 = batch["masks0"].to(device)
            bboxes0 = batch["bboxes0"].to(device)
            masks1 = batch["masks1"].to(device)
            bboxes1 = batch["bboxes1"].to(device)

            out = model(img0, img1, masks0, bboxes0, masks1, bboxes1)
            P = out["prob_matrix"][0]  # (M0+1, M1+1)

            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(P)

            gt_matches = set(batch["gt_matches"][0])
            pred_set = set(pred_matches)

            tp = len(pred_set.intersection(gt_matches))
            fp = len(pred_set - gt_matches)
            fn = len(gt_matches - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Dustbin accuracy
            gt_un0 = set(batch["unmatched_0"][0])
            gt_un1 = set(batch["unmatched_1"][0])

            dustbin_correct = len(set(pred_un0).intersection(gt_un0)) + len(set(pred_un1).intersection(gt_un1))
            total_dustbin_correct += dustbin_correct
            total_dustbin_items += (len(gt_un0) + len(gt_un1))

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    dustbin_acc = total_dustbin_correct / (total_dustbin_items + 1e-6) if total_dustbin_items > 0 else 1.0

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "dustbin_acc": dustbin_acc * 100.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


# ==============================================================================
# 3. "连连看" Visualizer
# ==============================================================================
def visualize_object_matches(
    raw_img0: np.ndarray,
    raw_img1: np.ndarray,
    objs0: list[dict[str, Any]],
    objs1: list[dict[str, Any]],
    pred_matches: list[tuple[int, int]],
    gt_matches: list[tuple[int, int]],
    pred_un0: list[int],
    pred_un1: list[int],
    out_path: Path,
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

    # Place raw images
    canvas[y_offset : y_offset + H0, x0_offset : x0_offset + W0] = raw_img0
    canvas[y_offset : y_offset + H1, x1_offset : x1_offset + W1] = raw_img1

    # Title header
    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)

    gt_set = set(gt_matches)
    matched_i = {i for i, _ in pred_matches}
    matched_j = {j for _, j in pred_matches}

    # Draw Unmatched / Dustbin in View 0 (Red dashed/solid boxes)
    for i in pred_un0:
        if i < len(objs0):
            bx = objs0[i]["bbox"]
            p1 = (int(bx[0] + x0_offset), int(bx[1] + y_offset))
            p2 = (int(bx[2] + x0_offset), int(bx[3] + y_offset))
            cv2.rectangle(canvas, p1, p2, (40, 40, 230), 2)
            cv2.putText(canvas, f"{objs0[i]['category']} (unmatched)", (p1[0], max(p1[1] - 6, y_offset + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    # Draw Unmatched / Dustbin in View 1 (Red boxes)
    for j in pred_un1:
        if j < len(objs1):
            bx = objs1[j]["bbox"]
            p1 = (int(bx[0] + x1_offset), int(bx[1] + y_offset))
            p2 = (int(bx[2] + x1_offset), int(bx[3] + y_offset))
            cv2.rectangle(canvas, p1, p2, (40, 40, 230), 2)
            cv2.putText(canvas, f"{objs1[j]['category']} (unmatched)", (p1[0], max(p1[1] - 6, y_offset + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    # Draw Matches (Green lines connecting box centroids)
    for i, j in pred_matches:
        if i >= len(objs0) or j >= len(objs1):
            continue
        b0 = objs0[i]["bbox"]
        b1 = objs1[j]["bbox"]
        c0 = objs0[i]["centroid"]
        c1 = objs1[j]["centroid"]

        is_correct = (i, j) in gt_set
        color = (40, 190, 40) if is_correct else (200, 30, 220)  # Green if correct, Magenta if wrong

        # Box 0
        p0_min = (int(b0[0] + x0_offset), int(b0[1] + y_offset))
        p0_max = (int(b0[2] + x0_offset), int(b0[3] + y_offset))
        cv2.rectangle(canvas, p0_min, p0_max, color, 2)
        cv2.putText(canvas, objs0[i]["category"], (p0_min[0], max(p0_min[1] - 6, y_offset + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Box 1
        p1_min = (int(b1[0] + x1_offset), int(b1[1] + y_offset))
        p1_max = (int(b1[2] + x1_offset), int(b1[3] + y_offset))
        cv2.rectangle(canvas, p1_min, p1_max, color, 2)
        cv2.putText(canvas, objs1[j]["category"], (p1_min[0], max(p1_min[1] - 6, y_offset + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Connection line across images
        pt_from = (int(c0[0] + x0_offset), int(c0[1] + y_offset))
        pt_to = (int(c1[0] + x1_offset), int(c1[1] + y_offset))
        cv2.line(canvas, pt_from, pt_to, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_from, 5, color, -1)
        cv2.circle(canvas, pt_to, 5, color, -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


# ==============================================================================
# 4. Main Training Loop
# ==============================================================================
def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Datasets
    train_ds = ObjectMatchingDataset(data_dir=args.data_dir, split="train")
    val_ds = ObjectMatchingDataset(data_dir=args.data_dir, split="val")
    test_ds = ObjectMatchingDataset(data_dir=args.data_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single_pair, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_pair, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_single_pair, num_workers=2)

    # Model & Loss
    model = XFeatObjectMatcher(
        feat_dim=64,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        sinkhorn_iters=20,
    ).to(device)

    loss_fn = ObjectMatchingLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=args.amp)

    best_val_f1 = 0.0
    ckpt_dir = Path("checkpoints/xfeat_object_matcher")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pth"

    print(f"\nStarting training XFeatObjectMatcher for {args.epochs} epochs (AMP={args.amp})...")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            img0 = batch["img0"].to(device)
            img1 = batch["img1"].to(device)
            masks0 = batch["masks0"].to(device)
            bboxes0 = batch["bboxes0"].to(device)
            masks1 = batch["masks1"].to(device)
            bboxes1 = batch["bboxes1"].to(device)

            with autocast(enabled=args.amp):
                out = model(img0, img1, masks0, bboxes0, masks1, bboxes1)
                log_P = out["log_P"]
                loss = loss_fn(
                    log_P,
                    batch["gt_matches"],
                    batch["unmatched_0"],
                    batch["unmatched_1"],
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            step_count += 1

        train_loss = total_loss / max(step_count, 1)
        epoch_sec = time.time() - t0

        # Validate
        val_metrics = evaluate_dataset(model, val_loader, device)
        val_f1 = val_metrics["f1"]

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
            }, best_ckpt_path)

        star = " ★ BEST" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_sec:.1f}s] - Train Loss: {train_loss:.4f} | "
              f"Val Prec: {val_metrics['precision']:.1f}% | Rec: {val_metrics['recall']:.1f}% | "
              f"F1: {val_f1:.1f}% | Dustbin: {val_metrics['dustbin_acc']:.1f}%{star}")

    print("=" * 80)
    print(f"Training Complete! Best Val F1: {best_val_f1:.2f}%. Saved -> {best_ckpt_path}")

    # Final Benchmark on Unseen Test Set
    print("\nEvaluating best checkpoint on unseen test scenes...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_dataset(model, test_loader, device)

    print("-" * 60)
    print(f"TEST SET BENCHMARK RESULTS (10 unseen scenes, {len(test_ds)} pairs):")
    print(f"  Precision       : {test_metrics['precision']:.2f}%")
    print(f"  Recall          : {test_metrics['recall']:.2f}%")
    print(f"  F1-Score        : {test_metrics['f1']:.2f}%")
    print(f"  Dustbin Accuracy: {test_metrics['dustbin_acc']:.2f}%")
    print(f"  Total TP Matches: {test_metrics['tp']}")
    print(f"  Total FP Matches: {test_metrics['fp']}")
    print(f"  Total FN Matches: {test_metrics['fn']}")
    print("-" * 60)

    # Export Visualizations
    vis_dir = Path("outputs/object_matching_vis")
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting sample '连连看' visualizations to {vis_dir}...")

    model.eval()
    with torch.no_grad():
        for vis_idx, batch in enumerate(test_loader):
            if vis_idx >= 5:
                break
            img0 = batch["img0"].to(device)
            img1 = batch["img1"].to(device)
            masks0 = batch["masks0"].to(device)
            bboxes0 = batch["bboxes0"].to(device)
            masks1 = batch["masks1"].to(device)
            bboxes1 = batch["bboxes1"].to(device)

            out = model(img0, img1, masks0, bboxes0, masks1, bboxes1)
            P = out["prob_matrix"][0]
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(P)

            save_file = vis_dir / f"test_sample_{vis_idx + 1}_{batch['scene_name']}_v{batch['view_i']}_v{batch['view_j']}.jpg"
            visualize_object_matches(
                batch["raw_img0"],
                batch["raw_img1"],
                batch["objs0"],
                batch["objs1"],
                pred_matches,
                batch["gt_matches"][0],
                pred_un0,
                pred_un1,
                save_file,
                title=f"Sample {vis_idx + 1}: {batch['scene_name']} (View {batch['view_i']} <-> View {batch['view_j']})",
            )
            print(f"  Saved -> {save_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XFeatObjectMatcher")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_object_matching_scenes")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
