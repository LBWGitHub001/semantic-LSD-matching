"""
Unified Training Entrypoint for Object and Line Matching Networks.

Usage:
  # 1. Train Object Matcher ('连连看' XFeatObjectMatcher):
  python train.py --model xfeat_object --epochs 12 --lr 5e-4 --amp

  # 2. Train 6-Channel Geometry Field Line Matcher (MaskLineFeatMatchNet):
  python train.py --model mask_line --epochs 8 --lr 2e-4 --amp

  # 3. Train Baseline Line-Transformer (LineMatcherNet):
  python train.py --model line_transformer --epochs 8 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_loss, build_model
from datasets import (
    ObjectMatchingDataset,
    collate_single_pair,
    predict_bipartite_matches,
    visualize_object_matches,
    LineFeatPairDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Trainer for Line & Object Matching Networks")
    parser.add_argument("--model", "--model-type", dest="model_type", type=str, default="xfeat_object",
                        choices=["xfeat_object", "xfeat", "mask_line", "mask", "line_transformer", "linefeat"],
                        help="Model architecture: 'xfeat_object' (XFeatObjectMatcher '连连看'), 'mask_line' (MaskLineFeatMatchNet), 'line_transformer' (LineMatcherNet)")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory or index path")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 5e-4 for xfeat_object, 1e-4 for line models)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum GT matches per pair (for line models)")
    parser.add_argument("--d-model", type=int, default=128, help="Feature dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (FP16)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Checkpoint directory")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per epoch for quick dry-run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ==============================================================================
# Object Matching Training Pipeline
# ==============================================================================
def train_object_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir or "data/synthetic_object_matching_scenes")
    save_dir = Path(args.save_dir or "checkpoints/xfeat_object_matcher")
    save_dir.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else 5e-4
    device = torch.device(args.device)

    print("=================================================================")
    print("      Unified Trainer: XFeatObjectMatcher ('连连看') Training     ")
    print("=================================================================")
    print(f"  Device:             {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Dataset Directory:  {data_dir}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Learning Rate:      {lr}")
    print(f"  Model Dimension:    {args.d_model} (Layers: {args.num_layers}, Heads: {args.nhead})")
    print(f"  Mixed Precision:    {'Enabled (AMP FP16)' if args.amp else 'Disabled (FP32)'}")
    print(f"  Checkpoint Dir:     {save_dir}")
    print("=================================================================\n")

    # Auto-detect whether data_dir is a SemanticRegionDataset or ObjectMatchingDataset
    use_region_ds = False
    p_data = Path(data_dir)
    if (p_data / "train_split.json").exists():
        with open(p_data / "train_split.json", "r", encoding="utf-8") as f:
            splits_data = json.load(f)
            if splits_data and (p_data / splits_data[0] / "correspondence.json").exists():
                use_region_ds = True

    if use_region_ds:
        from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
        train_ds = SemanticRegionDataset(dataset_dir=data_dir, split="train")
        val_ds = SemanticRegionDataset(dataset_dir=data_dir, split="val")
        test_ds = SemanticRegionDataset(dataset_dir=data_dir, split="test")
        collate_fn = collate_region_pairs
    else:
        train_ds = ObjectMatchingDataset(data_dir=data_dir, split="train")
        val_ds = ObjectMatchingDataset(data_dir=data_dir, split="val")
        test_ds = ObjectMatchingDataset(data_dir=data_dir, split="test")
        collate_fn = collate_single_pair

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    model = build_model(
        args.model_type,
        feat_dim=64,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        sinkhorn_iters=20,
    ).to(device)

    loss_fn = build_loss(args.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_val_f1 = 0.0
    best_ckpt_path = save_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        t0 = time.time()

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            img0 = batch["img0"].to(device)
            img1 = batch["img1"].to(device)
            masks0 = batch["masks0"].to(device)
            bboxes0 = batch["bboxes0"].to(device)
            masks1 = batch["masks1"].to(device)
            bboxes1 = batch["bboxes1"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(img0, img1, masks0, bboxes0, masks1, bboxes1)
                loss = loss_fn(out["log_P"], batch["gt_matches"], batch["unmatched_0"], batch["unmatched_1"])
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            step_count += 1

            if step_count % args.grad_accum == 0 or step == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

        train_loss = total_loss / max(step_count, 1)
        epoch_sec = time.time() - t0

        if epoch % args.val_interval == 0:
            val_metrics = evaluate_object_pipeline(model, val_loader, device)
            val_f1 = val_metrics["f1"]
            is_best = val_f1 >= best_val_f1
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
    print(f"[+] Training Complete! Best Val F1: {best_val_f1:.2f}%. Saved -> {best_ckpt_path}")

    # Benchmark test set & save visualizations
    print("\nEvaluating best checkpoint on unseen test scenes...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_object_pipeline(model, test_loader, device)

    print("-" * 65)
    print(f"TEST SET BENCHMARK RESULTS ({len(test_ds)} pairs):")
    print(f"  Precision       : {test_metrics['precision']:.2f}%")
    print(f"  Recall          : {test_metrics['recall']:.2f}%")
    print(f"  F1-Score        : {test_metrics['f1']:.2f}%")
    print(f"  Dustbin Accuracy: {test_metrics['dustbin_acc']:.2f}%")
    print(f"  Matches         : {test_metrics['tp']} TP, {test_metrics['fp']} FP, {test_metrics['fn']} FN")
    print("-" * 65)

    vis_dir = Path("outputs/object_matching_vis")
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for vis_idx, batch in enumerate(test_loader):
            if vis_idx >= 5:
                break
            img0, img1 = batch["img0"].to(device), batch["img1"].to(device)
            m0, b0 = batch["masks0"].to(device), batch["bboxes0"].to(device)
            m1, b1 = batch["masks1"].to(device), batch["bboxes1"].to(device)
            out = model(img0, img1, m0, b0, m1, b1)
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(out["prob_matrix"][0])
            save_file = vis_dir / f"test_sample_{vis_idx + 1}_{batch['scene_name']}_v{batch['view_i']}_v{batch['view_j']}.jpg"
            visualize_object_matches(
                batch["raw_img0"], batch["raw_img1"], batch["objs0"], batch["objs1"],
                pred_matches, batch["gt_matches"][0], pred_un0, pred_un1, save_file,
                title=f"Sample {vis_idx + 1}: {batch['scene_name']} (View {batch['view_i']} <-> View {batch['view_j']})",
            )


def evaluate_object_pipeline(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    total_dustbin_correct, total_dustbin_items = 0, 0

    with torch.no_grad():
        for batch in loader:
            img0, img1 = batch["img0"].to(device), batch["img1"].to(device)
            m0, b0 = batch["masks0"].to(device), batch["bboxes0"].to(device)
            m1, b1 = batch["masks1"].to(device), batch["bboxes1"].to(device)

            out = model(img0, img1, m0, b0, m1, b1)
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(out["prob_matrix"][0])

            gt_matches = set(batch["gt_matches"][0])
            pred_set = set(pred_matches)

            total_tp += len(pred_set.intersection(gt_matches))
            total_fp += len(pred_set - gt_matches)
            total_fn += len(gt_matches - pred_set)

            gt_un0, gt_un1 = set(batch["unmatched_0"][0]), set(batch["unmatched_1"][0])
            total_dustbin_correct += len(set(pred_un0).intersection(gt_un0)) + len(set(pred_un1).intersection(gt_un1))
            total_dustbin_items += len(gt_un0) + len(gt_un1)

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
# Line Matching Training Pipeline
# ==============================================================================
def train_line_pipeline(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir or "checkpoints/linefeat")
    save_dir.mkdir(parents=True, exist_ok=True)
    lr = args.lr if args.lr is not None else 1e-4
    device = torch.device(args.device)

    print("=================================================================")
    print(f"      Unified Trainer: Line Matching ({args.model_type})         ")
    print("=================================================================")
    print(f"  Device:           {device}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning Rate:    {lr}")
    print(f"  Checkpoint Dir:   {save_dir}")
    print("=================================================================\n")

    train_dataset = LineFeatPairDataset(
        index_path="data/line_gt_dataset/index.json",
        split="train",
        min_matches=args.min_matches,
        load_images=True,
    )
    val_dataset = LineFeatPairDataset(
        index_path="data/line_gt_dataset/index.json",
        split="val",
        min_matches=args.min_matches,
        load_images=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = build_model(
        args.model_type,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)

    criterion = build_loss(args.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, sample in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            gt_lines0 = sample["gt_lines0"][0].to(device)
            gt_lines1 = sample["gt_lines1"][0].to(device)
            M = len(gt_lines0)
            if M < 2:
                continue

            img0 = (sample["image0"].to(device) - mean) / std
            img1 = (sample["image1"].to(device) - mean) / std
            lines0 = gt_lines0.unsqueeze(0)
            lines1 = gt_lines1.unsqueeze(0)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(img0, img1, lines0, lines1)
                loss, metrics = criterion(out["log_assignment"], num_matches=M)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss += metrics["loss"]
            step_count += 1

        print(f"Epoch {epoch:02d}/{args.epochs:02d} - Train Loss: {epoch_loss/max(step_count, 1):.4f}")


def main() -> None:
    args = parse_args()
    if args.model_type in ["xfeat_object", "xfeat"]:
        train_object_pipeline(args)
    else:
        train_line_pipeline(args)


if __name__ == "__main__":
    main()
