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
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback generator if tqdm is missing
    def tqdm(iterable, *args, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_loss, build_model
from datasets import (
    ObjectMatchingDataset,
    collate_single_pair,
    predict_bipartite_matches,
    visualize_object_matches,
    visualize_semantic_region_matches,
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
    parser.add_argument("--save-interval", type=int, default=5, help="Save historical checkpoint every N epochs (e.g. epoch_005.pth, default: 5)")
    parser.add_argument("--no-save-last", dest="save_last", action="store_false", default=True, help="Disable saving last_model.pth")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    parser.add_argument("--dustbin-weight", type=float, default=0.3, help="Dustbin rejection loss weight for ObjectMatchingLoss (default: 0.3)")
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
    print(f"  Dustbin Weight:     {args.dustbin_weight}")
    print(f"  Model Dimension:    {args.d_model} (Layers: {args.num_layers}, Heads: {args.nhead})")
    print(f"  Mixed Precision:    {'Enabled (AMP FP16)' if args.amp else 'Disabled (FP32)'}")
    print(f"  Checkpoint Dir:     {save_dir}")
    print(f"  Save Interval:      Every {args.save_interval} epoch(s)")
    if args.resume:
        print(f"  Resume From:        {args.resume}")
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

    loss_fn = build_loss(args.model_type, dustbin_weight=args.dustbin_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch = 1
    best_val_f1 = 0.0
    best_ckpt_path = save_dir / "best_model.pth"
    last_ckpt_path = save_dir / "last_model.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume not found: {resume_path}")
        print(f"[*] Resuming object training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"    [!] Warning: Failed to load optimizer state: {e}")
        if args.amp and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"    [!] Warning: Failed to load scaler state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", ckpt.get("val_metrics", {}).get("f1", 0.0))
        print(f"    Resumed at epoch {start_epoch}, previous best Val F1: {best_val_f1:.2f}%\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        t0 = time.time()

        optimizer.zero_grad()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]",
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )

        for step, batch in enumerate(pbar):
            if args.max_steps is not None and step >= args.max_steps:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(patches0, patches1)
                loss = loss_fn(out["log_P"], batch["gt_matches"], batch["unmatched_0"], batch["unmatched_1"])
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            step_count += 1

            if step_count % args.grad_accum == 0 or step == len(train_loader) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = float(loss.item())
            total_loss += loss_val
            cur_avg_loss = total_loss / max(step_count, 1)

            # Real-time status in progress bar
            m0_num = patches0.shape[1]
            m1_num = patches1.shape[1]
            n_gt = len(batch["gt_matches"][0]) if "gt_matches" in batch and batch["gt_matches"] else 0
            mem_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "avg": f"{cur_avg_loss:.4f}",
                "patches": f"{m0_num}v{m1_num}",
                "gt": n_gt,
                "gpu": f"{mem_mb:.0f}MB",
            })

        train_loss = total_loss / max(step_count, 1)
        epoch_sec = time.time() - t0

        val_metrics = None
        is_best = False
        if epoch % args.val_interval == 0:
            val_metrics = evaluate_object_pipeline(model, val_loader, device, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]")
            val_f1 = val_metrics["f1"]
            is_best = val_f1 >= best_val_f1
            if is_best:
                best_val_f1 = val_f1

            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) "
                  f"| Loss: {train_loss:.4f} "
                  f"| Val Prec: {val_metrics['precision']:5.1f}% | Rec: {val_metrics['recall']:5.1f}% "
                  f"| F1: {val_f1:5.1f}% | Dustbin: P={val_metrics['dustbin_prec']:4.1f}%/R={val_metrics['dustbin_rec']:4.1f}%{star}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | Loss: {train_loss:.4f}")

        # Checkpoint Management
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if args.amp else None,
            "best_val_f1": best_val_f1,
            "val_metrics": val_metrics,
            "train_loss": train_loss,
            "args": vars(args),
        }

        # 1. Update best checkpoint
        if is_best:
            torch.save(ckpt_data, best_ckpt_path)

        # 2. Update latest checkpoint
        if args.save_last:
            torch.save(ckpt_data, last_ckpt_path)

        # 3. Save periodic historical checkpoint
        if args.save_interval > 0 and (epoch % args.save_interval == 0 or epoch == args.epochs):
            epoch_ckpt_path = save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(ckpt_data, epoch_ckpt_path)
            print(f"  [Checkpoint] Saved historical checkpoint -> {epoch_ckpt_path.name}")

    print("=" * 80)
    print(f"[+] Training Complete! Best Val F1: {best_val_f1:.2f}%.")
    eval_ckpt_path = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    if not eval_ckpt_path.exists():
        print("[!] No checkpoint found to evaluate.")
        return

    # Benchmark test set & save visualizations
    print(f"\nEvaluating checkpoint on unseen test scenes ({eval_ckpt_path.name})...")
    ckpt = torch.load(eval_ckpt_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)

    # 1. Unconstrained visual matching evaluation
    raw_test_metrics = evaluate_object_pipeline(model, test_loader, device, desc="Testing (Raw Visual)")

    # 2. Category-constrained evaluation (if class labels available)
    cat_test_metrics = evaluate_object_pipeline(model, test_loader, device, desc="Testing (Semantic Constrained)", enforce_same_category=True)

    print("-" * 75)
    print(f"TEST SET BENCHMARK RESULTS ({len(test_ds)} pairs):")
    print("  [1] Raw Visual Appearance Matching (Zero Class Prior):")
    print(f"      Precision          : {raw_test_metrics['precision']:.2f}%")
    print(f"      Recall             : {raw_test_metrics['recall']:.2f}%")
    print(f"      F1-Score           : {raw_test_metrics['f1']:.2f}%")
    print(f"      Matches            : {raw_test_metrics['tp']} TP, {raw_test_metrics['fp']} FP, {raw_test_metrics['fn']} FN")
    print(f"      Cross-Category FP  : {raw_test_metrics['cross_cat_fp']} ({raw_test_metrics['cross_cat_rate']:.1f}% of all FPs)")
    print(f"      Dustbin Precision  : {raw_test_metrics['dustbin_prec']:.2f}% (Recall: {raw_test_metrics['dustbin_rec']:.2f}%)")
    print("  [2] Semantic Category Constrained Matching:")
    print(f"      Precision          : {cat_test_metrics['precision']:.2f}%")
    print(f"      Recall             : {cat_test_metrics['recall']:.2f}%")
    print(f"      F1-Score           : {cat_test_metrics['f1']:.2f}%")
    print(f"      Matches            : {cat_test_metrics['tp']} TP, {cat_test_metrics['fp']} FP, {cat_test_metrics['fn']} FN")
    print("-" * 75)

    vis_dir = Path("outputs/object_matching_vis")
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for vis_idx, batch in enumerate(test_loader):
            if vis_idx >= 5:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            out = model(patches0, patches1)
            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(out["prob_matrix"][0])
            scene_tag = batch.get("scene_name", f"pair_{vis_idx+1}")
            save_file = vis_dir / f"test_sample_{vis_idx + 1}_{scene_tag}.jpg"

            if use_region_ds and "raw_img0" in batch and batch["raw_img0"] is not None:
                visualize_semantic_region_matches(
                    bgr0=batch["raw_img0"],
                    bgr1=batch["raw_img1"],
                    masks0=batch["masks0"],
                    masks1=batch["masks1"],
                    regs0=batch.get("regs0", []),
                    regs1=batch.get("regs1", []),
                    pred_matches=pred_matches,
                    gt_matches=batch["gt_matches"][0],
                    pred_un0=pred_un0,
                    pred_un1=pred_un1,
                    out_path=save_file,
                    title=f"Sample {vis_idx + 1}: {scene_tag}",
                )
            elif "raw_img0" in batch and batch["raw_img0"] is not None:
                visualize_object_matches(
                    batch["raw_img0"], batch["raw_img1"], batch["objs0"], batch["objs1"],
                    pred_matches, batch["gt_matches"][0], pred_un0, pred_un1, save_file,
                    title=f"Sample {vis_idx + 1}: {scene_tag}",
                )


def evaluate_object_pipeline(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
    enforce_same_category: bool = False,
) -> dict[str, float]:
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    cross_cat_fp = 0
    same_cat_fp = 0
    dustbin_tp, dustbin_fp, dustbin_fn, dustbin_tn = 0, 0, 0, 0

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=False, unit="pair")
    with torch.no_grad():
        for batch in pbar:
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)

            out = model(patches0, patches1)

            cats0 = [o.get("category", "") for o in batch.get("objs0", [])] if enforce_same_category else None
            cats1 = [o.get("category", "") for o in batch.get("objs1", [])] if enforce_same_category else None

            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                out["prob_matrix"][0], thresh=0.2, categories_0=cats0, categories_1=cats1
            )

            gt_matches = set(batch["gt_matches"][0])
            pred_set = set(pred_matches)

            cur_tp = len(pred_set.intersection(gt_matches))
            cur_fp_set = pred_set - gt_matches
            cur_fn = len(gt_matches - pred_set)

            total_tp += cur_tp
            total_fp += len(cur_fp_set)
            total_fn += cur_fn

            objs0 = batch.get("objs0", [])
            objs1 = batch.get("objs1", [])
            for i, j in cur_fp_set:
                c0 = objs0[i].get("category", "") if i < len(objs0) else ""
                c1 = objs1[j].get("category", "") if j < len(objs1) else ""
                if c0 and c1 and c0 != c1:
                    cross_cat_fp += 1
                else:
                    same_cat_fp += 1

            # Dustbin 4-class confusion matrix
            gt_un0 = set(batch["unmatched_0"][0])
            gt_un1 = set(batch["unmatched_1"][0])
            pred_un0_set = set(pred_un0)
            pred_un1_set = set(pred_un1)

            m0_num = patches0.shape[1]
            m1_num = patches1.shape[1]

            for i in range(m0_num):
                is_un_gt = (i in gt_un0)
                is_un_pred = (i in pred_un0_set)
                if is_un_gt and is_un_pred:
                    dustbin_tp += 1
                elif not is_un_gt and is_un_pred:
                    dustbin_fp += 1
                elif is_un_gt and not is_un_pred:
                    dustbin_fn += 1
                else:
                    dustbin_tn += 1

            for j in range(m1_num):
                is_un_gt = (j in gt_un1)
                is_un_pred = (j in pred_un1_set)
                if is_un_gt and is_un_pred:
                    dustbin_tp += 1
                elif not is_un_gt and is_un_pred:
                    dustbin_fp += 1
                elif is_un_gt and not is_un_pred:
                    dustbin_fn += 1
                else:
                    dustbin_tn += 1

            cur_p = total_tp / (total_tp + total_fp + 1e-6)
            cur_r = total_tp / (total_tp + total_fn + 1e-6)
            cur_f1 = 2.0 * cur_p * cur_r / (cur_p + cur_r + 1e-6)
            pbar.set_postfix({"f1": f"{cur_f1*100:.1f}%", "tp": total_tp, "fp": total_fp, "fn": total_fn})

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    dustbin_prec = dustbin_tp / (dustbin_tp + dustbin_fp + 1e-6)
    dustbin_rec = dustbin_tp / (dustbin_tp + dustbin_fn + 1e-6)
    dustbin_acc = (dustbin_tp + dustbin_tn) / (dustbin_tp + dustbin_fp + dustbin_fn + dustbin_tn + 1e-6)
    cross_cat_rate = cross_cat_fp / (total_fp + 1e-6)

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "dustbin_prec": dustbin_prec * 100.0,
        "dustbin_rec": dustbin_rec * 100.0,
        "dustbin_acc": dustbin_acc * 100.0,
        "cross_cat_fp": cross_cat_fp,
        "cross_cat_rate": cross_cat_rate * 100.0,
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
    print(f"  Save Interval:    Every {args.save_interval} epoch(s)")
    if args.resume:
        print(f"  Resume From:      {args.resume}")
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

    start_epoch = 1
    best_val_acc = 0.0
    best_ckpt_path = save_dir / "best_model.pth"
    last_ckpt_path = save_dir / "last_model.pth"

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume not found: {resume_path}")
        print(f"[*] Resuming line training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                print(f"    [!] Warning: Failed to load optimizer state: {e}")
        if args.amp and "scaler_state" in ckpt and ckpt["scaler_state"] is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                print(f"    [!] Warning: Failed to load scaler state: {e}")
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"    Resumed at epoch {start_epoch}, previous best Val Acc: {best_val_acc:.2f}%\n")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [LineTrain]",
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )

        for step, sample in enumerate(pbar):
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

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "avg": f"{epoch_loss/max(step_count, 1):.4f}",
                "lines": f"{M}",
                "acc": f"{metrics.get('acc', 0.0):.1f}%",
            })

        epoch_sec = time.time() - t0
        train_loss = epoch_loss / max(step_count, 1)

        # Validation
        val_acc = 0.0
        val_loss = 0.0
        val_steps = 0
        is_best = False
        if epoch % args.val_interval == 0 and len(val_loader) > 0:
            model.eval()
            with torch.no_grad():
                for sample in val_loader:
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
                        loss_v, metrics_v = criterion(out["log_assignment"], num_matches=M)
                    val_loss += metrics_v["loss"]
                    val_acc += metrics_v.get("acc", 0.0)
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            avg_val_acc = val_acc / max(val_steps, 1)
            is_best = avg_val_acc >= best_val_acc
            if is_best:
                best_val_acc = avg_val_acc

            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) "
                  f"| Train Loss: {train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.1f}%{star}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | Train Loss: {train_loss:.4f}")

        # Checkpoint Management
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if args.amp else None,
            "best_val_acc": best_val_acc,
            "train_loss": train_loss,
            "args": vars(args),
        }

        # 1. Update best checkpoint
        if is_best:
            torch.save(ckpt_data, best_ckpt_path)

        # 2. Update latest checkpoint
        if args.save_last:
            torch.save(ckpt_data, last_ckpt_path)

        # 3. Save periodic historical checkpoint
        if args.save_interval > 0 and (epoch % args.save_interval == 0 or epoch == args.epochs):
            epoch_ckpt_path = save_dir / f"epoch_{epoch:03d}.pth"
            torch.save(ckpt_data, epoch_ckpt_path)
            print(f"  [Checkpoint] Saved historical checkpoint -> {epoch_ckpt_path.name}")


def main() -> None:
    args = parse_args()
    if args.model_type in ["xfeat_object", "xfeat"]:
        train_object_pipeline(args)
    else:
        train_line_pipeline(args)


if __name__ == "__main__":
    main()
