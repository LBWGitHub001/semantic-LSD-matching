"""
LineFeat Training Script.

Trains the Line-Transformer matching network on Structured3D 3D-to-2D wireframe ground truth.

Supports:
  - Standard training from images (default)
  - Blazing fast training from precomputed feature cache (--use-cache, ~50x speedup)
  - Automatic Mixed Precision (--amp)
  - Multi-worker prefetching (--num-workers)

Usage:
  # Fast Cached Mode (10 epochs in ~2 minutes):
  python tools/cache_line_features.py
  python train_linefeat.py --use-cache --amp --epochs 10

  # Standard Mode:
  python train_linefeat.py --epochs 10 --amp --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import LineFeatPairDataset, LineMatcherNet, LineMatchingLoss


class CachedLineFeatureDataset(Dataset):
    """In-memory dataset of precomputed visual line tokens for blazing fast training."""

    def __init__(self, cache_file: str | Path) -> None:
        super().__init__()
        path = Path(cache_file)
        if not path.is_file():
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                f"Please run 'python tools/cache_line_features.py' first!"
            )
        print(f"[*] Loading cached line features from: {path.name}...")
        self.data = torch.load(path)
        print(f"    Loaded {len(self.data)} pairs into RAM.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        return self.data[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LineFeat Line-Transformer Matcher")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--min-matches", type=int, default=3, help="Minimum GT matches per pair")
    parser.add_argument("--d-model", type=int, default=128, help="Feature dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (FP16)")
    parser.add_argument("--use-cache", action="store_true", help="Train from precomputed feature cache for 50x speedup")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "line_feature_cache")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints" / "linefeat", help="Checkpoint directory")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per epoch for quick dry-run")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(
    model: LineMatcherNet,
    criterion: LineMatchingLoss,
    dataloader: DataLoader,
    device: torch.device,
    use_cache: bool = False,
    amp: bool = False,
    max_samples: int = 100,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_gt = 0
    count = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if count >= max_samples:
                break

            with torch.cuda.amp.autocast(enabled=amp):
                if use_cache:
                    vis0 = sample["vis0"].to(device)  # Already (1, M, D)
                    vis1 = sample["vis1"].to(device)
                    lines0 = sample["gt_lines0"].to(device)  # Already (1, M, 4)
                    lines1 = sample["gt_lines1"].to(device)
                    M = int(sample["num_matches"])
                    H = int(sample["H"])
                    W = int(sample["W"])
                    out = model.forward_from_vis_tokens(vis0, vis1, lines0, lines1, image_size=(H, W))
                else:
                    gt_lines0 = sample["gt_lines0"][0].to(device)  # (M, 4)
                    gt_lines1 = sample["gt_lines1"][0].to(device)
                    M = len(gt_lines0)
                    if M < 2:
                        continue

                    img0 = (sample["image0"].to(device) - mean) / std  # (1, 3, H, W)
                    img1 = (sample["image1"].to(device) - mean) / std
                    lines0 = gt_lines0.unsqueeze(0)  # (1, M, 4)
                    lines1 = gt_lines1.unsqueeze(0)
                    out = model(img0, img1, lines0, lines1)

                loss, metrics = criterion(out["log_assignment"], num_matches=M)

            total_loss += metrics["loss"]
            total_correct += metrics["correct"]
            total_gt += metrics["total_gt"]
            count += 1

    avg_loss = total_loss / max(count, 1)
    overall_acc = (total_correct / max(total_gt, 1)) * 100.0
    return {
        "val_loss": avg_loss,
        "val_acc": overall_acc,
        "val_pairs": count,
        "val_gt_matches": total_gt,
    }


def train() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=================================================================")
    print("               LineFeat Line-Transformer Training               ")
    print("=================================================================")
    print(f"  Device:           {device}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Model Dimension:  {args.d_model} (Layers: {args.num_layers})")
    print(f"  Gradient Accum:   {args.grad_accum}")
    print(f"  Mixed Precision:  {'Enabled (AMP FP16)' if args.amp else 'Disabled (FP32)'}")
    print(f"  Training Mode:    {'Fast Cached Mode (~50x speedup)' if args.use_cache else 'Standard Image Mode'}")
    print(f"  Checkpoint Dir:   {args.save_dir}")
    print("=================================================================\n")

    # 1. Dataset & Dataloaders
    print("[1/4] Loading Datasets...")
    if args.use_cache:
        train_cache_file = args.cache_dir / "train_cache.pt"
        val_cache_file = args.cache_dir / "val_cache.pt"
        train_dataset = CachedLineFeatureDataset(train_cache_file)
        val_dataset = CachedLineFeatureDataset(val_cache_file)
        # In-memory datasets don't need multiple worker processes
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        train_dataset = LineFeatPairDataset(
            index_path="data/line_gt_dataset/index.json",
            split="train",
            min_matches=args.min_matches,
            load_images=True,
            load_cameras=False,
        )
        val_dataset = LineFeatPairDataset(
            index_path="data/line_gt_dataset/index.json",
            split="val",
            min_matches=args.min_matches,
            load_images=True,
            load_cameras=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print(f"      Training Pairs:   {len(train_dataset)}")
    print(f"      Validation Pairs: {len(val_dataset)}")

    # 2. Model & Loss
    print("\n[2/4] Initializing LineMatcherNet...")
    model = LineMatcherNet(
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=4,
        freeze_backbone=True,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    print(f"      Total Parameters:     {total_params:,} ({total_params/1e6:.2f} M)")

    criterion = LineMatchingLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 3. Training Loop
    print("\n[3/4] Starting Training Loop...")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_gt = 0
        step_count = 0
        optimizer.zero_grad()

        start_time = time.time()
        for step, sample in enumerate(train_loader):
            if args.max_steps is not None and step >= args.max_steps:
                break

            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.use_cache:
                    vis0 = sample["vis0"].to(device)  # Already (1, M, D)
                    vis1 = sample["vis1"].to(device)
                    lines0 = sample["gt_lines0"].to(device)  # Already (1, M, 4)
                    lines1 = sample["gt_lines1"].to(device)
                    M = int(sample["num_matches"])
                    H = int(sample["H"])
                    W = int(sample["W"])
                    out = model.forward_from_vis_tokens(vis0, vis1, lines0, lines1, image_size=(H, W))
                else:
                    gt_lines0 = sample["gt_lines0"][0].to(device)  # (M, 4)
                    gt_lines1 = sample["gt_lines1"][0].to(device)
                    M = len(gt_lines0)
                    if M < 2:
                        continue

                    img0 = (sample["image0"].to(device) - mean) / std  # (1, 3, H, W)
                    img1 = (sample["image1"].to(device) - mean) / std
                    lines0 = gt_lines0.unsqueeze(0)  # (1, M, 4)
                    lines1 = gt_lines1.unsqueeze(0)
                    out = model(img0, img1, lines0, lines1)

                loss, metrics = criterion(out["log_assignment"], num_matches=M)
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()

            epoch_loss += metrics["loss"]
            epoch_correct += metrics["correct"]
            epoch_gt += metrics["total_gt"]
            step_count += 1

            if step_count % args.grad_accum == 0 or step == len(train_loader) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step_count % 100 == 0:
                current_acc = (epoch_correct / max(epoch_gt, 1)) * 100.0
                elapsed_now = time.time() - start_time
                speed = step_count / max(elapsed_now, 1e-4)
                print(f"      Epoch [{epoch}/{args.epochs}] Step {step_count:4d} | Loss: {metrics['loss']:.4f} | Running Acc: {current_acc:.1f}% ({speed:.1f} pairs/s)")

        scheduler.step()
        elapsed = time.time() - start_time
        avg_train_loss = epoch_loss / max(step_count, 1)
        train_acc = (epoch_correct / max(epoch_gt, 1)) * 100.0

        print(f"\n---> Epoch {epoch:02d} Summary [{elapsed:.1f}s]: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% ({step_count/elapsed:.1f} pairs/s)")

        # 4. Validation
        if epoch % args.val_interval == 0:
            print("     Running Validation...")
            val_results = evaluate(
                model,
                criterion,
                val_loader,
                device,
                use_cache=args.use_cache,
                amp=args.amp,
                max_samples=100,
            )
            val_loss = val_results["val_loss"]
            val_acc = val_results["val_acc"]
            print(f"     [Validation] Loss: {val_loss:.4f} | Accuracy: {val_acc:.1f}% ({val_results['val_gt_matches']} lines across {val_results['val_pairs']} pairs)")

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, args.save_dir / "latest_model.pth")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, args.save_dir / "best_model.pth")
                print(f"     ★ New Best Model Saved! (Acc: {best_val_acc:.1f}%)")

    print(f"\n[+] Training Complete! Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"    Checkpoints saved in: {args.save_dir}")


if __name__ == "__main__":
    train()
