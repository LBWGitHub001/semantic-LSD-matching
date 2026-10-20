"""
End-to-End Training & Evaluation on Multi-View Synthetic 3D Scenes.

Trains MaskLineFeatMatchNet on the procedurally generated multi-view dataset:
  - 80 Scenes for Training (~2,240 cross-view line pairs)
  - 10 Scenes for Validation (280 cross-view line pairs)
  - 10 Scenes for Final Testing (280 cross-view line pairs)

Features:
  - RAM preloading (reads all 800 images into memory in ~2s, 0 disk I/O bottleneck)
  - MaskedMatrixLoss supervision (Sinkhorn NLL + Masked BCE on active 255x255 subregion)
  - Automatic Mixed Precision (AMP FP16)
  - Comprehensive rotation-bracket breakdown on the test set
  - Visual result exports

Usage:
  # Quick training on GPU:
  python scripts/train_synthetic_multiview.py --epochs 10 --amp

  # Custom parameters:
  python scripts/train_synthetic_multiview.py --epochs 15 --lr 2e-4 --d-model 128 --num-layers 4 --amp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import MaskLineFeatMatchNet, MaskedMatrixLoss, LineMatcher


# ==============================================================================
# 1. Multi-View Graph PyTorch Dataset
# ==============================================================================
class SyntheticMultiViewDataset(Dataset):
    """
    Parses scenes from data/synthetic_multiview_scenes/ and extracts view graph edges.

    Splits:
      - 'train': scenes 0000 ~ 0079 (80 scenes)
      - 'val':   scenes 0080 ~ 0089 (10 scenes)
      - 'test':  scenes 0090 ~ 0099 (10 scenes)
    """

    def __init__(
        self,
        dataset_root: str | Path = ROOT / "data" / "synthetic_multiview_scenes",
        split: str = "train",
        min_matches: int = 4,
        preload_images: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(dataset_root)
        self.split = split
        self.preload_images = preload_images

        # 1. Discover and slice scenes according to 80 / 10 / 10 split
        scene_dirs = sorted([d for d in self.root.iterdir() if d.is_dir() and (d / "view_graph.json").exists()])
        total_scenes = len(scene_dirs)
        if total_scenes < 3:
            raise RuntimeError(f"Expected at least 3 scenes in {self.root}, found {total_scenes}!")

        # 80 / 10 / 10 partitioning
        train_end = min(80, int(total_scenes * 0.8))
        val_end = min(90, train_end + int(total_scenes * 0.1))

        if split == "train":
            target_dirs = scene_dirs[:train_end]
        elif split == "val":
            target_dirs = scene_dirs[train_end:val_end]
        elif split == "test":
            target_dirs = scene_dirs[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}. Expected 'train', 'val', or 'test'.")

        self.scene_dirs = target_dirs
        print(f"[*] Initializing '{split}' dataset ({len(target_dirs)} scenes)...")

        # 2. Flatten all view_graph edges into item tuples
        self.items: list[dict[str, Any]] = []
        self.image_cache: dict[str, np.ndarray] = {}

        t0 = time.time()
        for s_dir in target_dirs:
            graph_path = s_dir / "view_graph.json"
            with open(graph_path, "r", encoding="utf-8") as f:
                g = json.load(f)

            views_dir = s_dir / "views"
            for edge in g.get("edges", []):
                matches = edge.get("matches", [])
                if len(matches) < min_matches:
                    continue

                i, j = edge["view_i"], edge["view_j"]
                img0_path = views_dir / f"view_{i:02d}.png"
                img1_path = views_dir / f"view_{j:02d}.png"
                lines0_path = views_dir / f"lines_{i:02d}.json"
                lines1_path = views_dir / f"lines_{j:02d}.json"

                # Preload images into RAM if enabled
                if self.preload_images:
                    for p in (img0_path, img1_path):
                        key = str(p)
                        if key not in self.image_cache:
                            bgr = cv2.imread(key)
                            if bgr is not None:
                                self.image_cache[key] = bgr

                self.items.append({
                    "scene_id": s_dir.name,
                    "img0_path": str(img0_path),
                    "img1_path": str(img1_path),
                    "lines0_path": str(lines0_path),
                    "lines1_path": str(lines1_path),
                    "view_i": i,
                    "view_j": j,
                    "rotation_deg": edge.get("rotation_deg", 0.0),
                    "baseline_m": edge.get("baseline_meters", 0.0),
                    "matches": [(m["idx_i"], m["idx_j"]) for m in matches],
                })

        print(f"    Loaded {len(self.items):,} image pairs across {len(target_dirs)} scenes ({time.time()-t0:.2f}s).")
        if self.preload_images:
            cache_mb = sum(img.nbytes for img in self.image_cache.values()) / (1024 * 1024)
            print(f"    RAM Image Cache: {len(self.image_cache)} images preloaded ({cache_mb:.1f} MB).")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]

        # Read or fetch images
        if self.preload_images and item["img0_path"] in self.image_cache:
            bgr0 = self.image_cache[item["img0_path"]].copy()
            bgr1 = self.image_cache[item["img1_path"]].copy()
        else:
            bgr0 = cv2.imread(item["img0_path"])
            bgr1 = cv2.imread(item["img1_path"])

        # Read line segment coordinates
        with open(item["lines0_path"], "r", encoding="utf-8") as f:
            lines0 = np.array(json.load(f), dtype=np.float32)

        with open(item["lines1_path"], "r", encoding="utf-8") as f:
            lines1 = np.array(json.load(f), dtype=np.float32)

        rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
        rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)

        # Convert to Tensor (1, 3, H, W) normalized [0, 1]
        t0 = torch.from_numpy(rgb0).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t1 = torch.from_numpy(rgb1).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        return {
            "image0": t0,
            "image1": t1,
            "lines0": torch.from_numpy(lines0).unsqueeze(0),
            "lines1": torch.from_numpy(lines1).unsqueeze(0),
            "gt_pairs": item["matches"],
            "rotation_deg": item["rotation_deg"],
            "baseline_m": item["baseline_m"],
            "bgr0": bgr0,
            "bgr1": bgr1,
        }


# ==============================================================================
# 2. Evaluation Functions
# ==============================================================================
def evaluate(
    model: nn.Module,
    criterion: MaskedMatrixLoss,
    dataloader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    amp: bool = True,
    max_eval_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_gt = 0
    count = 0

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if max_eval_samples is not None and i >= max_eval_samples:
                break

            img0 = (sample["image0"].to(device) - mean) / std
            img1 = (sample["image1"].to(device) - mean) / std
            lines0 = sample["lines0"].to(device)
            lines1 = sample["lines1"].to(device)
            gt_pairs = sample["gt_pairs"]
            N0, N1 = lines0.shape[1], lines1.shape[1]

            with torch.amp.autocast("cuda", enabled=amp):
                out = model(img0, img1, lines0, lines1)
                loss, metrics = criterion(out["matrix_255"], out["log_assignment"], gt_pairs, N0, N1)

            total_loss += metrics["loss"]
            total_correct += metrics["correct"]
            total_gt += len(gt_pairs)
            count += 1

    avg_loss = total_loss / max(count, 1)
    acc = (total_correct / max(total_gt, 1)) * 100.0
    return {
        "val_loss": avg_loss,
        "val_acc": acc,
        "correct": total_correct,
        "total_gt": total_gt,
        "pairs_evaluated": count,
    }


def run_test_benchmark(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_dir: Path,
) -> None:
    print("\n" + "=" * 80)
    print("      Final Benchmark on Test Set (10 Unseen Scenes, 280 Pairs)")
    print("=" * 80)
    model.eval()

    brackets = {
        "Small Rotation (< 30°)": {"correct": 0, "top1": 0, "total": 0},
        "Medium Rotation (30° ~ 60°)": {"correct": 0, "top1": 0, "total": 0},
        "Large Rotation (> 60°)": {"correct": 0, "top1": 0, "total": 0},
    }

    total_pred = 0
    total_correct = 0
    total_top1_correct = 0
    total_gt = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            img0 = (sample["image0"].to(device) - mean) / std
            img1 = (sample["image1"].to(device) - mean) / std
            lines0 = sample["lines0"].to(device)
            lines1 = sample["lines1"].to(device)
            gt_pairs = sample["gt_pairs"]
            rot = float(sample["rotation_deg"])
            N0, N1 = lines0.shape[1], lines1.shape[1]

            out = model(img0, img1, lines0, lines1)
            matrix_255 = out["matrix_255"][0].cpu().numpy()
            log_P = out["log_assignment"][0].cpu().numpy()

            # Predict matches via Mutual Nearest Neighbor + argmax
            P_sub = log_P[:N0, :N1]
            best1 = np.argmax(P_sub, axis=1)
            best0 = np.argmax(P_sub, axis=0)

            gt_dict = {i0: i1 for i0, i1 in gt_pairs}
            pred_matches = []

            for idx0 in range(N0):
                idx1 = int(best1[idx0])
                if int(best0[idx1]) == idx0:
                    prob = float(matrix_255[idx0, idx1])
                    if prob > 1e-4:  # Valid non-zero probability
                        pred_matches.append((idx0, idx1, prob))

            correct_in_pair = 0
            for idx0, idx1, _ in pred_matches:
                if gt_dict.get(idx0) == idx1:
                    correct_in_pair += 1

            top1_in_pair = sum(1 for i0, i1 in gt_pairs if int(best1[i0]) == i1)

            total_pred += len(pred_matches)
            total_correct += correct_in_pair
            total_top1_correct += top1_in_pair
            total_gt += len(gt_pairs)

            # Record into rotation bracket
            if rot < 30.0:
                b_name = "Small Rotation (< 30°)"
            elif rot <= 60.0:
                b_name = "Medium Rotation (30° ~ 60°)"
            else:
                b_name = "Large Rotation (> 60°)"

            brackets[b_name]["correct"] += correct_in_pair
            brackets[b_name]["top1"] += top1_in_pair
            brackets[b_name]["total"] += len(gt_pairs)

            # Export sample visualizations for first 4 test pairs
            if vis_count < 4:
                bgr0 = sample["bgr0"]
                bgr1 = sample["bgr1"]
                l0_np = lines0[0].cpu().numpy()
                l1_np = lines1[0].cpu().numpy()
                vis_img = LineMatcher.draw_matches(bgr0, bgr1, l0_np, l1_np, pred_matches, max_draw=len(pred_matches))
                vis_path = out_dir / f"test_sample_{vis_count+1}_rot_{rot:.1f}deg.jpg"
                cv2.imwrite(str(vis_path), vis_img)
                vis_count += 1

    overall_top1_acc = (total_top1_correct / max(total_gt, 1)) * 100.0
    overall_prec = (total_correct / max(total_pred, 1)) * 100.0
    overall_recall = (total_correct / max(total_gt, 1)) * 100.0

    print(f"\n[*] Overall Test Set Performance (10 Scenes, 280 Pairs):")
    print(f"    - Total Ground Truth Lines:          {total_gt:,}")
    print(f"    - Top-1 Match Accuracy:              {overall_top1_acc:.2f}% ({total_top1_correct:,}/{total_gt:,})")
    print(f"    - Mutual Nearest Neighbor Precision: {overall_prec:.2f}% ({total_correct:,}/{total_pred:,})")
    print(f"    - Mutual Nearest Neighbor Recall:    {overall_recall:.2f}% ({total_correct:,}/{total_gt:,})")

    print("\n[*] Breakdown by Camera Rotation Difficulty:")
    for b_name, stats in brackets.items():
        b_acc = (stats["top1"] / max(stats["total"], 1)) * 100.0
        b_prec = (stats["correct"] / max(stats["top1"], 1)) * 100.0
        print(f"    • {b_name:28s}: Top-1 {stats['top1']:4d} / {stats['total']:4d} ({b_acc:5.1f}%) | MNN {stats['correct']:4d}")

    print(f"\n[+] Exported sample visual matching results to: {out_dir}")
    print("=" * 80)


# ==============================================================================
# 3. Main Training Script
# ==============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskLineFeatMatchNet on Synthetic Multi-View Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--d-model", type=int, default=128, help="Feature dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=4, help="Transformer layers (default: 4)")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--lambda-bce", type=float, default=1.0, help="Weight of Masked BCE loss (default: 1.0)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision FP16")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "checkpoints" / "synthetic_masknet")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "synthetic_multiview_scenes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 80)
    print("      MaskLineFeatMatchNet Training on Multi-View 3D Scenes")
    print("=" * 80)
    print(f"[*] Hyperparameters:")
    print(f"    - Device:          {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"    - Epochs:          {args.epochs}")
    print(f"    - Learning Rate:   {args.lr}")
    print(f"    - d_model:         {args.d_model}, layers: {args.num_layers}, nhead: {args.nhead}")
    print(f"    - AMP FP16:        {args.amp}")
    print(f"    - Grad Accum:      {args.grad_accum}")
    print(f"    - Output Weights:  {args.save_dir}")
    print("=" * 80)

    # 1. Datasets & DataLoaders
    train_dataset = SyntheticMultiViewDataset(args.data_root, split="train", preload_images=True)
    val_dataset = SyntheticMultiViewDataset(args.data_root, split="val", preload_images=True)
    test_dataset = SyntheticMultiViewDataset(args.data_root, split="test", preload_images=True)

    # Custom collate for batch_size=1 (native variable line counts)
    def collate_single(batch):
        return batch[0]

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_single)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_single)

    # 2. Model & Loss
    print("\n[2/4] Initializing MaskLineFeatMatchNet...")
    model = MaskLineFeatMatchNet(
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        max_matrix_size=255,
        freeze_backbone_stages=1,
    ).to(device)

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"      Trainable Parameters: {trainable_p:,} ({trainable_p/1e6:.2f} M)")
    print(f"      Total Parameters:     {total_p:,} ({total_p/1e6:.2f} M)")

    criterion = MaskedMatrixLoss(lambda_bce=args.lambda_bce)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_weights_path = args.save_dir / "best_model.pth"

    # 3. Training Loop
    print("\n[3/4] Starting Multi-View Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_gt = 0
        optimizer.zero_grad()

        start_time = time.time()
        for step, sample in enumerate(train_loader):
            img0 = (sample["image0"].to(device) - mean) / std
            img1 = (sample["image1"].to(device) - mean) / std
            lines0 = sample["lines0"].to(device)
            lines1 = sample["lines1"].to(device)
            gt_pairs = sample["gt_pairs"]
            N0, N1 = lines0.shape[1], lines1.shape[1]

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(img0, img1, lines0, lines1)
                loss, metrics = criterion(out["matrix_255"], out["log_assignment"], gt_pairs, N0, N1)
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()

            epoch_loss += metrics["loss"]
            epoch_correct += metrics["correct"]
            epoch_gt += len(gt_pairs)

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (step + 1) % 500 == 0:
                cur_acc = (epoch_correct / max(epoch_gt, 1)) * 100.0
                print(f"      Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step+1:4d}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Running Train Acc: {cur_acc:.1f}%")

        scheduler.step()
        train_acc = (epoch_correct / max(epoch_gt, 1)) * 100.0
        epoch_sec = time.time() - start_time

        # Validation on the 10 Validation scenes (280 pairs)
        val_metrics = evaluate(
            model=model,
            criterion=criterion,
            dataloader=val_loader,
            device=device,
            mean=mean,
            std=std,
            amp=args.amp,
        )

        is_best = val_metrics["val_acc"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["val_acc"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": best_val_acc,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
            }, best_weights_path)
            tag = "★ BEST"
        else:
            tag = "      "

        print(f"\n[*] Epoch [{epoch:02d}/{args.epochs:02d}] Finished in {epoch_sec:.1f}s ({len(train_loader)/epoch_sec:.1f} pairs/s)")
        print(f"    Train Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"    Val   Loss: {val_metrics['val_loss']:.4f} | Val   Acc: {val_metrics['val_acc']:.2f}% [{tag}]")
        print("-" * 75)

    # 4. Final Benchmark on Test Set
    print("\n[4/4] Loading Best Model for Final Test Set Evaluation...")
    best_ckpt = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    run_test_benchmark(
        model=model,
        test_loader=test_loader,
        device=device,
        mean=mean,
        std=std,
        out_dir=ROOT / "outputs" / "synthetic_multiview_eval",
    )


if __name__ == "__main__":
    main()
