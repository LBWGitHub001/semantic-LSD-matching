"""MaskLineFeatMatchNet Training Pipeline (Standalone & Modular)."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import apply_config_to_parser, load_config, pre_parse_config_path
from datasets import LineFeatPairDataset
from models.mask_line_matcher.mask_line_matcher import MaskLineFeatMatchNet
from models.mask_line_matcher.loss import LineMatchingLoss
from models.mask_line_matcher.eval import validate
from training_utils import (
    TQDM_BAR_FORMAT,
    create_experiment_dir,
    get_git_commit,
    save_config_json,
    update_latest_symlink,
    worker_init_fn,
)


def train(args: argparse.Namespace | None = None, **kwargs: Any) -> Path:
    """Trains MaskLineFeatMatchNet on LineFeatPairDataset."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    data_dir = Path(args.data_dir or "data/line_gt_dataset")
    base_save_dir = Path(getattr(args, "save_dir", None) or "checkpoints/mask_line")

    save_dir, run_name = create_experiment_dir(
        base_save_dir=base_save_dir,
        model_name=f"mask_line+{data_dir.name}",
        label=getattr(args, "label", None),
        no_timestamp=getattr(args, "no_timestamp", False),
        run_name=getattr(args, "run_name", None),
    )
    lr = args.lr if args.lr is not None else 2e-4
    device = torch.device(args.device)

    index_path = data_dir / "index.json"
    if not index_path.exists():
        for cand in ("data/line_gt_dataset/index.json", "data/synthetic_line_scenes/index.json"):
            if Path(cand).exists():
                index_path = Path(cand)
                break

    if not index_path.exists():
        print(f"[!] Notice: Line index file not found at {index_path}.")
        print("    Run 'python generate_dataset.py --config configs/mask_line.yaml' to generate it.")
        return save_dir

    train_ds = LineFeatPairDataset(index_path=str(index_path), split="train", min_matches=args.min_matches, load_images=True)
    val_ds = LineFeatPairDataset(index_path=str(index_path), split="val", min_matches=args.min_matches, load_images=True)

    orig_len = len(train_ds)
    if getattr(args, "data_ratio", 1.0) < 1.0 and orig_len > 0:
        ratio = max(0.001, min(1.0, float(args.data_ratio)))
        subset_size = max(1, int(orig_len * ratio))
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(orig_len, size=subset_size, replace=False).tolist()
        train_ds = torch.utils.data.Subset(train_ds, subset_indices)
        print(f"[*] Dataset subset: Sampled {len(train_ds)}/{orig_len} pairs.")

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = MaskLineFeatMatchNet(
        d_model=getattr(args, "d_model", 128),
        num_layers=getattr(args, "num_layers", 2),
    ).to(device)

    criterion = LineMatchingLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=getattr(args, "weight_decay", 1e-4))
    scaler = torch.amp.GradScaler("cuda", enabled=getattr(args, "amp", True), init_scale=2048.0)

    config_path = save_dir / "config.json"
    config_data = {
        "experiment_info": {
            "run_name": run_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
        },
        "model": {"model_type": "mask_line", "family": "line", "d_model": args.d_model, "num_layers": args.num_layers},
        "hyperparameters": {"epochs": args.epochs, "learning_rate": lr},
    }
    save_config_json(config_path, config_data)

    print(f"[*] MaskLineFeatMatchNet training | pairs={len(train_ds)} val={len(val_ds)} | device={device}")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        train_total = len(train_loader) if getattr(args, "max_steps", None) is None else min(len(train_loader), args.max_steps)
        pbar = tqdm(
            train_loader, total=train_total, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [LineTrain]",
            bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair",
        )
        for step, sample in enumerate(pbar):
            if getattr(args, "max_steps", None) is not None and step >= args.max_steps:
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
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            step_loss = metrics["loss"]
            if not math.isnan(step_loss) and not math.isinf(step_loss):
                epoch_loss += step_loss
            step_count += 1
            acc_val = metrics.get("acc", 0.0)
            pbar.set_postfix_str(f"loss={step_loss:6.4f}, avg={epoch_loss/max(step_count, 1):6.4f}, acc={acc_val:5.1f}%")

        train_loss = epoch_loss / max(step_count, 1)

        val_metrics = None
        is_best = False
        if epoch % getattr(args, "val_interval", 1) == 0 and len(val_loader) > 0:
            val_metrics = validate(model, criterion, val_loader, device, getattr(args, "max_steps", None))
            val_acc = val_metrics["acc"]
            is_best = val_acc >= best_val_acc
            if is_best:
                best_val_acc = val_acc
            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_acc:.1f}%{star}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train Loss: {train_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last_model.pth")
        if is_best:
            torch.save(ckpt, save_dir / "best_model.pth")

    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_config_json(config_path, config_data)
    print(f"[+] saved {save_dir}")
    return save_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaskLineFeatMatchNet Training Pipeline")
    parser.add_argument("--config", "-c", type=str, default="configs/mask_line.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data/line_gt_dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--no-timestamp", action="store_true")

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--min-matches", type=int, default=2)
    parser.add_argument("--data-ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/mask_line.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="train")

    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
