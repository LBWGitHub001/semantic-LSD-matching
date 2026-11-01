"""XFeat Object Matcher Training Pipeline (Standalone & Modular)."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
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
from models.object_matcher.xfeat_object_matcher import XFeatObjectMatcher
from models.object_matcher.xfeat_line_matcher import XFeatLineMatcher
from models.object_matcher.loss import ObjectMatchingLoss
from models.object_matcher.eval import validate
from tools.experiment_tracker import ExperimentLogger
from training_utils import (
    TQDM_BAR_FORMAT,
    create_experiment_dir,
    get_git_commit,
    save_config_json,
    update_latest_symlink,
    worker_init_fn,
    build_object_forward_kwargs,
)


def train(args: argparse.Namespace | None = None, **kwargs: Any) -> Path:
    """Trains XFeatObjectMatcher or XFeatLineMatcher."""
    from datasets import ObjectMatchingDataset, collate_single_pair
    if args is None:
        parser = build_parser()
        args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    data_dir = Path(args.data_dir or "data/structured3d_semantic_regions")
    model_type = getattr(args, "model_type", "xfeat_object")
    default_ckpt_dir = "checkpoints/xfeat_object_matcher" if model_type == "xfeat_object" else f"checkpoints/{model_type}"
    base_save_dir = Path(getattr(args, "save_dir", None) or default_ckpt_dir)

    save_dir, run_name = create_experiment_dir(
        base_save_dir=base_save_dir,
        model_name=f"{model_type}+{data_dir.name}",
        label=getattr(args, "label", None),
        no_timestamp=getattr(args, "no_timestamp", False),
        run_name=getattr(args, "run_name", None),
    )
    lr = args.lr if args.lr is not None else 5e-4
    device = torch.device(args.device)

    # Dataset resolution
    use_region_ds = False
    p_data = Path(data_dir)
    if (p_data / "train_split.json").exists():
        with open(p_data / "train_split.json", "r", encoding="utf-8") as f:
            splits_data = json.load(f)
            if splits_data and (p_data / splits_data[0] / "correspondence.json").exists():
                use_region_ds = True

    needs_lines = "line" in model_type.lower()
    if use_region_ds:
        from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
        train_ds = SemanticRegionDataset(dataset_dir=data_dir, split="train", min_area_ratio=args.min_area_ratio, load_lines=needs_lines)
        val_ds = SemanticRegionDataset(dataset_dir=data_dir, split="val", min_area_ratio=args.min_area_ratio, load_lines=needs_lines)
        collate_fn = collate_region_pairs
    else:
        train_ds = ObjectMatchingDataset(data_dir=data_dir, split="train")
        val_ds = ObjectMatchingDataset(data_dir=data_dir, split="val")
        collate_fn = collate_single_pair

    orig_train_len = len(train_ds)
    if getattr(args, "data_ratio", 1.0) < 1.0:
        ratio = max(0.001, min(1.0, float(args.data_ratio)))
        subset_size = max(1, int(orig_train_len * ratio))
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(orig_train_len, size=subset_size, replace=False).tolist()
        train_ds = torch.utils.data.Subset(train_ds, subset_indices)
        print(f"[*] Dataset subset: Sampled {len(train_ds)}/{orig_train_len} ({ratio*100:.1f}%) pairs.")

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Instantiate model
    ModelCls = XFeatLineMatcher if needs_lines else XFeatObjectMatcher
    model = ModelCls(
        d_model=getattr(args, "d_model", 128),
        num_layers=getattr(args, "num_layers", 2),
        nhead=getattr(args, "nhead", 4),
        use_pyramid=getattr(args, "use_pyramid", True),
    ).to(device)

    criterion = ObjectMatchingLoss(
        dustbin_weight=getattr(args, "dustbin_weight", 0.5),
        motion_weight=getattr(args, "motion_weight", 0.2),
        motion_margin=getattr(args, "motion_margin", 0.1),
    ).to(device)

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
        "model": {
            "model_type": model_type,
            "family": "object",
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "nhead": getattr(args, "nhead", 4),
            "use_pyramid": getattr(args, "use_pyramid", True),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "dustbin_weight": args.dustbin_weight,
            "motion_weight": args.motion_weight,
            "min_area_ratio": args.min_area_ratio,
        },
    }
    save_config_json(config_path, config_data)
    logger = ExperimentLogger(save_dir=save_dir, config_data=config_data, enable_stdout_tee=False)

    print(f"[*] Object Matcher ({model_type}) training | pairs={len(train_ds)} val={len(val_ds)} | device={device}")

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        step_count = 0
        train_total = len(train_loader) if getattr(args, "max_steps", None) is None else min(len(train_loader), args.max_steps)

        pbar = tqdm(
            train_loader, total=train_total, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]",
            bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair",
        )
        optimizer.zero_grad()
        for batch in pbar:
            if getattr(args, "max_steps", None) is not None and step_count >= args.max_steps:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)

            c0 = batch["centroids0"].to(device) if "centroids0" in batch and batch["centroids0"] is not None else None
            c1 = batch["centroids1"].to(device) if "centroids1" in batch and batch["centroids1"] is not None else None

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(patches0, patches1, **build_object_forward_kwargs(batch, device))
                loss_val_t = criterion(
                    out["log_P"].float(),
                    batch["gt_matches"],
                    batch["unmatched_0"],
                    batch["unmatched_1"],
                    centroids0=c0,
                    centroids1=c1,
                )
                loss = loss_val_t / getattr(args, "grad_accum", 1)

            scaler.scale(loss).backward()
            step_count += 1

            if step_count % getattr(args, "grad_accum", 1) == 0 or step_count == train_total:
                scaler.unscale_(optimizer)
                has_grads = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
                if has_grads:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad()

            loss_val = float(loss_val_t.detach().item())
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                total_loss += loss_val
            pbar.set_postfix_str(f"loss={loss_val:6.4f}, avg={total_loss / max(step_count, 1):6.4f}")

        train_loss = total_loss / max(step_count, 1)

        val_metrics = None
        is_best = False
        if epoch % getattr(args, "val_interval", 1) == 0:
            val_metrics = validate(
                model=model, loader=val_loader, device=device,
                desc=f"Val [Epoch {epoch:02d}]", min_area_ratio=args.min_area_ratio,
                max_steps=getattr(args, "max_steps", None),
            )
            val_f1 = val_metrics["f1"]
            is_best = val_f1 >= best_val_f1
            if is_best:
                best_val_f1 = val_f1

            star = " ★ BEST" if is_best else ""
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {train_loss:.4f} | Val F1: {val_f1:5.1f}% | Dustbin Acc: {val_metrics['dustbin_acc']:4.1f}%{star}")
        else:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {train_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last_model.pth")
        if is_best:
            torch.save(ckpt, save_dir / "best_model.pth")

        logger.log_epoch(epoch=epoch, train_loss=train_loss, val_metrics=val_metrics, is_best=is_best, epoch_sec=0.0)

    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_config_json(config_path, config_data)
    logger.finish()
    print(f"[+] saved {save_dir}")
    return save_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XFeat Object Matcher Training Pipeline")
    parser.add_argument("--config", "-c", type=str, default="configs/xfeat_object.yaml", help="Path to config YAML")
    parser.add_argument("--model-type", type=str, default="xfeat_object", choices=["xfeat_object", "xfeat_line_matcher"])
    parser.add_argument("--data-dir", type=str, default="data/structured3d_semantic_regions")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
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
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--use-pyramid", action="store_true", default=True)
    parser.add_argument("--dustbin-weight", type=float, default=0.5)
    parser.add_argument("--motion-weight", type=float, default=0.2)
    parser.add_argument("--motion-margin", type=float, default=0.1)
    parser.add_argument("--min-area-ratio", type=float, default=0.50)
    parser.add_argument("--data-ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/xfeat_object.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="train")

    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
