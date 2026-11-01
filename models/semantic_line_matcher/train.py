"""SemanticLineMatcher Training Pipeline (Standalone & Modular)."""

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
from models.semantic_line_matcher.model import SemanticLineMatcher
from models.semantic_line_matcher.loss import SemanticLineMatchingLoss
from models.semantic_line_matcher.eval import validate
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
    """Trains SemanticLineMatcher."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    data_dir = Path(args.data_dir or "data/synthetic_semantic_lines")
    base_save_dir = Path(getattr(args, "save_dir", None) or "checkpoints/semantic_line_matcher")

    save_dir, run_name = create_experiment_dir(
        base_save_dir=base_save_dir,
        model_name=f"semantic_line_matcher+{data_dir.name}",
        label=getattr(args, "label", None),
        no_timestamp=getattr(args, "no_timestamp", False),
        run_name=getattr(args, "run_name", None),
    )
    lr = args.lr if args.lr is not None else 5e-4
    device = torch.device(args.device)

    from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
    train_ds = SemanticRegionDataset(dataset_dir=data_dir, split="train", min_area_ratio=args.min_area_ratio, load_lines=True)
    val_ds = SemanticRegionDataset(dataset_dir=data_dir, split="val", min_area_ratio=args.min_area_ratio, load_lines=True)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_region_pairs,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_region_pairs, num_workers=0)

    model = SemanticLineMatcher(
        d_model=getattr(args, "d_model", 128),
        num_layers=getattr(args, "num_layers", 2),
        nhead=getattr(args, "nhead", 4),
        use_pyramid=getattr(args, "use_pyramid", True),
        with_geometry=getattr(args, "with_geometry", True),
    ).to(device)

    criterion = SemanticLineMatchingLoss(
        dustbin_weight=getattr(args, "dustbin_weight", 0.3),
        region_weight=getattr(args, "region_weight", 1.0),
        line_weight=getattr(args, "line_weight", 1.0),
        sem_weight=getattr(args, "sem_weight", 0.1),
        geo_weight=getattr(args, "geo_weight", 0.2),
        adj_sigma=getattr(args, "adj_sigma", 0.3),
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
            "model_type": "semantic_line_matcher",
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
            "region_weight": getattr(args, "region_weight", 1.0),
            "line_weight": getattr(args, "line_weight", 1.0),
            "min_area_ratio": args.min_area_ratio,
        },
    }
    save_config_json(config_path, config_data)
    logger = ExperimentLogger(save_dir=save_dir, config_data=config_data, enable_stdout_tee=False)

    print(f"[*] SemanticLineMatcher training | pairs={len(train_ds)} val={len(val_ds)} | device={device}")

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
                log_line = out.get("log_P_line")
                if torch.is_tensor(log_line):
                    log_line = log_line.float()
                loss_out = criterion(
                    log_P_region=out["log_P"].float(),
                    log_P_line=log_line,
                    gt_matches=batch["gt_matches"],
                    unmatched_0=batch["unmatched_0"],
                    unmatched_1=batch["unmatched_1"],
                    gt_line_pairs_flat=batch.get("gt_line_pairs_flat", [[]]),
                    unmatched_lines_flat_0=batch.get("unmatched_lines_flat_0", [[]]),
                    unmatched_lines_flat_1=batch.get("unmatched_lines_flat_1", [[]]),
                    line_parents0=out.get("line_parents0"),
                    line_parents1=out.get("line_parents1"),
                    centroids0=c0,
                    centroids1=c1,
                    has_line_ids=bool(batch.get("has_line_ids", False)),
                )
                loss_val_t = loss_out[0] if isinstance(loss_out, tuple) else loss_out
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
    parser = argparse.ArgumentParser(description="SemanticLineMatcher Training Pipeline")
    parser.add_argument("--config", "-c", type=str, default="configs/semantic_line_matcher.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_semantic_lines")
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
    parser.add_argument("--with-geometry", action="store_true", default=True)
    parser.add_argument("--dustbin-weight", type=float, default=0.3)
    parser.add_argument("--region-weight", type=float, default=1.0)
    parser.add_argument("--line-weight", type=float, default=1.0)
    parser.add_argument("--sem-weight", type=float, default=0.1)
    parser.add_argument("--geo-weight", type=float, default=0.2)
    parser.add_argument("--adj-sigma", type=float, default=0.3)
    parser.add_argument("--min-area-ratio", type=float, default=0.50)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/semantic_line_matcher.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="train")

    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
