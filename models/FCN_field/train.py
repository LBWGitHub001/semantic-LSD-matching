"""FCN_field Training Pipeline (Standalone & Modular)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

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
from models.FCN_field.model import FCNField
from models.FCN_field.loss import FCNFieldLoss
from models.FCN_field.eval import validate, _batch_to_device
from training_utils import (
    TQDM_BAR_FORMAT,
    create_experiment_dir,
    get_git_commit,
    save_config_json,
    update_latest_symlink,
)


def train(args: argparse.Namespace | None = None, **kwargs: Any) -> Path:
    """Trains FCNField homography estimator."""
    from datasets.homo_matcher_dataset import HomoMatcherDataset, collate_homo_matcher
    if args is None:
        parser = build_parser()
        args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    data_dir = Path(args.data_dir or "data/homo_matcher")
    base_save_dir = Path(args.save_dir or "checkpoints/FCN_field")
    save_dir, run_name = create_experiment_dir(
        base_save_dir=base_save_dir,
        model_name=f"FCN_field+{data_dir.name}",
        label=getattr(args, "label", None),
        no_timestamp=getattr(args, "no_timestamp", False),
        run_name=getattr(args, "run_name", None),
    )
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else 1e-4

    sample_pts = getattr(args, "line_sample_points", 100)
    train_ds = HomoMatcherDataset(data_dir, split="train", sample_points=sample_pts)
    val_ds = HomoMatcherDataset(data_dir, split="val", sample_points=sample_pts)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_homo_matcher, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_homo_matcher, num_workers=0)

    model = FCNField().to(device)
    loss_fn = FCNFieldLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=getattr(args, "weight_decay", 1e-4))

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
            "model_type": "FCN_field",
            "family": "homography",
            "feat_dim": model.feat_dim,
            "grid_size": model.grid_size,
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "loss": "symmetry",
        },
    }
    save_config_json(config_path, config_data)
    print(f"[*] FCN_field homography training | pairs={len(train_ds)} val={len(val_ds)} | device={device}")

    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        train_total = len(train_loader) if getattr(args, "max_steps", None) is None else min(len(train_loader), args.max_steps)
        pbar = tqdm(
            train_loader,
            total=train_total,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )
        for batch in pbar:
            if getattr(args, "max_steps", None) is not None and steps >= args.max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred = model(moved["image0"], moved["image1"])
            out = loss_fn(pred, moved)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_loss = float(out["loss"].detach())
            total += step_loss
            steps += 1
            pbar.set_postfix_str(f"symmetry={step_loss:.4f} avg={total / steps:.4f}")

        val_metrics = validate(model, loss_fn, val_loader, device, getattr(args, "max_steps", None))
        score = val_metrics["loss"]
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}  val symmetry {val_metrics['symmetry']:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last_model.pth")
        if score <= best_score:
            best_score = score
            torch.save(ckpt, save_dir / "best_model.pth")

    config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_config_json(config_path, config_data)
    print(f"[+] saved {save_dir}")
    return save_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCN_field Homography Training Pipeline")
    parser.add_argument("--config", "-c", type=str, default="configs/fcn_field.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data/homo_matcher", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--line-sample-points", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints/FCN_field")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--no-timestamp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/fcn_field.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="train")

    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
