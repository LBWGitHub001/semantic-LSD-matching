"""SuperPoint Training Pipeline (Standalone & Modular)."""

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
from datasets.point_match_dataset import PointMatchDataset, collate_point_matches
from models.superpoint.model import SuperPoint
from models.superpoint.loss import SuperPointLoss
from models.superpoint.eval import validate, _batch_to_device
from training_utils import (
    TQDM_BAR_FORMAT,
    create_experiment_dir,
    get_git_commit,
    save_config_json,
    update_latest_symlink,
)


def train(args: argparse.Namespace | None = None, **kwargs: Any) -> Path:
    """Trains SuperPoint network with stage control (detector, descriptor, joint)."""
    if args is None:
        parser = build_parser()
        args = parser.parse_args([])
    for k, v in kwargs.items():
        setattr(args, k, v)

    data_dir = Path(args.data_dir or "data/planar_letters")
    base_save_dir = Path(args.save_dir or "checkpoints/SuperPoint")
    train_stage = getattr(args, "train_stage", "joint").lower()
    save_dir, run_name = create_experiment_dir(
        base_save_dir=base_save_dir,
        model_name=f"SuperPoint_{train_stage}+{data_dir.name}",
        label=getattr(args, "label", None),
        no_timestamp=getattr(args, "no_timestamp", False),
        run_name=getattr(args, "run_name", None),
    )
    vis_dir = save_dir / "visualizations"
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else 1e-3

    train_ds = PointMatchDataset(data_dir, split="train")
    val_ds = PointMatchDataset(data_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_point_matches, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_point_matches, num_workers=0)

    loss_kwargs = {
        "weight_desc": getattr(args, "weight_desc", 1.0),
        "positive_weight": getattr(args, "positive_weight", 250.0),
        "positive_margin": getattr(args, "positive_margin", 1.0),
        "negative_margin": getattr(args, "negative_margin", 0.2),
        "weight_det": getattr(args, "weight_det", 1.0),
        "det_pos_weight": getattr(args, "det_pos_weight", 50.0),
        "weight_orphan": getattr(args, "weight_orphan", 1.0),
        "weight_len": getattr(args, "weight_len", 0.5),
        "length_exponent": getattr(args, "length_exponent", 1.5),
        "weight_premature": getattr(args, "weight_premature", 0.5),
        "probe_dist": getattr(args, "probe_dist", 4.0),
        "tau_df": getattr(args, "tau_df", 0.55),
        "tau_ang": getattr(args, "tau_ang", 0.10),
        "min_length": getattr(args, "min_length", 8.0),
        "num_candidates": getattr(args, "num_candidates", 16),
        "num_line_samples": getattr(args, "num_line_samples", 10),
        "nms_radius": getattr(args, "nms_radius", 4),
        "weight_intra": getattr(args, "weight_intra", 1.0),
        "weight_cross": getattr(args, "weight_cross", 1.5),
        "weight_inter": getattr(args, "weight_inter", 0.5),
        "weight_bg": getattr(args, "weight_bg", 0.3),
        "line_neg_margin": getattr(args, "line_neg_margin", 0.2),
        "bg_neg_margin": getattr(args, "bg_neg_margin", 0.1),
        "num_line_points": getattr(args, "num_line_points", 12),
        "num_bg_samples": getattr(args, "num_bg_samples", 64),
    }

    model = SuperPoint().to(device)
    loss_fn = SuperPointLoss(stage=train_stage, **loss_kwargs).to(device)

    # 阶段自适应参数冻结
    if hasattr(model, "set_train_stage"):
        model.set_train_stage(train_stage)
        trainable_params = model.get_trainable_parameters(train_stage)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=getattr(args, "weight_decay", 1e-4))

    config_path = save_dir / "config.json"
    config_data = {
        "experiment_info": {
            "run_name": run_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "git_commit": get_git_commit(),
            "command": " ".join(sys.argv),
        },
        "model": {"model_type": "SuperPoint", "family": "point", "train_stage": train_stage},
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": lr,
            "loss": f"superpoint_{train_stage}",
            "trainable_param_groups": len(trainable_params),
            "loss_weights": loss_kwargs,
        },
    }
    save_config_json(config_path, config_data)
    print(f"[*] SuperPoint stage: '{train_stage}' | pairs={len(train_ds)} val={len(val_ds)} | trainable params: {len(trainable_params)} tensors | device={device}")

    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        train_total = len(train_loader) if getattr(args, "max_steps", None) is None else min(len(train_loader), args.max_steps)
        pbar = tqdm(
            train_loader,
            total=train_total,
            desc=f"Epoch {epoch:02d}/{args.epochs:02d} [{train_stage.capitalize()}]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )
        for batch in pbar:
            if getattr(args, "max_steps", None) is not None and steps >= args.max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred0 = model(moved["image0"])
            pred1 = model(moved["image1"])
            out = loss_fn(
                pred0,
                pred1,
                moved["homography"],
                image0=moved["image0"],
                image1=moved["image1"],
                points0=moved.get("points0"),
                points1=moved.get("points1"),
                lines0=moved.get("lines0"),
                lines1=moved.get("lines1"),
            )
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step_loss = float(out["loss"].detach())
            total += step_loss
            steps += 1
            if train_stage == "descriptor":
                pbar.set_postfix_str(f"loss={step_loss:.4f} ld={float(out['ld']):.3f}")
            elif train_stage == "detector":
                pbar.set_postfix_str(
                    f"loss={step_loss:.4f} det={float(out['l_det']):.3f} orph={float(out['l_orphan']):.3f} len={float(out['l_len']):.3f}"
                )
            else:
                pbar.set_postfix_str(
                    f"loss={step_loss:.4f} desc={float(out.get('ld', 0.0)):.3f} orph={float(out.get('l_orphan', 0.0)):.3f} len={float(out.get('l_len', 0.0)):.3f}"
                )

        val_metrics = validate(
            model=model,
            criterion=loss_fn,
            loader=val_loader,
            device=device,
            max_steps=getattr(args, "max_steps", None),
            vis_dir=vis_dir,
            epoch=epoch,
            train_stage=train_stage,
        )
        score = val_metrics["loss"]
        if train_stage == "detector":
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
                f"  val det {val_metrics['loss']:.4f}  lines {val_metrics.get('num_lines', 0.0):.1f}  orphan {val_metrics.get('orphan_rate', 0.0)*100:.1f}%"
            )
        elif train_stage == "descriptor":
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
                f"  val desc {val_metrics['loss']:.4f}  match {val_metrics.get('match_acc', 0.0) * 100:.1f}%"
            )
        else:
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] loss {total / max(steps, 1):.4f}"
                f"  val loss {val_metrics['loss']:.4f}  match {val_metrics.get('match_acc', 0.0) * 100:.1f}%  lines {val_metrics.get('num_lines', 0.0):.1f}"
            )

        ckpt = {
            "epoch": epoch,
            "train_stage": train_stage,
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
    parser = argparse.ArgumentParser(description="SuperPoint Training Pipeline")
    parser.add_argument("--config", "-c", type=str, default="configs/superpoint.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default="data/planar_letters", help="Dataset directory")
    parser.add_argument("--train-stage", type=str, default="joint", choices=["joint", "detector", "descriptor"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints/SuperPoint")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--no-timestamp", action="store_true")

    # SuperPoint 16 lambdas & geometric thresholds
    parser.add_argument("--weight-desc", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=250.0)
    parser.add_argument("--positive-margin", type=float, default=1.0)
    parser.add_argument("--negative-margin", type=float, default=0.2)
    parser.add_argument("--weight-det", type=float, default=1.0)
    parser.add_argument("--det-pos-weight", type=float, default=50.0)
    parser.add_argument("--weight-orphan", type=float, default=1.0)
    parser.add_argument("--weight-len", type=float, default=0.5)
    parser.add_argument("--length-exponent", type=float, default=1.5)
    parser.add_argument("--weight-premature", type=float, default=0.5)
    parser.add_argument("--probe-dist", type=float, default=4.0)
    parser.add_argument("--tau-df", type=float, default=0.55)
    parser.add_argument("--tau-ang", type=float, default=0.10)
    parser.add_argument("--min-length", type=float, default=8.0)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--num-line-samples", type=int, default=10)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--weight-intra", type=float, default=1.0)
    parser.add_argument("--weight-cross", type=float, default=1.5)
    parser.add_argument("--weight-inter", type=float, default=0.5)
    parser.add_argument("--weight-bg", type=float, default=0.3)
    parser.add_argument("--line-neg-margin", type=float, default=0.2)
    parser.add_argument("--bg-neg-margin", type=float, default=0.1)
    parser.add_argument("--num-line-points", type=int, default=12)
    parser.add_argument("--num-bg-samples", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/superpoint.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="train")

    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
