"""SuperPoint 极简训练骨架 (Minimal Training Skeleton)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import apply_config_to_parser, load_config, pre_parse_config_path
from models.superpoint.utils import SuperPointTrainerHelper, build_train_parser


def train(args: argparse.Namespace | None = None, **kwargs: Any) -> Path:
    """SuperPoint 极简训练主循环骨架。"""
    helper = SuperPointTrainerHelper(args, **kwargs)
    model, loss_fn, optimizer, train_loader, val_loader = helper.setup()

    best_score = float("inf")
    for epoch in range(1, helper.epochs + 1):
        stage = helper.update_stage(model, loss_fn, epoch)
        model.train()

        pbar = helper.progress_bar(train_loader, epoch, stage)
        epoch_loss = 0.0
        step_count = 0
        for batch in pbar:
            loss, loss_dict = helper.train_step(model, loss_fn, optimizer, batch)
            helper.log_step(pbar, loss, loss_dict, stage)
            epoch_loss += loss
            step_count += 1
            if helper.max_steps is not None and step_count >= helper.max_steps:
                break

        val_metrics = helper.validate(model, loss_fn, val_loader, epoch, stage)
        is_best = val_metrics["loss"] < best_score
        if is_best:
            best_score = val_metrics["loss"]
        helper.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
        print(f"Epoch [{epoch:02d}/{helper.epochs:02d}] [{stage}] loss: {epoch_loss / max(step_count, 1):.4f} | val loss: {val_metrics['loss']:.4f}")

    print(f"[+] Training completed. Checkpoints saved to: {helper.save_dir}")
    return helper.save_dir


def build_parser() -> argparse.ArgumentParser:
    return build_train_parser()


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
