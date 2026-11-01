"""SuperPoint 极简评测与 Benchmark 骨架 (Minimal Evaluation Skeleton)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import apply_config_to_parser, load_config, pre_parse_config_path
from models.superpoint.utils import (
    SuperPointEvalHelper,
    batch_to_device,
    build_eval_parser,
    run_validation_loop,
)

# 兼容导出
_batch_to_device = batch_to_device
validate = run_validation_loop


def evaluate(
    args: argparse.Namespace | None = None,
    checkpoint: str | Path | None = None,
    data_dir: str | Path = "data/planar_letters",
    split: str = "val",
    out_dir: str | Path = "outputs/superpoint_eval",
    num_vis: int = 5,
    eval_stage: str = "all",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """SuperPoint 离线全套 Benchmark 评测主骨架。"""
    for k, v in {
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "split": split,
        "out_dir": out_dir,
        "num_vis": num_vis,
        "eval_stage": eval_stage,
        "device": device,
        "num_workers": num_workers,
    }.items():
        if v is not None:
            kwargs[k] = v
    helper = SuperPointEvalHelper(args=args, **kwargs)
    model, criterion, val_loader = helper.setup()

    model.eval()
    with torch.inference_mode():
        pbar = helper.progress_bar(val_loader)
        for batch in pbar:
            pred0, pred1, loss_dict = helper.forward_step(model, criterion, batch)
            helper.accumulate_step(batch, pred0, pred1, loss_dict, criterion=criterion)

    metrics = helper.compute_metrics()
    vis_paths = helper.render_visualizations(model, criterion, val_loader)
    metrics["visualizations"] = vis_paths
    helper.save_reports(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    return build_eval_parser()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/superpoint.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="eval")

    args = parser.parse_args(argv)
    return evaluate(args=args)


if __name__ == "__main__":
    main()
