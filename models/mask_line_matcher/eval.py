"""MaskLineFeatMatchNet Evaluation, Validation, and Benchmark Suite."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from datasets import LineFeatPairDataset
from models.mask_line_matcher.mask_line_matcher import MaskLineFeatMatchNet
from models.mask_line_matcher.loss import LineMatchingLoss
from training_utils import TQDM_BAR_FORMAT, safe_load_checkpoint


def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    """In-training validation step for MaskLineFeatMatchNet."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_steps = 0

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    val_total = len(loader) if max_steps is None else min(len(loader), max_steps)
    pbar = tqdm(loader, total=val_total, desc="Val [MaskLine]", bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")

    with torch.no_grad():
        for sample in pbar:
            if max_steps is not None and val_steps >= max_steps:
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

            out = model(img0, img1, lines0, lines1)
            loss_v, metrics_v = criterion(out["log_assignment"], num_matches=M)

            val_loss += metrics_v["loss"]
            val_acc += metrics_v.get("acc", 0.0)
            val_steps += 1

    denom = max(val_steps, 1)
    return {
        "loss": val_loss / denom,
        "acc": val_acc / denom,
        "total_pairs": val_steps,
    }


def evaluate(
    checkpoint: str | Path | None = None,
    data_dir: str | Path = "data/line_gt_dataset",
    split: str = "val",
    out_dir: str | Path = "outputs/mask_line_eval",
    num_vis: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 2,
    max_steps: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Full offline benchmark evaluation for MaskLineFeatMatchNet."""
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()
    p_out.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        checkpoint = "checkpoints/mask_line/latest"
    ckpt_path = Path(checkpoint).resolve()
    if ckpt_path.is_dir() or not ckpt_path.exists():
        for candidate in (ckpt_path / "best_model.pth", ckpt_path / "last_model.pth"):
            if candidate.is_file():
                ckpt_path = candidate
                break

    model = MaskLineFeatMatchNet().to(dev)
    if ckpt_path.is_file():
        ckpt_data = safe_load_checkpoint(ckpt_path, map_location=dev)
        state_dict = ckpt_data.get("model_state", ckpt_data.get("state_dict", ckpt_data))
        model.load_state_dict(state_dict, strict=False)
        print(f"[✓] Loaded MaskLine weights from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}, evaluating uninitialized model")

    criterion = LineMatchingLoss().to(dev)
    index_file = p_data / "index.json"
    if not index_file.is_file():
        for cand in ("data/line_gt_dataset/index.json", "data/synthetic_line_scenes/index.json"):
            if Path(cand).is_file():
                index_file = Path(cand)
                break

    if not index_file.is_file():
        print(f"[!] Notice: Line index file not found at {index_file}.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0, "loss": 0.0}

    val_ds = LineFeatPairDataset(index_path=str(index_file), split=split, load_images=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    metrics = validate(model, criterion, val_loader, dev, max_steps=max_steps)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    eval_json_path = p_out / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    eval_md_path = p_out / "eval_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# MaskLineFeatMatchNet 评测报告 (Evaluation Report)\n\n")
        f.write(f"- 检查点: `{ckpt_path}`\n")
        f.write(f"- 数据集: `{p_data.name}` (Split: `{split}`, 共 {metrics['total_pairs']} 对样本)\n\n")
        f.write("| 评测指标 | 结果数值 | 说明 |\n| :--- | :---: | :--- |\n")
        f.write(f"| **匹配准确率 (Line Accuracy)** | **{metrics['acc']:.2f}%** | 直线几何双向匹配正确率 |\n")
        f.write(f"| **匹配损失 (Line Loss)** | {metrics['loss']:.4f} | Optimal Transport 交叉熵损失 |\n")

    print(f"[✓] Saved evaluation metrics to : {eval_json_path}")
    print(f"[✓] Saved evaluation report to  : {eval_md_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaskLine Evaluation Entrypoint")
    parser.add_argument("--config", "-c", type=str, default="configs/mask_line.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file or directory")
    parser.add_argument("--data-dir", type=str, default="data/line_gt_dataset", help="Dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs/mask_line_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/mask_line.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="eval")

    args = parser.parse_args(argv)
    return evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        out_dir=args.out_dir,
        num_vis=args.num_vis,
        device=args.device,
        num_workers=args.num_workers,
        max_steps=getattr(args, "max_steps", None),
    )


if __name__ == "__main__":
    main()
