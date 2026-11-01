"""FCN_field Homography Evaluation, Validation, and Benchmark Suite."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
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
from training_utils import TQDM_BAR_FORMAT, safe_load_checkpoint


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
) -> dict[str, float]:
    """In-training validation step for FCN_field homography model."""
    model.eval()
    totals = {"symmetry": 0.0, "loss": 0.0}
    steps = 0
    val_total = len(loader) if max_steps is None else min(len(loader), max_steps)
    pbar = tqdm(
        loader,
        total=val_total,
        desc="Val [Homography]",
        bar_format=TQDM_BAR_FORMAT,
        dynamic_ncols=True,
        leave=False,
        unit="pair",
    )
    with torch.inference_mode():
        for batch in pbar:
            if max_steps is not None and steps >= max_steps:
                break
            moved = _batch_to_device(batch, device)
            pred = model(moved["image0"], moved["image1"])
            out = criterion(pred, moved)
            totals["symmetry"] += float(out["symmetry"])
            totals["loss"] += float(out["loss"])
            steps += 1
            pbar.set_postfix_str(f"symmetry={totals['symmetry'] / steps:.4f}")
    denom = max(steps, 1)
    return {key: value / denom for key, value in totals.items()} | {"total_pairs": steps}


def evaluate(
    checkpoint: str | Path | None = None,
    data_dir: str | Path = "data/homo_matcher",
    split: str = "val",
    out_dir: str | Path = "outputs/fcn_field_eval",
    num_vis: int = 5,
    sample_points: int = 100,
    max_steps: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Full offline benchmark evaluation for FCNField homography estimator."""
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()
    p_out.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        checkpoint = "checkpoints/FCN_field/latest"
    ckpt_path = Path(checkpoint).resolve()
    if ckpt_path.is_dir() or not ckpt_path.exists():
        for candidate in (ckpt_path / "best_model.pth", ckpt_path / "last_model.pth"):
            if candidate.is_file():
                ckpt_path = candidate
                break

    model = FCNField().to(dev)
    if ckpt_path.is_file():
        ckpt_data = safe_load_checkpoint(ckpt_path, map_location=dev)
        state_dict = ckpt_data.get("model_state", ckpt_data.get("state_dict", ckpt_data))
        model.load_state_dict(state_dict, strict=True)
        print(f"[✓] Loaded FCN_field weights from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}, evaluating uninitialized model")

    criterion = FCNFieldLoss().to(dev)
    from datasets.homo_matcher_dataset import HomoMatcherDataset, collate_homo_matcher
    val_ds = HomoMatcherDataset(p_data, split=split, sample_points=sample_points)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_homo_matcher, num_workers=num_workers)

    metrics = validate(model, criterion, val_loader, dev, max_steps=max_steps)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    eval_json_path = p_out / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    eval_md_path = p_out / "eval_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# FCN_field Homography 评测报告 (Evaluation Report)\n\n")
        f.write(f"- 检查点: `{ckpt_path}`\n")
        f.write(f"- 数据集: `{p_data.name}` (Split: `{split}`, 共 {metrics['total_pairs']} 对样本)\n\n")
        f.write("| 评测指标 | 结果数值 | 说明 |\n| :--- | :---: | :--- |\n")
        f.write(f"| **单应对称误差 (Symmetry)** | `{metrics['symmetry']:.4f}` | 跨视角双向重投影一致性偏差 |\n")
        f.write(f"| **总损失 (Total Loss)** | `{metrics['loss']:.4f}` | 综合单应评估损失 |\n")

    print(f"[✓] Saved evaluation metrics to : {eval_json_path}")
    print(f"[✓] Saved evaluation report to  : {eval_md_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FCN_field Evaluation & Benchmark Entrypoint")
    parser.add_argument("--config", "-c", type=str, default="configs/fcn_field.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file or directory")
    parser.add_argument("--data-dir", type=str, default="data/homo_matcher", help="Dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs/fcn_field_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--sample-points", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/fcn_field.yaml")
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
        sample_points=getattr(args, "sample_points", 100),
        max_steps=getattr(args, "max_steps", None),
        device=args.device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
