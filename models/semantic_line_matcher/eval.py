"""SemanticLineMatcher Evaluation, Validation, and Benchmark Suite."""

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
from models.semantic_line_matcher.model import SemanticLineMatcher
from models.object_matcher.eval import validate as object_validate
from training_utils import TQDM_BAR_FORMAT, safe_load_checkpoint, build_object_forward_kwargs
from visualization.matching_visualizer import save_pair_visualization


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
    min_area_ratio: float = 0.50,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Validation step for dual region + line matching."""
    return object_validate(
        model=model,
        loader=loader,
        device=device,
        desc=desc,
        min_area_ratio=min_area_ratio,
        max_steps=max_steps,
    )


def evaluate(
    checkpoint: str | Path | None = None,
    data_dir: str | Path = "data/synthetic_semantic_lines",
    split: str = "test",
    out_dir: str | Path = "outputs/semantic_line_eval",
    num_vis: int = 5,
    min_area_ratio: float = 0.50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 2,
    max_steps: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Full offline benchmark evaluation for SemanticLineMatcher."""
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()
    p_out.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        checkpoint = "checkpoints/semantic_line_matcher/latest"
    ckpt_path = Path(checkpoint).resolve()
    if ckpt_path.is_dir() or not ckpt_path.exists():
        for candidate in (ckpt_path / "best_model.pth", ckpt_path / "last_model.pth"):
            if candidate.is_file():
                ckpt_path = candidate
                break

    model = SemanticLineMatcher().to(dev)
    if ckpt_path.is_file():
        ckpt_data = safe_load_checkpoint(ckpt_path, map_location=dev)
        state_dict = ckpt_data.get("model_state", ckpt_data.get("state_dict", ckpt_data))
        model.load_state_dict(state_dict, strict=False)
        print(f"[✓] Loaded SemanticLineMatcher weights from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}, evaluating uninitialized model")

    from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
    val_ds = SemanticRegionDataset(dataset_dir=p_data, split=split, min_area_ratio=min_area_ratio, load_lines=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_region_pairs, num_workers=num_workers)

    metrics = validate(model, val_loader, dev, desc="Benchmarking", min_area_ratio=min_area_ratio, max_steps=max_steps)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Qualitative visualizations
    vis_paths: list[str] = []
    if num_vis > 0:
        from datasets import predict_bipartite_matches
        vis_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if vis_count >= num_vis:
                    break
                p0 = batch["patches0"].to(dev)
                p1 = batch["patches1"].to(dev)
                out = model(p0, p1, **build_object_forward_kwargs(batch, dev))
                pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                    out["prob_matrix"][0],
                    thresh=0.2,
                    areas_0=batch.get("areas0"),
                    areas_1=batch.get("areas1"),
                    min_area_ratio=min_area_ratio,
                )
                scene_tag = batch.get("scene_name", f"sample_{vis_count + 1}")
                vis_dest = p_out / f"eval_sample_{vis_count + 1:02d}_{scene_tag}.jpg"
                save_pair_visualization(
                    batch,
                    out,
                    vis_dest,
                    title=f"Sample {vis_count + 1}: {scene_tag}",
                    pred_matches=pred_matches,
                    pred_un0=pred_un0,
                    pred_un1=pred_un1,
                    data_dir=p_data,
                    scene_name=scene_tag,
                    use_region_ds=True,
                )
                vis_paths.append(str(vis_dest))
                vis_count += 1

    metrics["visualizations"] = vis_paths

    eval_json_path = p_out / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    eval_md_path = p_out / "eval_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# SemanticLineMatcher 评测报告 (Evaluation Report)\n\n")
        f.write(f"- 检查点: `{ckpt_path}`\n")
        f.write(f"- 数据集: `{p_data.name}` (Split: `{split}`, 共 {metrics['total_pairs']} 对样本)\n\n")
        f.write("| 评测指标 | 结果数值 | 说明 |\n| :--- | :---: | :--- |\n")
        f.write(f"| **Match Precision** | **{metrics['precision']:.2f}%** | 匹配查准率 |\n")
        f.write(f"| **Match Recall** | **{metrics['recall']:.2f}%** | 目标查全率 |\n")
        f.write(f"| **Match F1-Score** | **{metrics['f1']:.2f}%** | 综合性能指标 |\n")
        f.write(f"| **Dustbin Accuracy** | {metrics['dustbin_acc']:.2f}% | 垃圾桶判定准确率 |\n")

    print(f"[✓] Saved evaluation metrics to : {eval_json_path}")
    print(f"[✓] Saved evaluation report to  : {eval_md_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SemanticLineMatcher Evaluation Entrypoint")
    parser.add_argument("--config", "-c", type=str, default="configs/semantic_line_matcher.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file or directory")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_semantic_lines", help="Dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs/semantic_line_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--min-area-ratio", type=float, default=0.50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/semantic_line_matcher.yaml")
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
        min_area_ratio=args.min_area_ratio,
        device=args.device,
        num_workers=args.num_workers,
        max_steps=getattr(args, "max_steps", None),
    )


if __name__ == "__main__":
    main()
