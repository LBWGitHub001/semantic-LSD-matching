"""XFeat Object Matcher Evaluation, Validation, and Benchmark Suite."""

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
from models.object_matcher.xfeat_object_matcher import XFeatObjectMatcher
from models.object_matcher.xfeat_line_matcher import XFeatLineMatcher
from training_utils import TQDM_BAR_FORMAT, safe_load_checkpoint, build_object_forward_kwargs
from visualization.matching_visualizer import save_pair_visualization


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating",
    enforce_same_category: bool = False,
    min_area_ratio: float = 0.0,
    max_steps: int | None = None,
) -> dict[str, float]:
    """In-training validation step for Object Matcher ('连连看')."""
    from datasets import predict_bipartite_matches
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    cross_cat_fp = 0
    same_cat_fp = 0
    dustbin_tp, dustbin_fp, dustbin_fn, dustbin_tn = 0, 0, 0, 0
    steps = 0

    val_total = len(loader) if max_steps is None else min(len(loader), max_steps)
    pbar = tqdm(loader, total=val_total, desc=desc, bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")
    with torch.no_grad():
        for batch in pbar:
            if max_steps is not None and steps >= max_steps:
                break
            patches0 = batch["patches0"].to(device)
            patches1 = batch["patches1"].to(device)
            out = model(patches0, patches1, **build_object_forward_kwargs(batch, device))

            cats0 = [o.get("category", "") for o in batch.get("objs0", [])] if enforce_same_category else None
            cats1 = [o.get("category", "") for o in batch.get("objs1", [])] if enforce_same_category else None

            pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                out["prob_matrix"][0],
                thresh=0.2,
                categories_0=cats0,
                categories_1=cats1,
                areas_0=batch.get("areas0"),
                areas_1=batch.get("areas1"),
                min_area_ratio=min_area_ratio,
            )

            gt_matches = set(batch["gt_matches"][0])
            pred_set = set(pred_matches)

            cur_tp = len(pred_set.intersection(gt_matches))
            cur_fp_set = pred_set - gt_matches
            cur_fn = len(gt_matches - pred_set)

            total_tp += cur_tp
            total_fp += len(cur_fp_set)
            total_fn += cur_fn

            objs0 = batch.get("objs0", [])
            objs1 = batch.get("objs1", [])
            for i, j in cur_fp_set:
                c0 = objs0[i].get("category", "") if i < len(objs0) else ""
                c1 = objs1[j].get("category", "") if j < len(objs1) else ""
                if c0 and c1 and c0 != c1:
                    cross_cat_fp += 1
                else:
                    same_cat_fp += 1

            # Dustbin metrics
            gt_un0 = set(batch["unmatched_0"][0])
            gt_un1 = set(batch["unmatched_1"][0])
            pred_un0_set = set(pred_un0)
            pred_un1_set = set(pred_un1)

            m0_num = patches0.shape[1]
            m1_num = patches1.shape[1]

            for i in range(m0_num):
                is_un_gt = (i in gt_un0)
                is_un_pred = (i in pred_un0_set)
                if is_un_gt and is_un_pred:
                    dustbin_tp += 1
                elif not is_un_gt and is_un_pred:
                    dustbin_fp += 1
                elif is_un_gt and not is_un_pred:
                    dustbin_fn += 1
                else:
                    dustbin_tn += 1

            for j in range(m1_num):
                is_un_gt = (j in gt_un1)
                is_un_pred = (j in pred_un1_set)
                if is_un_gt and is_un_pred:
                    dustbin_tp += 1
                elif not is_un_gt and is_un_pred:
                    dustbin_fp += 1
                elif is_un_gt and not is_un_pred:
                    dustbin_fn += 1
                else:
                    dustbin_tn += 1

            steps += 1
            cur_p = total_tp / (total_tp + total_fp + 1e-6)
            cur_r = total_tp / (total_tp + total_fn + 1e-6)
            cur_f1 = 2.0 * cur_p * cur_r / (cur_p + cur_r + 1e-6)
            pbar.set_postfix_str(f"F1={cur_f1*100:5.1f}%, TP={total_tp:>3d}, FP={total_fp:>3d}, FN={total_fn:>3d}")

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    dustbin_prec = dustbin_tp / (dustbin_tp + dustbin_fp + 1e-6)
    dustbin_rec = dustbin_tp / (dustbin_tp + dustbin_fn + 1e-6)
    dustbin_acc = (dustbin_tp + dustbin_tn) / (dustbin_tp + dustbin_fp + dustbin_fn + dustbin_tn + 1e-6)
    cross_cat_rate = cross_cat_fp / (total_fp + 1e-6)

    return {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "dustbin_prec": dustbin_prec * 100.0,
        "dustbin_rec": dustbin_rec * 100.0,
        "dustbin_acc": dustbin_acc * 100.0,
        "cross_cat_fp": cross_cat_fp,
        "cross_cat_rate": cross_cat_rate * 100.0,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "total_pairs": steps,
    }


evaluate_object_pipeline = validate


def evaluate(
    checkpoint: str | Path | None = None,
    data_dir: str | Path = "data/structured3d_semantic_regions",
    split: str = "test",
    out_dir: str | Path = "outputs/object_matcher_eval",
    num_vis: int = 5,
    min_area_ratio: float = 0.50,
    model_type: str = "xfeat_object",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 2,
    max_steps: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Full offline benchmark evaluation for Object Matcher."""
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()
    p_out.mkdir(parents=True, exist_ok=True)

    if checkpoint is None:
        checkpoint = "checkpoints/xfeat_object_matcher/latest"
    ckpt_path = Path(checkpoint).resolve()
    if ckpt_path.is_dir() or not ckpt_path.exists():
        for candidate in (ckpt_path / "best_model.pth", ckpt_path / "last_model.pth"):
            if candidate.is_file():
                ckpt_path = candidate
                break

    use_line_matcher = "line" in model_type.lower()
    ModelCls = XFeatLineMatcher if use_line_matcher else XFeatObjectMatcher
    model = ModelCls().to(dev)

    if ckpt_path.is_file():
        ckpt_data = safe_load_checkpoint(ckpt_path, map_location=dev)
        state_dict = ckpt_data.get("model_state", ckpt_data.get("state_dict", ckpt_data))
        model.load_state_dict(state_dict, strict=False)
        print(f"[✓] Loaded ObjectMatcher weights from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}, evaluating uninitialized model")

    # Dataset loader
    use_region = (p_data / "train_split.json").exists() or (p_data / "test_split.json").exists()
    if use_region:
        from datasets.semantic_region_dataset import SemanticRegionDataset, collate_region_pairs
        val_ds = SemanticRegionDataset(dataset_dir=p_data, split=split, min_area_ratio=min_area_ratio, load_lines=use_line_matcher)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_region_pairs, num_workers=num_workers)
    else:
        from datasets import ObjectMatchingDataset, collate_single_pair
        val_ds = ObjectMatchingDataset(data_dir=p_data, split=split)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single_pair, num_workers=num_workers)

    metrics = validate(
        model=model,
        loader=val_loader,
        device=dev,
        desc="Benchmarking",
        min_area_ratio=min_area_ratio,
        max_steps=max_steps,
    )
    metrics["checkpoint"] = str(ckpt_path)
    metrics["evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Render qualitative visualizations
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
                    use_region_ds=use_region,
                )
                vis_paths.append(str(vis_dest))
                vis_count += 1

    metrics["visualizations"] = vis_paths

    eval_json_path = p_out / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    eval_md_path = p_out / "eval_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# XFeat Object Matcher 评测报告 (Evaluation Report)\n\n")
        f.write(f"- 检查点: `{ckpt_path}`\n")
        f.write(f"- 数据集: `{p_data.name}` (Split: `{split}`, 共 {metrics['total_pairs']} 对样本)\n\n")
        f.write("| 评测指标 | 结果数值 | 说明 |\n| :--- | :---: | :--- |\n")
        f.write(f"| **Match Precision** | **{metrics['precision']:.2f}%** | 匹配查准率 |\n")
        f.write(f"| **Match Recall** | **{metrics['recall']:.2f}%** | 目标查全率 |\n")
        f.write(f"| **Match F1-Score** | **{metrics['f1']:.2f}%** | 核心综合性能指标 |\n")
        f.write(f"| **Dustbin Accuracy** | {metrics['dustbin_acc']:.2f}% | 垃圾桶判定准确率 |\n")
        f.write(f"| **Cross-Category FP** | {metrics['cross_cat_fp']} ({metrics['cross_cat_rate']:.1f}%) | 跨类别误匹配率 |\n")
        f.write(f"| **Correspondences** | {metrics['tp']} TP, {metrics['fp']} FP, {metrics['fn']} FN | 匹配计数明细 |\n")

    print(f"[✓] Saved evaluation metrics to : {eval_json_path}")
    print(f"[✓] Saved evaluation report to  : {eval_md_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Object Matcher Evaluation & Benchmark Entrypoint")
    parser.add_argument("--config", "-c", type=str, default="configs/xfeat_object.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file or directory")
    parser.add_argument("--data-dir", type=str, default="data/structured3d_semantic_regions", help="Dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs/object_matcher_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--min-area-ratio", type=float, default=0.50)
    parser.add_argument("--model-type", type=str, default="xfeat_object")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/xfeat_object.yaml")
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
        model_type=args.model_type,
        device=args.device,
        num_workers=args.num_workers,
        max_steps=getattr(args, "max_steps", None),
    )


if __name__ == "__main__":
    main()
