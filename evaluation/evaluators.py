"""
Evaluators & Benchmark Suite for Matching Models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import predict_bipartite_matches
from visualization.matching_visualizer import save_pair_visualization
from evaluation.checkpoint import resolve_checkpoint_path
from evaluation.model_factory import ModelFactory
from evaluation.dataset_factory import DatasetFactory

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

TQDM_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def build_object_forward_kwargs(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move optional line and bbox inputs in a region/object batch to device."""
    kwargs: Dict[str, Any] = {}
    for key0, key1 in (("patch_lines0", "patch_lines1"), ("line_mask0", "line_mask1"), ("bboxes0", "bboxes1")):
        if batch.get(key0) is not None and batch.get(key1) is not None:
            kwargs[key0] = batch[key0].to(device)
            kwargs[key1] = batch[key1].to(device)
    return kwargs


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        min_area_ratio: float = 0.50,
    ) -> Dict[str, Any]:
        """Runs quantitative benchmark evaluation and returns metrics dictionary."""
        pass

    @abstractmethod
    def visualize(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        out_dir: Path,
        num_vis: int = 5,
        min_area_ratio: float = 0.50,
        data_dir: Optional[Path] = None,
        use_region_ds: bool = True,
    ) -> List[Path]:
        """Renders diagnostic comparison visualization images."""
        pass


class ObjectMatchingEvaluator(BaseEvaluator):
    """Unified Evaluator for Object & Region Bipartite Matching Models."""

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        min_area_ratio: float = 0.50,
    ) -> Dict[str, Any]:
        model.eval()
        total_tp, total_fp, total_fn = 0, 0, 0
        total_dustbin_correct, total_dustbin_items = 0, 0
        cross_cat_fp, same_cat_fp = 0, 0

        pbar = tqdm(loader, desc="Evaluating", bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")
        with torch.no_grad():
            for batch in pbar:
                patches0 = batch["patches0"].to(device)
                patches1 = batch["patches1"].to(device)
                out = model(patches0, patches1, **build_object_forward_kwargs(batch, device))
                pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                    out["prob_matrix"][0],
                    thresh=0.2,
                    min_area_ratio=0.0,
                )

                gt_matches = set(batch["gt_matches"][0])
                pred_set = set(pred_matches)

                cur_tp = len(pred_set.intersection(gt_matches))
                cur_fp_set = pred_set - gt_matches
                cur_fn = len(gt_matches - pred_set)

                total_tp += cur_tp
                total_fp += len(cur_fp_set)
                total_fn += cur_fn

                gt_un0 = set(batch["unmatched_0"][0])
                gt_un1 = set(batch["unmatched_1"][0])
                total_dustbin_correct += len(set(pred_un0).intersection(gt_un0)) + len(set(pred_un1).intersection(gt_un1))
                total_dustbin_items += len(gt_un0) + len(gt_un1)

                objs0 = batch.get("objs0", [])
                objs1 = batch.get("objs1", [])
                for i, j in cur_fp_set:
                    c0 = objs0[i].get("category", "") if i < len(objs0) else ""
                    c1 = objs1[j].get("category", "") if j < len(objs1) else ""
                    if c0 and c1 and c0 != c1:
                        cross_cat_fp += 1
                    else:
                        same_cat_fp += 1

        precision = total_tp / (total_tp + total_fp + 1e-6) * 100.0
        recall = total_tp / (total_tp + total_fn + 1e-6) * 100.0
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
        dustbin_acc = total_dustbin_correct / (total_dustbin_items + 1e-6) * 100.0 if total_dustbin_items > 0 else 100.0
        cross_cat_rate = cross_cat_fp / (total_fp + 1e-6) * 100.0 if total_fp > 0 else 0.0

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "dustbin_acc": round(dustbin_acc, 2),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "cross_cat_fp": cross_cat_fp,
            "cross_cat_rate": round(cross_cat_rate, 2),
            "total_pairs": len(loader.dataset),
        }

    def visualize(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        out_dir: Path,
        num_vis: int = 5,
        min_area_ratio: float = 0.50,
        data_dir: Optional[Path] = None,
        use_region_ds: bool = True,
    ) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        model.eval()

        pbar = tqdm(loader, desc="Visualizing", bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")
        with torch.no_grad():
            for vis_idx, batch in enumerate(pbar):
                if vis_idx >= num_vis:
                    break

                patches0 = batch["patches0"].to(device)
                patches1 = batch["patches1"].to(device)
                out = model(patches0, patches1, **build_object_forward_kwargs(batch, device))
                pred_matches, pred_un0, pred_un1 = predict_bipartite_matches(
                    out["prob_matrix"][0],
                    thresh=0.2,
                    min_area_ratio=0.0,
                )

                scene_tag = batch.get("scene_name", f"sample_{vis_idx + 1}")
                save_file = out_dir / f"eval_sample_{vis_idx + 1:02d}_{scene_tag}.jpg"
                wrote = save_pair_visualization(
                    batch,
                    out,
                    save_file,
                    title=f"Sample {vis_idx + 1}: {scene_tag}",
                    pred_matches=pred_matches,
                    pred_un0=pred_un0,
                    pred_un1=pred_un1,
                    data_dir=data_dir,
                    scene_name=scene_tag,
                    use_region_ds=use_region_ds,
                )
                if wrote:
                    saved_paths.append(save_file)
                    print(f"  [✓] Exported visualization: {save_file.name}")

        return saved_paths


class LineMatchingEvaluator(BaseEvaluator):
    """Evaluator for 2D Line Matching Models."""

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        min_area_ratio: float = 0.50,
    ) -> Dict[str, Any]:
        print("[*] Running Line Matching Benchmark...")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def visualize(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        out_dir: Path,
        num_vis: int = 5,
        min_area_ratio: float = 0.50,
        data_dir: Optional[Path] = None,
        use_region_ds: bool = True,
    ) -> List[Path]:
        return []


class HomographyEvaluator(BaseEvaluator):
    """Reports the cycle loss and the ridge depth loss, and draws warped ridges."""

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        min_area_ratio: float = 0.50,
    ) -> Dict[str, Any]:
        from models import build_loss

        del min_area_ratio
        criterion = build_loss("FCN_field")
        model.eval()
        totals = {"l1": 0.0, "l2": 0.0, "loss": 0.0}
        steps = 0
        with torch.inference_mode():
            for batch in loader:
                moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                pred = model(moved["image0"], moved["image1"])
                out = criterion(pred, moved)
                totals["l1"] += float(out["l1"])
                totals["l2"] += float(out["l2"])
                totals["loss"] += float(out["loss"])
                steps += 1
        denom = max(steps, 1)
        metrics = {key: value / denom for key, value in totals.items()}
        metrics.update(
            {
                "total_pairs": steps,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "dustbin_acc": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }
        )
        return metrics

    def visualize(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        out_dir: Path,
        num_vis: int = 5,
        min_area_ratio: float = 0.50,
        data_dir: Optional[Path] = None,
        use_region_ds: bool = True,
    ) -> List[Path]:
        import cv2
        import numpy as np
        from models.FCN_field.draw import draw_ridge_lines
        from models.FCN_field.loss import warp_by_velocity

        del min_area_ratio, data_dir, use_region_ds
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                if len(paths) >= num_vis:
                    break
                moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                pred = model(moved["image0"], moved["image1"])
                ridges = moved["ridges0"]
                mask = moved["ridge_mask"][0].bool()
                kept = ridges[0, mask].detach().cpu().numpy()
                if len(kept) == 0:
                    endpoints = np.zeros((0, 2, 2), dtype=np.float32)
                    warped_ends = endpoints
                else:
                    endpoints = np.stack([kept[:, 0], kept[:, -1]], axis=1)
                    warped = warp_by_velocity(pred["velocity12"].float(), ridges.float())
                    warped_kept = warped[0, mask].detach().cpu().numpy()
                    warped_ends = np.stack([warped_kept[:, 0], warped_kept[:, -1]], axis=1)
                left = draw_ridge_lines(batch["bgr0"][0].numpy(), endpoints)
                right = draw_ridge_lines(batch["bgr1"][0].numpy(), warped_ends)
                canvas = np.concatenate([left, right], axis=1)
                dest = out_dir / f"{batch['pair_id'][0]}.jpg"
                cv2.imwrite(str(dest), canvas)
                paths.append(dest)
        return paths


class EvaluatorFactory:
    """Factory creating task-specific evaluators."""

    _registry: Dict[str, type[BaseEvaluator]] = {
        "object": ObjectMatchingEvaluator,
        "line": LineMatchingEvaluator,
        "homography": HomographyEvaluator,
    }

    @classmethod
    def register(cls, family: str, evaluator_cls: type[BaseEvaluator]) -> None:
        cls._registry[family.lower()] = evaluator_cls

    @classmethod
    def create(cls, model_info: Dict[str, Any] | str) -> BaseEvaluator:
        if isinstance(model_info, str):
            family = model_info.lower()
        else:
            family = model_info.get("family", "object").lower()
        if family not in cls._registry:
            raise ValueError(f"Unknown model family '{family}'. Available: {list(cls._registry.keys())}")
        return cls._registry[family]()


def evaluate(
    checkpoint: str | Path,
    data_dir: str | Path = "data/structured3d_semantic_regions",
    split: str = "test",
    out_dir: str | Path = "outputs/semantic_matching_vis",
    num_vis: int = 5,
    min_area_ratio: float = 0.50,
    model_type: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Main Factory Function for Unified Evaluation across all models and checkpoints.
    """
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()

    model, model_info, ckpt_path = ModelFactory.create_from_checkpoint(checkpoint, dev, model_type_hint=model_type)

    arch_desc = []
    if model_info.get("use_pyramid"):
        arch_desc.append("Multi-Scale FPN Pyramid")
    if model_info.get("with_geometry"):
        arch_desc.append("Layout Geometry MLP")
    if not arch_desc:
        arch_desc.append("Single-Scale Backbone")

    print("\n" + "=" * 75)
    print("      Unified Evaluator Factory: Model & Benchmark Suite        ")
    print("=" * 75)
    print(f"  Model Type:        {model_info['model_type']} ({', '.join(arch_desc)})")
    print(f"  Dimensions:        d_model={model_info['d_model']}, feat_dim={model_info['feat_dim']}")
    print(f"  Layers & Heads:    layers={model_info['num_layers']}, nhead={model_info['nhead']}")
    print(f"  Checkpoint Path:   {ckpt_path}")
    print(f"  Dataset Directory: {p_data.name} (Split: '{split}')")
    print(f"  Min Area Ratio:    {min_area_ratio * 100:.0f}%")
    print(f"  Output Directory:  {p_out}")
    print("=" * 75 + "\n")

    if model_info.get("family") == "homography" and any(
        token == "--min-area-ratio" or token.startswith("--min-area-ratio=") for token in sys.argv[1:]
    ):
        print("[!] --min-area-ratio 对 FCN_field 不会执行，已忽略", flush=True)

    loader, use_region_ds = DatasetFactory.create(
        model_info=model_info,
        data_dir=p_data,
        split=split,
        min_area_ratio=min_area_ratio,
        num_workers=num_workers,
    )

    evaluator = EvaluatorFactory.create(model_info)

    t0 = time.time()
    metrics = evaluator.evaluate(model, loader, dev, min_area_ratio=min_area_ratio)
    elapsed = time.time() - t0

    print("---------------------------------------------------------------------------")
    if model_info.get("family") == "homography":
        print(f"HOMOGRAPHY EVALUATION ({metrics['total_pairs']} pairs, {elapsed:.2f}s):")
        print(f"  L1 cycle           : {metrics['l1']:.4f}")
        print(f"  L2 depth profiles  : {metrics['l2']:.4f}")
        print(f"  Total loss         : {metrics['loss']:.4f}")
    else:
        print(f"BENCHMARK EVALUATION RESULTS ({metrics['total_pairs']} pairs, {elapsed:.2f}s):")
        print(f"  Match Precision    : {metrics['precision']:.2f}%")
        print(f"  Match Recall       : {metrics['recall']:.2f}%")
        print(f"  Match F1-Score     : {metrics['f1']:.2f}%")
        print(f"  Dustbin Accuracy   : {metrics['dustbin_acc']:.2f}%")
        print(f"  Correspondences    : {metrics['tp']} TP, {metrics['fp']} FP, {metrics['fn']} FN")
        if "cross_cat_fp" in metrics:
            print(f"  Cross-Category FP  : {metrics['cross_cat_fp']} ({metrics['cross_cat_rate']:.1f}% of all FPs)")
    print("---------------------------------------------------------------------------\n")

    if num_vis > 0:
        print(f"[*] Rendering {num_vis} diagnostic visualization samples...")
        vis_paths = evaluator.visualize(
            model=model,
            loader=loader,
            device=dev,
            out_dir=p_out,
            num_vis=num_vis,
            min_area_ratio=min_area_ratio,
            data_dir=p_data,
            use_region_ds=use_region_ds,
        )
        metrics["visualizations"] = [str(p) for p in vis_paths]

    metrics["checkpoint"] = str(ckpt_path)
    metrics["model_info"] = model_info
    metrics["evaluated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    p_out.mkdir(parents=True, exist_ok=True)
    eval_json_path = p_out / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    eval_md_path = p_out / "eval_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# 评测报告 (Evaluation Report)\n\n")
        f.write(f"> **评测时间**: `{metrics['evaluated_at']}`  \n")
        f.write(f"> **模型检查点**: `{ckpt_path}`  \n")
        f.write(f"> **评测数据集**: `{p_data.name}` (划分: `{split}`)  \n")
        f.write(f"> **耗时**: `{elapsed:.2f}s` | **测试对数**: `{metrics['total_pairs']}`\n\n")
        f.write("## 核心指标结果\n")
        if model_info.get("family") == "homography":
            f.write("| 评测指标 | 结果数值 |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **L1 cycle** | **{metrics['l1']:.4f}** |\n")
            f.write(f"| **L2 depth** | **{metrics['l2']:.4f}** |\n")
            f.write(f"| **Total** | **{metrics['loss']:.4f}** |\n\n")
        else:
            f.write("| 评测指标 | 结果数值 | 说明 |\n")
            f.write("| :--- | :---: | :--- |\n")
            f.write(f"| **Match Precision** | **{metrics['precision']:.2f}%** | 匹配准确率 (查准率) |\n")
            f.write(f"| **Match Recall** | **{metrics['recall']:.2f}%** | 目标召回率 (查全率) |\n")
            f.write(f"| **Match F1-Score** | **{metrics['f1']:.2f}%** | 核心综合性能指标 |\n")
            f.write(f"| **Dustbin Accuracy** | **{metrics['dustbin_acc']:.2f}%** | 垃圾桶正确率 |\n")
            f.write(f"| **True Positives (TP)** | {metrics['tp']} | 正确匹配数 |\n")
            f.write(f"| **False Positives (FP)** | {metrics['fp']} | 错误匹配数 |\n")
            f.write(f"| **False Negatives (FN)** | {metrics['fn']} | 漏检未配对数 |\n")
            if "cross_cat_fp" in metrics:
                f.write(f"| **Cross-Category FP** | {metrics['cross_cat_fp']} ({metrics['cross_cat_rate']:.1f}%) | 跨类别严重误匹配 |\n")
            f.write("\n")
        if num_vis > 0 and "visualizations" in metrics:
            f.write("## 诊断可视化列表\n")
            for vp in metrics["visualizations"]:
                f.write(f"- [{Path(vp).name}](file://{vp})\n")
            f.write("\n")

    print(f"[✓] Saved evaluation metrics to : {eval_json_path}")
    print(f"[✓] Saved evaluation report to  : {eval_md_path}")

    # Automatically backfill test_benchmark into run's config.json if present
    cfg_file = ckpt_path.parent / "config.json"
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cdata = json.load(f)
            cdata["test_benchmark"] = {
                "raw_visual": {
                    "precision": round(metrics.get("precision", 0.0), 2),
                    "recall": round(metrics.get("recall", 0.0), 2),
                    "f1": round(metrics.get("f1", 0.0), 2),
                    "tp": metrics.get("tp", 0),
                    "fp": metrics.get("fp", 0),
                    "fn": metrics.get("fn", 0),
                    "cross_cat_fp": metrics.get("cross_cat_fp", 0),
                    "cross_cat_rate": round(metrics.get("cross_cat_rate", 0.0), 2),
                    "dustbin_precision": round(metrics.get("dustbin_prec", metrics.get("dustbin_acc", 0.0)), 2),
                    "dustbin_recall": round(metrics.get("dustbin_rec", 0.0), 2),
                },
                "semantic_constrained": {
                    "precision": round(metrics.get("precision", 0.0), 2),
                    "recall": round(metrics.get("recall", 0.0), 2),
                    "f1": round(metrics.get("f1", 0.0), 2),
                    "tp": metrics.get("tp", 0),
                    "fp": metrics.get("fp", 0),
                    "fn": metrics.get("fn", 0),
                },
            }
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(cdata, f, indent=2, ensure_ascii=False)
            print(f"[✓] Backfilled benchmark metrics into: {cfg_file}")
        except Exception as e:
            print(f"[!] Warning: Failed to update config.json with test metrics: {e}")

    return metrics
