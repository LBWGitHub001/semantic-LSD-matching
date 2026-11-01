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
import numpy as np

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
    """Reports the symmetry loss and draws ridges moved by v12."""

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
        totals = {"symmetry": 0.0, "loss": 0.0}
        steps = 0
        with torch.inference_mode():
            for batch in loader:
                moved = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                pred = model(moved["image0"], moved["image1"])
                out = criterion(pred, moved)
                totals["symmetry"] += float(out["symmetry"])
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


class PointMatchingEvaluator(BaseEvaluator):
    """Evaluates SuperPoint endpoint detection, line geometry reconstruction, and metric matching."""

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        min_area_ratio: float = 0.50,
        eval_stage: str = "all",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from models import build_loss
        from visualization.point_matches import (
            match_accuracy_mma,
            calculate_corner_repeatability,
            extract_detected_lines,
        )

        del min_area_ratio, kwargs
        eval_stage = eval_stage.lower()
        criterion = build_loss("SuperPoint", stage=eval_stage).to(device)
        model.eval()

        total_loss = 0.0
        total_ld = 0.0
        total_l_det = 0.0
        total_l_orphan = 0.0
        total_l_len = 0.0
        total_l_prem = 0.0

        total_nn_acc = 0.0
        total_mma_1px = 0.0
        total_mma_2px = 0.0
        total_mma_3px = 0.0

        total_rep_15 = 0.0
        total_rep_30 = 0.0
        total_loc_err = 0.0

        total_corner_tp = 0
        total_corner_fp = 0
        total_corner_fn = 0

        total_lines_v0 = 0
        total_lines_v1 = 0
        total_line_len = 0.0
        total_line_df = 0.0
        max_line_len_overall = 0.0
        total_orphan_rate = 0.0
        steps = 0

        pbar = tqdm(loader, desc="Evaluating SuperPoint", bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")
        with torch.inference_mode():
            for batch in pbar:
                moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                img0, img1 = moved["image0"], moved["image1"]
                h_mat = moved["homography"]
                pts0_gt, pts1_gt = moved.get("points0"), moved.get("points1")
                _, _, height, width = img0.shape

                pred0 = model(img0)
                pred1 = model(img1)

                # 1. 完整计算 5 项联合损失
                out = criterion(
                    pred0,
                    pred1,
                    h_mat,
                    image0=img0,
                    image1=img1,
                    points0=pts0_gt,
                    points1=pts1_gt,
                )
                step_loss = float(out["loss"])
                total_loss += step_loss
                total_ld += float(out.get("ld", 0.0))
                total_l_det += float(out.get("l_det", 0.0))
                total_l_orphan += float(out.get("l_orphan", 0.0))
                total_l_len += float(out.get("l_len", 0.0))
                total_l_prem += float(out.get("l_prem", 0.0))

                # 2. 描述子匹配指标 (NN Acc & MMA)
                if eval_stage in ("all", "descriptor"):
                    mma_res = match_accuracy_mma(pred0["descriptor"], pred1["descriptor"], pts0_gt, pts1_gt, h_mat)
                    total_nn_acc += mma_res["nn_acc"]
                    total_mma_1px += mma_res["mma_1.0px"]
                    total_mma_2px += mma_res["mma_2.0px"]
                    total_mma_3px += mma_res["mma_3.0px"]

                # 3. 直线几何拓扑提取与孤立率
                if eval_stage in ("all", "detector"):
                    geom0 = extract_detected_lines(pred0["heatmap"], img0[:, 3:4], img0[:, 4:5], criterion)
                    geom1 = extract_detected_lines(pred1["heatmap"], img1[:, 3:4], img1[:, 4:5], criterion)

                    total_lines_v0 += geom0["num_lines"]
                    total_lines_v1 += geom1["num_lines"]
                    avg_pair_len = (geom0["mean_length"] + geom1["mean_length"]) / 2.0
                    total_line_len += avg_pair_len
                    total_line_df += (geom0["mean_df"] + geom1["mean_df"]) / 2.0
                    max_line_len_overall = max(max_line_len_overall, geom0["max_length"], geom1["max_length"])
                    total_orphan_rate += (geom0["orphan_rate"] + geom1["orphan_rate"]) / 2.0

                    # 4. 角点跨视角重复率
                    rep15, _ = calculate_corner_repeatability(geom0["endpoints"], geom1["endpoints"], h_mat, height, width, 1.5)
                    rep30, err30 = calculate_corner_repeatability(geom0["endpoints"], geom1["endpoints"], h_mat, height, width, 3.0)
                    total_rep_15 += rep15
                    total_rep_30 += rep30
                    total_loc_err += err30

                    # 5. 端点查准率与查全率 (以 3px 为阈值对比 GT 角点)
                    if pts0_gt is not None and len(pts0_gt) > 0:
                        gt_np = pts0_gt.detach().cpu().numpy()
                        pred_pts = geom0["endpoints"]
                        if len(pred_pts) > 0:
                            dist_to_gt = np.linalg.norm(pred_pts[:, None, :] - gt_np[None, :, :], axis=-1)
                            tp_pred = (dist_to_gt.min(axis=1) <= 3.0).sum()
                            fp_pred = len(pred_pts) - tp_pred
                            dist_to_pred = dist_to_gt.T
                            tp_gt = (dist_to_pred.min(axis=1) <= 3.0).sum()
                            fn_gt = len(gt_np) - tp_gt

                            total_corner_tp += int(tp_pred)
                            total_corner_fp += int(fp_pred)
                            total_corner_fn += int(fn_gt)

                steps += 1

        denom = max(steps, 1)
        mean_nn_acc = total_nn_acc / denom
        corner_prec = 100.0 * total_corner_tp / max(total_corner_tp + total_corner_fp, 1)
        corner_rec = 100.0 * total_corner_tp / max(total_corner_tp + total_corner_fn, 1)
        corner_f1 = 2 * corner_prec * corner_rec / max(corner_prec + corner_rec, 1e-4)

        metrics = {
            "loss": total_loss / denom,
            "loss_desc": total_ld / denom,
            "loss_det": total_l_det / denom,
            "loss_orphan": total_l_orphan / denom,
            "loss_len": total_l_len / denom,
            "loss_prem": total_l_prem / denom,
            "match_acc": mean_nn_acc,
            "mma_1px": total_mma_1px / denom,
            "mma_2px": total_mma_2px / denom,
            "mma_3px": total_mma_3px / denom,
            "corner_repeatability_1_5px": total_rep_15 / denom,
            "corner_repeatability_3_0px": total_rep_30 / denom,
            "corner_loc_error": total_loc_err / denom,
            "corner_precision": corner_prec,
            "corner_recall": corner_rec,
            "corner_f1": corner_f1,
            "valid_lines_per_view": (total_lines_v0 + total_lines_v1) / (2.0 * denom),
            "mean_line_length": total_line_len / denom,
            "max_line_length": max_line_len_overall,
            "line_df_alignment": total_line_df / denom,
            "orphan_rate": total_orphan_rate / denom,
            "total_pairs": steps,
            # 通用基准字段兼容
            "precision": mean_nn_acc * 100.0,
            "recall": mean_nn_acc * 100.0,
            "f1": mean_nn_acc * 100.0,
            "dustbin_acc": 0.0,
            "tp": int(total_nn_acc * 100),
            "fp": 0,
            "fn": 0,
        }
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
        eval_stage: str = "all",
        **kwargs: Any,
    ) -> List[Path]:
        from models import build_loss
        from visualization.point_matches import (
            match_accuracy,
            calculate_corner_repeatability,
            save_detection_mask_lines_figure,
            save_descriptor_matching_figure,
            save_superpoint_diagnostic_sheet,
        )

        del min_area_ratio, data_dir, use_region_ds, kwargs
        eval_stage = eval_stage.lower()
        criterion = build_loss("SuperPoint", stage=eval_stage).to(device)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        model.eval()

        with torch.inference_mode():
            for batch in loader:
                if len(paths) >= num_vis:
                    break
                moved = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                img0, img1 = moved["image0"], moved["image1"]
                h_mat = moved["homography"]
                pts0_gt, pts1_gt = moved.get("points0"), moved.get("points1")
                pair_id = batch["pair_id"]

                pred0 = model(img0)
                pred1 = model(img1)

                loss_out = criterion(
                    pred0,
                    pred1,
                    h_mat,
                    image0=img0,
                    image1=img1,
                    points0=pts0_gt,
                    points1=pts1_gt,
                )

                metrics_summary = {
                    "loss": float(loss_out["loss"]),
                    "ld": float(loss_out.get("ld", 0.0)),
                    "l_det": float(loss_out.get("l_det", 0.0)),
                    "l_orphan": float(loss_out.get("l_orphan", 0.0)),
                    "l_len": float(loss_out.get("l_len", 0.0)),
                    "l_prem": float(loss_out.get("l_prem", 0.0)),
                }

                # 1. 检测验证: 原始掩膜 + 直线 (*_detection.jpg)
                if eval_stage in ("detector", "all"):
                    det_dest = out_dir / f"{pair_id}_detection.jpg"
                    save_detection_mask_lines_figure(
                        det_dest,
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0,
                        pred1,
                        img0[:, 3:4],
                        img1[:, 3:4],
                        img0[:, 4:5],
                        img1[:, 4:5],
                        criterion,
                        pair_id=pair_id,
                        metrics_summary=metrics_summary,
                        points0=pts0_gt,
                        points1=pts1_gt,
                    )
                    paths.append(det_dest)

                # 2. 描述验证: 配对情况 (*_matching.jpg)
                if eval_stage in ("descriptor", "all") and pts0_gt is not None and pts1_gt is not None:
                    desc_dest = out_dir / f"{pair_id}_matching.jpg"
                    save_descriptor_matching_figure(
                        desc_dest,
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0["descriptor"],
                        pred1["descriptor"],
                        pts0_gt,
                        pts1_gt,
                        h_mat,
                        pair_id=pair_id,
                        metrics_summary=metrics_summary,
                    )
                    paths.append(desc_dest)

                # 3. 联合全景诊断大图 (当 eval_stage == 'all' 且尚未生成时保存)
                if eval_stage == "all":
                    diag_dest = out_dir / f"{pair_id}.jpg"
                    save_superpoint_diagnostic_sheet(
                        diag_dest,
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0,
                        pred1,
                        img0[:, 3:4],
                        img1[:, 3:4],
                        img0[:, 4:5],
                        img1[:, 4:5],
                        pts0_gt,
                        pts1_gt,
                        h_mat,
                        pair_id,
                        metrics_summary,
                        criterion,
                    )
                    paths.append(diag_dest)
        return paths


class EvaluatorFactory:
    """Factory creating task-specific evaluators."""

    _registry: Dict[str, type[BaseEvaluator]] = {
        "object": ObjectMatchingEvaluator,
        "line": LineMatchingEvaluator,
        "homography": HomographyEvaluator,
        "point": PointMatchingEvaluator,
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
    eval_stage: str = "all",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Main Factory Function for Unified Evaluation across all models and checkpoints.
    """
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()
    eval_stage = eval_stage.lower()

    model, model_info, ckpt_path = ModelFactory.create_from_checkpoint(checkpoint, dev, model_type_hint=model_type)

    if model_info.get("family") == "point" and str(data_dir) == "data/structured3d_semantic_regions":
        p_data = Path("data/planar_letters").resolve()
    elif model_info.get("family") == "homography" and str(data_dir) == "data/structured3d_semantic_regions":
        p_data = Path("data/homo_matcher").resolve()

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
    if model_info.get("family") == "point":
        metrics = evaluator.evaluate(model, loader, dev, min_area_ratio=min_area_ratio, eval_stage=eval_stage)
    else:
        metrics = evaluator.evaluate(model, loader, dev, min_area_ratio=min_area_ratio)
    elapsed = time.time() - t0

    print("---------------------------------------------------------------------------")
    if model_info.get("family") == "homography":
        print(f"HOMOGRAPHY EVALUATION ({metrics['total_pairs']} pairs, {elapsed:.2f}s):")
        print(f"  Symmetry T12/T21   : {metrics['symmetry']:.4f}")
        print(f"  Total loss         : {metrics['loss']:.4f}")
    elif model_info.get("family") == "point":
        print(f"SUPERPOINT [{eval_stage.upper()}] BENCHMARK EVALUATION ({metrics['total_pairs']} pairs, {elapsed:.2f}s):")
        if eval_stage in ("all", "descriptor"):
            print(f"  [1] Descriptor Loss: L_desc={metrics['loss_desc']:.4f} (Total Loss={metrics['loss']:.4f})")
            print(f"  [2] Metric Match   : NN Acc={metrics['match_acc']*100:.1f}% | MMA@3px={metrics['mma_3px']*100:.1f}% | MMA@1px={metrics['mma_1px']*100:.1f}%")
        if eval_stage in ("all", "detector"):
            print(f"  [3] Endpoint Det   : Rep@3.0px={metrics['corner_repeatability_3_0px']*100:.1f}% | Rep@1.5px={metrics['corner_repeatability_1_5px']*100:.1f}% | LocErr={metrics['corner_loc_error']:.2f}px")
            print(f"                       Corner Precision={metrics['corner_precision']:.1f}% | Recall={metrics['corner_recall']:.1f}% | F1={metrics['corner_f1']:.1f}%")
            print(f"  [4] Line Geometry  : Valid Lines={metrics['valid_lines_per_view']:.1f}/view | MeanLen={metrics['mean_line_length']:.1f}px | MaxLen={metrics['max_line_length']:.1f}px")
            print(f"                       DF Alignment={metrics['line_df_alignment']:.2f} | Orphan Rate={metrics['orphan_rate']*100:.1f}%")
            print(f"  [5] Detector Losses: det={metrics['loss_det']:.3f} | orph={metrics['loss_orphan']:.3f} | len={metrics['loss_len']:.3f} | prem={metrics['loss_prem']:.3f}")
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
        print(f"[*] Rendering {num_vis} diagnostic visualization samples (stage: {eval_stage})...")
        vis_kwargs = {
            "model": model,
            "loader": loader,
            "device": dev,
            "out_dir": p_out,
            "num_vis": num_vis,
            "min_area_ratio": min_area_ratio,
            "data_dir": p_data,
            "use_region_ds": use_region_ds,
        }
        if model_info.get("family") == "point":
            vis_kwargs["eval_stage"] = eval_stage
        vis_paths = evaluator.visualize(**vis_kwargs)
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
            f.write(f"| **Symmetry T12/T21** | **{metrics['symmetry']:.4f}** |\n")
            f.write(f"| **Total** | **{metrics['loss']:.4f}** |\n\n")
        elif model_info.get("family") == "point":
            f.write("### 1. 损失函数细粒度分解 (Loss Breakdown)\n")
            f.write("| 损失分项 | 数值 | 权重与物理目标 |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **总联合损失 (Total Loss)** | **`{metrics['loss']:.4f}`** | 联合优化目标 |\n")
            f.write(f"| 描述子合页损失 (L_desc) | `{metrics['loss_desc']:.4f}` | 250x 正样本加权，同名点特征对齐 |\n")
            f.write(f"| 端点检测 BCE 损失 (L_det) | `{metrics['loss_det']:.4f}` | 角点真值正向监督，防止零塌陷 |\n")
            f.write(f"| 孤立端点惩罚损失 (L_orphan) | `{metrics['loss_orphan']:.4f}` | 惩罚未连成合法直线的孤立端点 |\n")
            f.write(f"| 超线性长线收益奖励 (L_len) | `{metrics['loss_len']:.4f}` | 指数 1.5 加权，极大化直线跨度 |\n")
            f.write(f"| 射线外推早退探针 (L_prem) | `{metrics['loss_prem']:.4f}` | 端点外侧 4px 探针推动至物理线尾 |\n\n")

            f.write("### 2. 端点检测质量与跨视角重复率 (Endpoint Detection Quality)\n")
            f.write("| 评测指标 | 结果数值 | 说明 |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **角点重复率 (Repeatability @ 3.0px)** | **`{metrics['corner_repeatability_3_0px']*100:.2f}%`** | 跨视角单应变换后成功重现的端点比例 |\n")
            f.write(f"| 角点重复率 (Repeatability @ 1.5px) | `{metrics['corner_repeatability_1_5px']*100:.2f}%` | 高精度几何一致性重现比例 |\n")
            f.write(f"| 亚像素定位均方误差 (Loc Error) | `{metrics['corner_loc_error']:.2f}px` | 重投影中心像素偏移误差 |\n")
            f.write(f"| 端点查准率 (Corner Precision) | `{metrics['corner_precision']:.2f}%` | 预测端点命中真值角点的比例 |\n")
            f.write(f"| 端点查全率 (Corner Recall) | `{metrics['corner_recall']:.2f}%` | 真值角点被成功检测出的比例 |\n")
            f.write(f"| 端点综合 F1-Score | `{metrics['corner_f1']:.2f}%` | 端点检测综合性能调和值 |\n\n")

            f.write("### 3. 长线几何拓扑重构与长线收益 (Line Geometry Quality)\n")
            f.write("| 几何拓扑指标 | 结果数值 | 说明 |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **平均有效直线数 (Valid Lines / View)** | **`{metrics['valid_lines_per_view']:.1f}`** | 满足 DF>=0.55 与 AF极差<=0.10 的线段数 |\n")
            f.write(f"| **平均直线长度 (Mean Line Length)** | **`{metrics['mean_line_length']:.1f}px`** | 验证超线性长线奖励拉长效果 |\n")
            f.write(f"| 最长直线跨度 (Max Line Length) | `{metrics['max_line_length']:.1f}px` | 画面中被连通的最长直线段 |\n")
            f.write(f"| 直线距离场对齐度 (DF Alignment) | `{metrics['line_df_alignment']:.2f}` | 沿重构直线采样的平均对数距离场数值 |\n")
            f.write(f"| 孤立端点率 (Orphan Rate) | `{metrics['orphan_rate']*100:.2f}%` | 未参与任何合法线段的端点比例 (越低越好) |\n\n")

            f.write("### 4. 跨视角描述子度量匹配精度 (Cross-View Matching)\n")
            f.write("| 匹配指标 | 结果数值 | 说明 |\n")
            f.write("| :--- | :---: |\n")
            f.write(f"| **最近邻匹配准确率 (NN Match Accuracy)** | **`{metrics['match_acc']*100:.2f}%`** | 256 维特征度量最近邻正确率 |\n")
            f.write(f"| 平均匹配精度 (MMA @ 3.0px) | `{metrics['mma_3px']*100:.2f}%` | 误差 3px 内匹配对应率 |\n")
            f.write(f"| 平均匹配精度 (MMA @ 2.0px) | `{metrics['mma_2px']*100:.2f}%` | 误差 2px 内匹配对应率 |\n")
            f.write(f"| 平均匹配精度 (MMA @ 1.0px) | `{metrics['mma_1px']*100:.2f}%` | 误差 1px 内匹配对应率 |\n\n")
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
