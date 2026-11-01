"""SuperPoint 训练与评测助手工具库 (Helper Classes & Pipeline Utilities).

将数据批次搬运、设备调度、指标累加统计、tqdm 进度条格式化、权重保存与可视化渲染
全部从 train.py 与 eval.py 中剥离下沉至此类中，使主程序保持极简骨架。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
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
from datasets.point_match_dataset import PointMatchDataset, collate_point_matches
from models.superpoint.model import SuperPoint
from models.superpoint.loss import SuperPointLoss, STAGE_LOSS_WEIGHTS
from training_utils import (
    TQDM_BAR_FORMAT,
    create_experiment_dir,
    get_git_commit,
    safe_load_checkpoint,
    save_config_json,
)
from visualization.point_matches import (
    calculate_corner_repeatability,
    extract_detected_lines,
    match_accuracy,
    match_accuracy_mma,
    save_descriptor_matching_figure,
    save_detection_mask_lines_figure,
    save_line_descriptor_pca_figure,
)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """递归将批次中的张量移送至目标计算设备。"""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


# =============================================================================
# 1. 训练管线助手类 (SuperPointTrainerHelper)
# =============================================================================

class SuperPointTrainerHelper:
    """SuperPoint 训练管线助手类，负责环境搭建、逐步阶段调度、单步前向反向与持久化。"""

    def __init__(self, args: argparse.Namespace | None = None, **kwargs: Any) -> None:
        if args is None:
            parser = build_train_parser()
            args = parser.parse_args([])
        for k, v in kwargs.items():
            setattr(args, k, v)
        self.args = args

        self.data_dir = Path(args.data_dir or "data/planar_letters")
        self.base_save_dir = Path(args.save_dir or "checkpoints/SuperPoint")
        self.train_stage = getattr(args, "train_stage", "joint").lower()
        self.device = torch.device(args.device)
        self.epochs = int(args.epochs)
        self.lr = float(args.lr if args.lr is not None else 1e-3)
        self.weight_decay = float(getattr(args, "weight_decay", 1e-4))
        self.max_steps = getattr(args, "max_steps", None)

        self.save_dir, self.run_name = create_experiment_dir(
            base_save_dir=self.base_save_dir,
            model_name=f"SuperPoint_{self.train_stage}+{self.data_dir.name}",
            label=getattr(args, "label", None),
            no_timestamp=getattr(args, "no_timestamp", False),
            run_name=getattr(args, "run_name", None),
        )
        self.vis_dir = self.save_dir / "visualizations"
        self.config_path = self.save_dir / "config.json"

        # 课程调度表 (如果启用逐步训练)
        self.curriculum = getattr(args, "curriculum", None)
        self.loss_kwargs = {
            k: getattr(args, k)
            for k in (
                "weight_dist", "weight_ang",
                "weight_desc", "positive_weight", "positive_margin", "negative_margin",
                "weight_det", "det_pos_weight", "weight_orphan", "weight_len",
                "length_exponent", "weight_premature", "probe_dist",
                "tau_df", "tau_ang", "min_length", "num_candidates", "num_line_samples",
                "nms_radius", "weight_intra", "weight_cross", "weight_inter",
                "weight_bg", "line_neg_margin", "bg_neg_margin",
                "num_line_points", "num_bg_samples",
            )
            if hasattr(args, k)
        }

    def setup(self) -> tuple[SuperPoint, SuperPointLoss, torch.optim.Optimizer, DataLoader, DataLoader]:
        """初始化数据集、模型、损失函数与优化器。"""
        train_ds = PointMatchDataset(self.data_dir, split="train")
        val_ds = PointMatchDataset(self.data_dir, split="val")
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_point_matches, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_point_matches, num_workers=0)

        model = SuperPoint().to(self.device)
        loss_fn = SuperPointLoss(stage=self.train_stage, **self.loss_kwargs).to(self.device)

        if hasattr(model, "set_train_stage"):
            model.set_train_stage(self.train_stage)
            trainable_params = model.get_trainable_parameters(self.train_stage)
        else:
            trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        # 记录初始配置
        config_data = {
            "experiment_info": {
                "run_name": self.run_name,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": None,
                "git_commit": get_git_commit(),
                "command": " ".join(sys.argv),
            },
            "model": {"model_type": "SuperPoint", "family": "point", "train_stage": self.train_stage},
            "hyperparameters": {
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "loss": f"superpoint_{self.train_stage}",
                "trainable_param_groups": len(trainable_params),
                "loss_weights": self.loss_kwargs,
            },
        }
        save_config_json(self.config_path, config_data)
        print(f"[*] SuperPoint stage: '{self.train_stage}' | train={len(train_ds)} val={len(val_ds)} | device={self.device}")

        return model, loss_fn, optimizer, train_loader, val_loader

    def update_stage(self, model: SuperPoint, loss_fn: SuperPointLoss, epoch: int) -> str:
        """根据当前 Epoch 判定并应用训练阶段 (支持静态阶段与课程递进)。"""
        stage = self.train_stage
        if getattr(self.args, "progressive", False) and self.curriculum and isinstance(self.curriculum, list):
            for item in self.curriculum:
                ep_range = item.get("epochs", [1, self.epochs])
                if ep_range[0] <= epoch <= ep_range[1]:
                    stage = item.get("stage", stage)
                    break

        loss_fn.set_stage(stage)
        if hasattr(model, "set_train_stage"):
            model.set_train_stage(stage)
        return stage

    def progress_bar(self, loader: DataLoader, epoch: int, stage: str) -> Any:
        total = len(loader) if self.max_steps is None else min(len(loader), self.max_steps)
        return tqdm(
            loader,
            total=total,
            desc=f"Epoch {epoch:02d}/{self.epochs:02d} [{stage}]",
            bar_format=TQDM_BAR_FORMAT,
            dynamic_ncols=True,
            leave=False,
            unit="pair",
        )

    def train_step(
        self,
        model: SuperPoint,
        loss_fn: SuperPointLoss,
        optimizer: torch.optim.Optimizer,
        batch: dict[str, Any],
    ) -> tuple[float, dict[str, torch.Tensor]]:
        """执行单批次前向传播、损失计算与反向传播。"""
        moved = batch_to_device(batch, self.device)
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
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        return float(out["loss"].detach()), out

    def log_step(self, pbar: Any, loss: float, out: dict[str, torch.Tensor], stage: str) -> None:
        """在进度条上展示当前阶段的核心指标。"""
        if "desc" in stage:
            pbar.set_postfix_str(f"loss={loss:.4f} ld={float(out.get('ld', 0.0)):.3f}")
        elif "det" in stage:
            l_dist = float(out.get("l_dist", 0.0).detach()) if torch.is_tensor(out.get("l_dist")) else float(out.get("l_dist", 0.0))
            l_ang = float(out.get("l_ang", 0.0).detach()) if torch.is_tensor(out.get("l_ang")) else float(out.get("l_ang", 0.0))
            if l_dist > 0 or l_ang > 0:
                pbar.set_postfix_str(f"loss={loss:.4f} dist={l_dist:.3f} ang={l_ang:.3f}")
            else:
                pbar.set_postfix_str(
                    f"loss={loss:.4f} det={float(out.get('l_det', 0.0).detach()):.3f} orph={float(out.get('l_orphan', 0.0).detach()):.3f}"
                )
        else:
            ld_val = float(out["ld"].detach()) if torch.is_tensor(out.get("ld")) else float(out.get("ld", 0.0))
            lorph_val = float(out["l_orphan"].detach()) if torch.is_tensor(out.get("l_orphan")) else float(out.get("l_orphan", 0.0))
            pbar.set_postfix_str(
                f"loss={loss:.4f} desc={ld_val:.3f} orph={lorph_val:.3f}"
            )

    def validate(
        self,
        model: SuperPoint,
        loss_fn: SuperPointLoss,
        val_loader: DataLoader,
        epoch: int,
        stage: str,
    ) -> dict[str, float]:
        """执行周期性验证。"""
        return run_validation_loop(
            model=model,
            criterion=loss_fn,
            loader=val_loader,
            device=self.device,
            max_steps=self.max_steps,
            vis_dir=self.vis_dir,
            epoch=epoch,
            train_stage=stage,
        )

    def save_checkpoint(
        self,
        model: SuperPoint,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_metrics: dict[str, float],
        is_best: bool,
    ) -> None:
        """保存最新的检查点与最佳模型权重。"""
        ckpt = {
            "epoch": epoch,
            "train_stage": self.train_stage,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()},
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, self.save_dir / "last_model.pth")
        if is_best:
            torch.save(ckpt, self.save_dir / "best_model.pth")


# =============================================================================
# 2. 评测管线助手类 (SuperPointEvalHelper)
# =============================================================================

class SuperPointEvalHelper:
    """SuperPoint 评测与诊断报告助手类。"""

    def __init__(self, args: argparse.Namespace | None = None, **kwargs: Any) -> None:
        if args is None:
            parser = build_eval_parser()
            args = parser.parse_args([])
        for k, v in kwargs.items():
            setattr(args, k, v)
        self.args = args

        self.device = torch.device(args.device)
        self.data_dir = Path(args.data_dir or "data/planar_letters").resolve()
        self.split = getattr(args, "split", "val")
        self.out_dir = Path(getattr(args, "out_dir", "outputs/superpoint_eval")).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_vis = int(getattr(args, "num_vis", 5))
        self.eval_stage = getattr(args, "eval_stage", "all").lower()
        self.checkpoint = getattr(args, "checkpoint", "checkpoints/SuperPoint/latest")
        self.ckpt_path = self._resolve_checkpoint(self.checkpoint)

        self.loss_kwargs = {
            k: getattr(args, k)
            for k in (
                "weight_desc", "positive_weight", "positive_margin", "negative_margin",
                "weight_det", "det_pos_weight", "weight_orphan", "weight_len",
                "length_exponent", "weight_premature", "probe_dist",
                "tau_df", "tau_ang", "min_length", "num_candidates", "num_line_samples",
            )
            if hasattr(args, k) and getattr(args, k) is not None
        }

        # 统计累加器
        self.total_loss = 0.0
        self.total_ld = 0.0
        self.total_l_det = 0.0
        self.total_l_orphan = 0.0
        self.total_l_len = 0.0
        self.total_l_prem = 0.0
        self.total_nn_acc = 0.0
        self.total_mma_1px = 0.0
        self.total_mma_2px = 0.0
        self.total_mma_3px = 0.0
        self.total_rep_15 = 0.0
        self.total_rep_30 = 0.0
        self.total_loc_err = 0.0
        self.total_corner_tp = 0
        self.total_corner_fp = 0
        self.total_corner_fn = 0
        self.total_lines_v0 = 0
        self.total_lines_v1 = 0
        self.total_line_len = 0.0
        self.total_line_df = 0.0
        self.max_line_len_overall = 0.0
        self.total_orphan_rate = 0.0
        self.steps = 0

    def _resolve_checkpoint(self, path_str: str) -> Path:
        p = Path(path_str).resolve()
        if p.is_dir() or not p.exists():
            for c in (p / "best_model.pth", p / "last_model.pth"):
                if c.is_file():
                    return c
        return p

    def setup(self) -> tuple[SuperPoint, SuperPointLoss, DataLoader]:
        """加载模型权重、准备损失函数与数据加载器。"""
        model = SuperPoint().to(self.device)
        if self.ckpt_path.is_file():
            ckpt_data = safe_load_checkpoint(self.ckpt_path, map_location=self.device)
            state = ckpt_data.get("model_state", ckpt_data.get("state_dict", ckpt_data))
            model.load_state_dict(state, strict=True)
            print(f"[✓] Loaded SuperPoint weights from: {self.ckpt_path}")
        else:
            print(f"[!] Warning: Checkpoint not found: {self.ckpt_path}, evaluating uninitialized model")

        criterion = SuperPointLoss(stage=self.eval_stage, **self.loss_kwargs).to(self.device)
        ds = PointMatchDataset(self.data_dir, split=self.split)
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_point_matches, num_workers=getattr(self.args, "num_workers", 0))
        return model, criterion, loader

    def progress_bar(self, loader: DataLoader) -> Any:
        return tqdm(loader, desc="Evaluating SuperPoint", bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")

    def forward_step(
        self,
        model: SuperPoint,
        loss_fn: SuperPointLoss,
        batch: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """执行单步推理与损失评估。"""
        moved = batch_to_device(batch, self.device)
        pred0 = model(moved["image0"])
        pred1 = model(moved["image1"])
        loss_out = loss_fn(
            pred0, pred1, moved["homography"],
            image0=moved["image0"], image1=moved["image1"],
            points0=moved.get("points0"), points1=moved.get("points1"),
            lines0=moved.get("lines0"), lines1=moved.get("lines1"),
        )
        return pred0, pred1, loss_out

    def accumulate_step(
        self,
        batch: dict[str, Any],
        pred0: dict[str, torch.Tensor],
        pred1: dict[str, torch.Tensor],
        loss_dict: dict[str, torch.Tensor],
        criterion: SuperPointLoss | None = None,
    ) -> None:
        """累加单步度量指标。"""
        batch = batch_to_device(batch, self.device)
        img0, img1 = batch["image0"], batch["image1"]
        h_mat = batch["homography"]
        pts0_gt, pts1_gt = batch.get("points0"), batch.get("points1")
        _, _, height, width = img0.shape

        self.total_loss += float(loss_dict["loss"])
        self.total_ld += float(loss_dict.get("ld", 0.0))
        self.total_l_det += float(loss_dict.get("l_det", 0.0))
        self.total_l_orphan += float(loss_dict.get("l_orphan", 0.0))
        self.total_l_len += float(loss_dict.get("l_len", 0.0))
        self.total_l_prem += float(loss_dict.get("l_prem", 0.0))

        if self.eval_stage in ("all", "descriptor") and pts0_gt is not None and pts1_gt is not None:
            mma_res = match_accuracy_mma(pred0["descriptor"], pred1["descriptor"], pts0_gt, pts1_gt, h_mat)
            self.total_nn_acc += mma_res["nn_acc"]
            self.total_mma_1px += mma_res["mma_1.0px"]
            self.total_mma_2px += mma_res["mma_2.0px"]
            self.total_mma_3px += mma_res["mma_3.0px"]

        if self.eval_stage in ("all", "detector") and criterion is not None:
            geom0 = extract_detected_lines(pred0["heatmap"], img0[:, 3:4], img0[:, 4:5], criterion)
            geom1 = extract_detected_lines(pred1["heatmap"], img1[:, 3:4], img1[:, 4:5], criterion)

            self.total_lines_v0 += geom0["num_lines"]
            self.total_lines_v1 += geom1["num_lines"]
            self.total_line_len += (geom0["mean_length"] + geom1["mean_length"]) / 2.0
            self.total_line_df += (geom0["mean_df"] + geom1["mean_df"]) / 2.0
            self.max_line_len_overall = max(self.max_line_len_overall, geom0["max_length"], geom1["max_length"])
            self.total_orphan_rate += (geom0["orphan_rate"] + geom1["orphan_rate"]) / 2.0

            rep15, _ = calculate_corner_repeatability(geom0["endpoints"], geom1["endpoints"], h_mat, height, width, 1.5)
            rep30, err30 = calculate_corner_repeatability(geom0["endpoints"], geom1["endpoints"], h_mat, height, width, 3.0)
            self.total_rep_15 += rep15
            self.total_rep_30 += rep30
            self.total_loc_err += err30

            if pts0_gt is not None and len(pts0_gt) > 0:
                gt_np = pts0_gt[0].detach().cpu().numpy() if pts0_gt.dim() == 3 else pts0_gt.detach().cpu().numpy()
                pred_pts = geom0["endpoints"]
                if len(pred_pts) > 0:
                    dist_to_gt = np.linalg.norm(pred_pts[:, None, :] - gt_np[None, :, :], axis=-1)
                    tp_pred = (dist_to_gt.min(axis=1) <= 3.0).sum()
                    fp_pred = len(pred_pts) - tp_pred
                    dist_to_pred = dist_to_gt.T
                    tp_gt = (dist_to_pred.min(axis=1) <= 3.0).sum()
                    fn_gt = len(gt_np) - tp_gt
                    self.total_corner_tp += int(tp_pred)
                    self.total_corner_fp += int(fp_pred)
                    self.total_corner_fn += int(fn_gt)

        self.steps += 1

    def compute_metrics(self) -> dict[str, Any]:
        """汇总计算最终评测指标。"""
        denom = max(self.steps, 1)
        mean_nn_acc = self.total_nn_acc / denom
        corner_prec = 100.0 * self.total_corner_tp / max(self.total_corner_tp + self.total_corner_fp, 1)
        corner_rec = 100.0 * self.total_corner_tp / max(self.total_corner_tp + self.total_corner_fn, 1)
        corner_f1 = 2 * corner_prec * corner_rec / max(corner_prec + corner_rec, 1e-4)

        metrics = {
            "loss": self.total_loss / denom,
            "loss_desc": self.total_ld / denom,
            "loss_det": self.total_l_det / denom,
            "loss_orphan": self.total_l_orphan / denom,
            "loss_len": self.total_l_len / denom,
            "loss_prem": self.total_l_prem / denom,
            "match_acc": mean_nn_acc,
            "mma_1px": self.total_mma_1px / denom,
            "mma_2px": self.total_mma_2px / denom,
            "mma_3px": self.total_mma_3px / denom,
            "corner_repeatability_1_5px": self.total_rep_15 / denom,
            "corner_repeatability_3_0px": self.total_rep_30 / denom,
            "corner_loc_error": self.total_loc_err / denom,
            "corner_precision": corner_prec,
            "corner_recall": corner_rec,
            "corner_f1": corner_f1,
            "valid_lines_per_view": (self.total_lines_v0 + self.total_lines_v1) / (2.0 * denom),
            "mean_line_length": self.total_line_len / denom,
            "max_line_length": self.max_line_len_overall,
            "line_df_alignment": self.total_line_df / denom,
            "orphan_rate": self.total_orphan_rate / denom,
            "total_pairs": self.steps,
            "checkpoint": str(self.ckpt_path),
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return metrics

    def render_visualizations(
        self,
        model: SuperPoint,
        criterion: SuperPointLoss,
        loader: DataLoader,
    ) -> list[str]:
        """渲染定性诊断图像 (检测、匹配流向图、线描述子 PCA 流形图)。"""
        if self.num_vis <= 0:
            return []
        print(f"[*] Rendering {self.num_vis} diagnostic visualization samples (stage: {self.eval_stage})...")
        vis_paths: list[str] = []
        vis_count = 0

        with torch.inference_mode():
            for batch in loader:
                if vis_count >= self.num_vis:
                    break
                moved = batch_to_device(batch, self.device)
                img0, img1 = moved["image0"], moved["image1"]
                h_mat = moved["homography"]
                pts0_gt, pts1_gt = moved.get("points0"), moved.get("points1")
                pair_id = batch["pair_id"]

                pred0 = model(img0)
                pred1 = model(img1)
                loss_out = criterion(
                    pred0, pred1, h_mat,
                    image0=img0, image1=img1, points0=pts0_gt, points1=pts1_gt,
                    lines0=moved.get("lines0"), lines1=moved.get("lines1"),
                )
                summary = {
                    "loss": float(loss_out["loss"]),
                    "ld": float(loss_out.get("ld", 0.0)),
                    "l_det": float(loss_out.get("l_det", 0.0)),
                    "l_orphan": float(loss_out.get("l_orphan", 0.0)),
                    "l_len": float(loss_out.get("l_len", 0.0)),
                    "l_prem": float(loss_out.get("l_prem", 0.0)),
                }

                if self.eval_stage in ("detector", "all"):
                    det_dest = self.out_dir / f"{pair_id}_detection.jpg"
                    save_detection_mask_lines_figure(
                        det_dest, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                        pred0, pred1, img0[:, 3:4], img1[:, 3:4], img0[:, 4:5], img1[:, 4:5],
                        criterion, pair_id=pair_id, metrics_summary=summary,
                        points0=pts0_gt, points1=pts1_gt,
                    )
                    vis_paths.append(str(det_dest))

                if self.eval_stage in ("descriptor", "all") and pts0_gt is not None and pts1_gt is not None:
                    desc_dest = self.out_dir / f"{pair_id}_matching.jpg"
                    save_descriptor_matching_figure(
                        desc_dest, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                        pred0["descriptor"], pred1["descriptor"], pts0_gt, pts1_gt, h_mat,
                        pair_id=pair_id, metrics_summary=summary,
                    )
                    vis_paths.append(str(desc_dest))

                if self.eval_stage in ("descriptor", "all"):
                    pca_dest = self.out_dir / f"{pair_id}_pca.jpg"
                    save_line_descriptor_pca_figure(
                        pca_dest, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                        pred0["descriptor"], pred1["descriptor"],
                        lines0=moved.get("lines0"), lines1=moved.get("lines1"),
                        pair_id=pair_id, metrics_summary=summary,
                    )
                    vis_paths.append(str(pca_dest))

                vis_count += 1
        return vis_paths

    def save_reports(self, metrics: dict[str, Any]) -> None:
        """保存评测 JSON 与 Markdown 诊断报告。"""
        eval_json_path = self.out_dir / "eval_results.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        eval_md_path = self.out_dir / "eval_report.md"
        with open(eval_md_path, "w", encoding="utf-8") as f:
            f.write("# SuperPoint 评测报告 (Evaluation Report)\n\n")
            f.write(f"- 检查点: `{metrics['checkpoint']}`\n")
            f.write(f"- 数据集: `{self.data_dir.name}` (Split: `{self.split}`, 共 {metrics['total_pairs']} 对样本)\n")
            f.write(f"- 评测阶段: `{self.eval_stage}` | 时间: `{metrics['evaluated_at']}`\n\n")
            f.write("### 核心指标概览\n\n")
            f.write("| 指标项 | 数值 | 说明 |\n| :--- | :---: | :--- |\n")
            f.write(f"| **总损失 (Total Loss)** | `{metrics['loss']:.4f}` | 综合加权总损失 |\n")
            f.write(f"| **最近邻匹配准确率 (NN Match Acc)** | **`{metrics['match_acc']*100:.2f}%`** | 描述子度量正确率 |\n")
            f.write(f"| **MMA @ 3.0px** | `{metrics['mma_3px']*100:.2f}%` | 3px 内匹配对应率 |\n")
            f.write(f"| **角点重复率 @ 3.0px** | `{metrics['corner_repeatability_3_0px']*100:.2f}%` | 跨视角端点几何重投影重复率 |\n")
            f.write(f"| **角点定位误差** | `{metrics['corner_loc_error']:.2f}px` | 匹配角点平均像素位移偏差 |\n")
            f.write(f"| **角点查准率 (Precision)** | `{metrics['corner_precision']:.2f}%` | 检测点命中 GT 的比例 |\n")
            f.write(f"| **角点查全率 (Recall)** | `{metrics['corner_recall']:.2f}%` | GT 被成功检出的比例 |\n")
            f.write(f"| **成线数量 (Lines/view)** | `{metrics['valid_lines_per_view']:.1f}` | 视角平均合法直线数量 |\n")
            f.write(f"| **孤立点率 (Orphan Rate)** | `{metrics['orphan_rate']*100:.2f}%` | 未连成任何合法线的孤立端点占比 |\n")

        print(f"[✓] Saved evaluation metrics to : {eval_json_path}")
        print(f"[✓] Saved evaluation report to  : {eval_md_path}")


# =============================================================================
# 3. 验证函数 (run_validation_loop / validate)
# =============================================================================

def run_validation_loop(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
    vis_dir: Path | None = None,
    epoch: int = 1,
    train_stage: str = "joint",
) -> dict[str, float]:
    """训练期周期性验证循环。"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_lines = 0.0
    total_orphan_rate = 0.0
    steps = 0
    val_total = len(loader) if max_steps is None else min(len(loader), max_steps)

    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(
        loader,
        total=val_total,
        desc=f"Val [{train_stage.capitalize()}]",
        bar_format=TQDM_BAR_FORMAT,
        dynamic_ncols=True,
        leave=False,
        unit="pair",
    )
    with torch.inference_mode():
        for batch in pbar:
            if max_steps is not None and steps >= max_steps:
                break
            moved = batch_to_device(batch, device)
            pred0 = model(moved["image0"])
            pred1 = model(moved["image1"])
            out = criterion(
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
            step_loss = float(out["loss"])
            total_loss += step_loss

            accuracy = 0.0
            if "descriptor" in pred0 and "descriptor" in pred1 and moved.get("points0") is not None and moved.get("points1") is not None:
                acc_val, _ = match_accuracy(
                    pred0["descriptor"], pred1["descriptor"],
                    moved["points0"], moved["points1"],
                )
                accuracy = acc_val
                total_acc += accuracy

            orphan_rate = 0.0
            num_lines = 0.0
            if "heatmap" in pred0 and moved["image0"].shape[1] >= 5:
                geom = extract_detected_lines(
                    pred0["heatmap"], moved["image0"][:, 3:4], moved["image0"][:, 4:5], criterion
                )
                orphan_rate = geom["orphan_rate"]
                num_lines = geom["num_lines"]
                total_lines += num_lines
                total_orphan_rate += orphan_rate

            steps += 1
            if steps == 1 and vis_dir is not None:
                vis_out = vis_dir / f"val_epoch_{epoch:03d}_{batch['pair_id']}.jpg"
                save_detection_mask_lines_figure(
                    vis_out, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                    pred0, pred1, moved["image0"][:, 3:4], moved["image1"][:, 3:4],
                    moved["image0"][:, 4:5], moved["image1"][:, 4:5],
                    criterion, pair_id=batch["pair_id"],
                )

    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "match_acc": total_acc / denom,
        "num_lines": total_lines / denom,
        "orphan_rate": total_orphan_rate / denom,
    }


# =============================================================================
# 4. 命令行参数解析器构造 (CLI Parser Builders)
# =============================================================================

def build_train_parser() -> argparse.ArgumentParser:
    """构建 SuperPoint 训练专用参数解析器。"""
    parser = argparse.ArgumentParser(description="SuperPoint Training Pipeline")
    parser.add_argument("--config", "-c", type=str, default="configs/superpoint.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default="data/planar_letters", help="Dataset directory")
    parser.add_argument(
        "--train-stage", type=str, default="stage5_joint",
        choices=["stage1_det_warmup", "stage2_det_geom", "stage3_desc_intra", "stage4_desc_cross", "stage5_joint", "detector", "descriptor", "joint"],
    )
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

    # 损失与几何超参数
    parser.add_argument("--weight-dist", type=float, default=1.0)
    parser.add_argument("--weight-ang", type=float, default=1.0)
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


def build_eval_parser() -> argparse.ArgumentParser:
    """构建 SuperPoint 评测专用参数解析器。"""
    parser = argparse.ArgumentParser(description="SuperPoint Evaluation & Benchmark")
    parser.add_argument("--config", "-c", type=str, default="configs/superpoint.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/SuperPoint/latest")
    parser.add_argument("--data-dir", type=str, default="data/planar_letters")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs/superpoint_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--eval-stage", type=str, default="all", choices=["all", "detector", "descriptor"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)

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
    return parser
