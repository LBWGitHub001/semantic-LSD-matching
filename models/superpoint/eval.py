"""SuperPoint Evaluation, Validation, and Benchmark Suite."""

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
from datasets.point_match_dataset import PointMatchDataset, collate_point_matches
from models.superpoint.model import SuperPoint
from models.superpoint.loss import SuperPointLoss
from training_utils import TQDM_BAR_FORMAT, safe_load_checkpoint
from visualization.point_matches import (
    calculate_corner_repeatability,
    extract_detected_lines,
    match_accuracy,
    match_accuracy_mma,
    save_descriptor_matching_figure,
    save_detection_mask_lines_figure,
    save_line_descriptor_pca_figure,
    save_superpoint_diagnostic_sheet,
)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int | None = None,
    vis_dir: Path | None = None,
    epoch: int = 1,
    train_stage: str = "joint",
) -> dict[str, float]:
    """In-training validation step for SuperPoint with periodic qualitative image saves."""
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
            moved = _batch_to_device(batch, device)
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
            if "descriptor" in pred0 and moved.get("points0") is not None and moved.get("points1") is not None:
                accuracy, _ = match_accuracy(pred0["descriptor"], pred1["descriptor"], moved["points0"], moved["points1"])
                total_acc += accuracy

            pair_lines = 0.0
            pair_orphan = 0.0
            if train_stage in ("detector", "joint") and "heatmap" in pred0 and moved.get("image0") is not None:
                geom0 = extract_detected_lines(pred0["heatmap"], moved["image0"][:, 3:4], moved["image0"][:, 4:5], criterion)
                pair_lines = float(geom0["num_lines"])
                pair_orphan = float(geom0["orphan_rate"])
                total_lines += pair_lines
                total_orphan_rate += pair_orphan

            steps += 1
            if train_stage == "detector":
                pbar.set_postfix_str(f"det={total_loss / steps:.4f} lines={total_lines / steps:.1f}")
            elif train_stage == "descriptor":
                pbar.set_postfix_str(f"desc={total_loss / steps:.4f} match={100 * total_acc / steps:.1f}%")
            else:
                pbar.set_postfix_str(f"loss={total_loss / steps:.4f} match={100 * total_acc / steps:.1f}% lines={total_lines / steps:.1f}")

            # Qualitative visual previews during training (first 4 pairs)
            if vis_dir is not None and steps <= 4:
                pair_id = batch["pair_id"]
                if train_stage in ("detector", "joint"):
                    save_detection_mask_lines_figure(
                        vis_dir / f"epoch_{epoch:02d}_{pair_id}_detection.jpg",
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0,
                        pred1,
                        moved["image0"][:, 3:4],
                        moved["image1"][:, 3:4],
                        moved["image0"][:, 4:5],
                        moved["image1"][:, 4:5],
                        criterion,
                        pair_id=pair_id,
                        metrics_summary={"loss": step_loss},
                        points0=moved.get("points0"),
                        points1=moved.get("points1"),
                    )

                if train_stage in ("descriptor", "joint") and moved.get("points0") is not None and moved.get("points1") is not None:
                    save_descriptor_matching_figure(
                        vis_dir / f"epoch_{epoch:02d}_{pair_id}_matching.jpg",
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0["descriptor"],
                        pred1["descriptor"],
                        moved["points0"],
                        moved["points1"],
                        moved["homography"],
                        pair_id=pair_id,
                        metrics_summary={"ld": float(out.get("ld", step_loss))},
                    )
                    save_line_descriptor_pca_figure(
                        vis_dir / f"epoch_{epoch:02d}_{pair_id}_pca.jpg",
                        batch["bgr0"].numpy(),
                        batch["bgr1"].numpy(),
                        pred0["descriptor"],
                        pred1["descriptor"],
                        lines0=moved.get("lines0"),
                        lines1=moved.get("lines1"),
                        pair_id=pair_id,
                        metrics_summary={"loss": step_loss},
                    )

    denom = max(steps, 1)
    return {
        "loss": total_loss / denom,
        "match_acc": total_acc / denom,
        "num_lines": total_lines / denom,
        "orphan_rate": total_orphan_rate / denom,
        "total_pairs": steps,
    }


def evaluate(
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
    """Full offline benchmark evaluation for SuperPoint with diagnostic report generation."""
    dev = torch.device(device)
    p_data = Path(data_dir).resolve()
    p_out = Path(out_dir).resolve()
    p_out.mkdir(parents=True, exist_ok=True)
    eval_stage = eval_stage.lower()

    if checkpoint is None:
        checkpoint = "checkpoints/SuperPoint/latest"
    ckpt_path = Path(checkpoint).resolve()
    if ckpt_path.is_dir() or not ckpt_path.exists():
        for candidate in (ckpt_path / "best_model.pth", ckpt_path / "last_model.pth"):
            if candidate.is_file():
                ckpt_path = candidate
                break

    model = SuperPoint().to(dev)
    if ckpt_path.is_file():
        ckpt_data = safe_load_checkpoint(ckpt_path, map_location=dev)
        state_dict = ckpt_data.get("model_state", ckpt_data.get("state_dict", ckpt_data))
        model.load_state_dict(state_dict, strict=True)
        print(f"[✓] Loaded SuperPoint weights from: {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint file not found: {ckpt_path}, evaluating uninitialized model")

    model.eval()

    loss_kwargs = {
        k: v for k, v in kwargs.items()
        if k in (
            "weight_desc", "positive_weight", "positive_margin", "negative_margin",
            "weight_det", "det_pos_weight", "weight_orphan", "weight_len",
            "length_exponent", "weight_premature", "probe_dist",
            "tau_df", "tau_ang", "min_length", "num_candidates", "num_line_samples",
        ) and v is not None
    }
    criterion = SuperPointLoss(stage=eval_stage, **loss_kwargs).to(dev)

    val_ds = PointMatchDataset(p_data, split=split)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_point_matches, num_workers=num_workers)

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

    pbar = tqdm(val_loader, desc="Evaluating SuperPoint", bar_format=TQDM_BAR_FORMAT, dynamic_ncols=True, leave=False, unit="pair")
    with torch.inference_mode():
        for batch in pbar:
            moved = _batch_to_device(batch, dev)
            img0, img1 = moved["image0"], moved["image1"]
            h_mat = moved["homography"]
            pts0_gt, pts1_gt = moved.get("points0"), moved.get("points1")
            _, _, height, width = img0.shape

            pred0 = model(img0)
            pred1 = model(img1)

            out = criterion(
                pred0,
                pred1,
                h_mat,
                image0=img0,
                image1=img1,
                points0=pts0_gt,
                points1=pts1_gt,
                lines0=moved.get("lines0"),
                lines1=moved.get("lines1"),
            )
            step_loss = float(out["loss"])
            total_loss += step_loss
            total_ld += float(out.get("ld", 0.0))
            total_l_det += float(out.get("l_det", 0.0))
            total_l_orphan += float(out.get("l_orphan", 0.0))
            total_l_len += float(out.get("l_len", 0.0))
            total_l_prem += float(out.get("l_prem", 0.0))

            if eval_stage in ("all", "descriptor") and pts0_gt is not None and pts1_gt is not None:
                mma_res = match_accuracy_mma(pred0["descriptor"], pred1["descriptor"], pts0_gt, pts1_gt, h_mat)
                total_nn_acc += mma_res["nn_acc"]
                total_mma_1px += mma_res["mma_1.0px"]
                total_mma_2px += mma_res["mma_2.0px"]
                total_mma_3px += mma_res["mma_3.0px"]

            if eval_stage in ("all", "detector"):
                geom0 = extract_detected_lines(pred0["heatmap"], img0[:, 3:4], img0[:, 4:5], criterion)
                geom1 = extract_detected_lines(pred1["heatmap"], img1[:, 3:4], img1[:, 4:5], criterion)

                total_lines_v0 += geom0["num_lines"]
                total_lines_v1 += geom1["num_lines"]
                total_line_len += (geom0["mean_length"] + geom1["mean_length"]) / 2.0
                total_line_df += (geom0["mean_df"] + geom1["mean_df"]) / 2.0
                max_line_len_overall = max(max_line_len_overall, geom0["max_length"], geom1["max_length"])
                total_orphan_rate += (geom0["orphan_rate"] + geom1["orphan_rate"]) / 2.0

                rep15, _ = calculate_corner_repeatability(geom0["endpoints"], geom1["endpoints"], h_mat, height, width, 1.5)
                rep30, err30 = calculate_corner_repeatability(geom0["endpoints"], geom1["endpoints"], h_mat, height, width, 3.0)
                total_rep_15 += rep15
                total_rep_30 += rep30
                total_loc_err += err30

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
        "checkpoint": str(ckpt_path),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Render qualitative visualizations
    vis_paths: list[str] = []
    if num_vis > 0:
        print(f"[*] Rendering {num_vis} diagnostic visualization samples (stage: {eval_stage})...")
        vis_count = 0
        with torch.inference_mode():
            for batch in val_loader:
                if vis_count >= num_vis:
                    break
                moved = _batch_to_device(batch, dev)
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
                metrics_summary = {
                    "loss": float(loss_out["loss"]),
                    "ld": float(loss_out.get("ld", 0.0)),
                    "l_det": float(loss_out.get("l_det", 0.0)),
                    "l_orphan": float(loss_out.get("l_orphan", 0.0)),
                    "l_len": float(loss_out.get("l_len", 0.0)),
                    "l_prem": float(loss_out.get("l_prem", 0.0)),
                }

                if eval_stage in ("detector", "all"):
                    det_dest = p_out / f"{pair_id}_detection.jpg"
                    save_detection_mask_lines_figure(
                        det_dest, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                        pred0, pred1, img0[:, 3:4], img1[:, 3:4], img0[:, 4:5], img1[:, 4:5],
                        criterion, pair_id=pair_id, metrics_summary=metrics_summary,
                        points0=pts0_gt, points1=pts1_gt,
                    )
                    vis_paths.append(str(det_dest))

                if eval_stage in ("descriptor", "all") and pts0_gt is not None and pts1_gt is not None:
                    desc_dest = p_out / f"{pair_id}_matching.jpg"
                    save_descriptor_matching_figure(
                        desc_dest, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                        pred0["descriptor"], pred1["descriptor"], pts0_gt, pts1_gt, h_mat,
                        pair_id=pair_id, metrics_summary=metrics_summary,
                    )
                    vis_paths.append(str(desc_dest))

                if eval_stage in ("descriptor", "all"):
                    pca_dest = p_out / f"{pair_id}_pca.jpg"
                    save_line_descriptor_pca_figure(
                        pca_dest, batch["bgr0"].numpy(), batch["bgr1"].numpy(),
                        pred0["descriptor"], pred1["descriptor"],
                        lines0=moved.get("lines0"), lines1=moved.get("lines1"),
                        pair_id=pair_id, metrics_summary=metrics_summary,
                    )
                    vis_paths.append(str(pca_dest))

                vis_count += 1

    metrics["visualizations"] = vis_paths

    eval_json_path = p_out / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    eval_md_path = p_out / "eval_report.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# SuperPoint 评测报告 (Evaluation Report)\n\n")
        f.write(f"- 检查点: `{ckpt_path}`\n")
        f.write(f"- 数据集: `{p_data.name}` (Split: `{split}`, 共 {metrics['total_pairs']} 对样本)\n")
        f.write(f"- 评测阶段: `{eval_stage}` | 时间: `{metrics['evaluated_at']}`\n\n")
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
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SuperPoint Evaluation & Benchmark Entrypoint")
    parser.add_argument("--config", "-c", type=str, default="configs/superpoint.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file or directory")
    parser.add_argument("--data-dir", type=str, default="data/planar_letters", help="Dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="outputs/superpoint_eval")
    parser.add_argument("--num-vis", type=int, default=5)
    parser.add_argument("--eval-stage", type=str, default="all", choices=["all", "detector", "descriptor"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)

    # SuperPoint geometric & loss thresholds
    parser.add_argument("--tau-df", type=float, default=None)
    parser.add_argument("--tau-ang", type=float, default=None)
    parser.add_argument("--min-length", type=float, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument("--num-line-samples", type=int, default=None)
    parser.add_argument("--weight-desc", type=float, default=None)
    parser.add_argument("--positive-weight", type=float, default=None)
    parser.add_argument("--positive-margin", type=float, default=None)
    parser.add_argument("--negative-margin", type=float, default=None)
    parser.add_argument("--weight-det", type=float, default=None)
    parser.add_argument("--det-pos-weight", type=float, default=None)
    parser.add_argument("--weight-orphan", type=float, default=None)
    parser.add_argument("--weight-len", type=float, default=None)
    parser.add_argument("--length-exponent", type=float, default=None)
    parser.add_argument("--weight-premature", type=float, default=None)
    parser.add_argument("--probe-dist", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    cfg_path = pre_parse_config_path(argv) or Path("configs/superpoint.yaml")
    if cfg_path.is_file():
        cfg = load_config(cfg_path)
        apply_config_to_parser(parser, cfg, section="eval")

    args = parser.parse_args(argv)
    standard_keys = {
        "config", "checkpoint", "data_dir", "split", "out_dir",
        "num_vis", "eval_stage", "device", "num_workers",
    }
    extra_kwargs = {
        k: v for k, v in vars(args).items()
        if k not in standard_keys and v is not None
    }
    return evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        out_dir=args.out_dir,
        num_vis=args.num_vis,
        eval_stage=args.eval_stage,
        device=args.device,
        num_workers=args.num_workers,
        **extra_kwargs,
    )


if __name__ == "__main__":
    main()
