"""GlueStick 匹配模型全量评测与可视化评估工具 (GlueStick Evaluation & Visualizer).

核心功能:
1. 在验证集 / 测试集上评测 GlueStick 或 SemanticGlueStick 的端到端跨视角匹配性能;
2. 统计全面量化指标:
   - 关键点指标: Point Precision (精确率), Point Recall (召回率), Point F1;
   - 线段几何指标: Line Precision, Line Recall, Line F1;
3. 渲染输出高质量跨视角匹配诊断对比大图:
   - 并排展示模型预测匹配连线 vs 真实投影真值匹配连线 (Ground Truth).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
gluestick_cands = [ROOT / "third_party" / "GlueStick", ROOT / "GlueStick"]
GLUESTICK_DIR = next((p for p in gluestick_cands if p.is_dir()), gluestick_cands[0])
if str(GLUESTICK_DIR) not in sys.path:
    sys.path.insert(0, str(GLUESTICK_DIR))

from datasets.semantic_gluestick_dataset import SemanticGlueStickDataset, semantic_gluestick_collate_fn
from models.gluestick_wrapper import GlueStickLoss, SemanticGlueStick, compute_matching_metrics


def draw_matches_side_by_side(
    img0: np.ndarray,
    img1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    matches0: np.ndarray,
    title: str = "Matches",
    max_draw: int = 50,
) -> np.ndarray:
    """并排绘制跨视角特征点匹配连线。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1

    valid_pairs = []
    for i, j in enumerate(matches0):
        if j >= 0 and j < len(pts1):
            valid_pairs.append((i, j))

    if len(valid_pairs) > max_draw:
        step = len(valid_pairs) // max_draw
        valid_pairs = valid_pairs[::step][:max_draw]

    cmap = cv2.applyColorMap(
        np.linspace(0, 255, max(len(valid_pairs), 1), dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET,
    )

    for k, (i, j) in enumerate(valid_pairs):
        p0 = (int(round(pts0[i, 0])), int(round(pts0[i, 1])))
        p1 = (int(round(pts1[j, 0])) + w0, int(round(pts1[j, 1])))
        col = tuple(int(c) for c in cmap[k, 0])

        cv2.circle(canvas, p0, 3, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, col, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p0, p1, col, 1, lineType=cv2.LINE_AA)

    # 标签
    cv2.putText(canvas, f"{title} ({len(valid_pairs)} shown / {int((matches0 >= 0).sum())} total)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_line_matches_side_by_side(
    img0: np.ndarray,
    img1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    line_matches0: np.ndarray,
    title: str = "Line Matches",
    max_draw: int = 40,
) -> np.ndarray:
    """并排绘制跨视角线段匹配连线。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1

    valid_pairs = []
    for i, j in enumerate(line_matches0):
        if j >= 0 and j < len(lines1):
            valid_pairs.append((i, j))

    if len(valid_pairs) > max_draw:
        step = len(valid_pairs) // max_draw
        valid_pairs = valid_pairs[::step][:max_draw]

    cmap = cv2.applyColorMap(
        np.linspace(0, 255, max(len(valid_pairs), 1), dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    )

    for k, (i, j) in enumerate(valid_pairs):
        l0 = lines0[i]
        l1 = lines1[j]
        col = tuple(int(c) for c in cmap[k, 0])

        p0_start = (int(round(l0[0, 0])), int(round(l0[0, 1])))
        p0_end = (int(round(l0[1, 0])), int(round(l0[1, 1])))
        p1_start = (int(round(l1[0, 0])) + w0, int(round(l1[0, 1])))
        p1_end = (int(round(l1[1, 0])) + w0, int(round(l1[1, 1])))

        cv2.line(canvas, p0_start, p0_end, col, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, p1_start, p1_end, col, 2, lineType=cv2.LINE_AA)

        mid0 = ((p0_start[0] + p0_end[0]) // 2, (p0_start[1] + p0_end[1]) // 2)
        mid1 = ((p1_start[0] + p1_end[0]) // 2, (p1_start[1] + p1_end[1]) // 2)
        cv2.line(canvas, mid0, mid1, col, 1, lineType=cv2.LINE_AA)

    cv2.putText(canvas, f"{title} ({len(valid_pairs)} shown / {int((line_matches0 >= 0).sum())} total)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="GlueStick 评测与可视化脚本")
    parser.add_argument("--checkpoint", default="checkpoints/gluestick_train/best_model.pth", help="模型权重文件路径")
    default_base_weights = ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
    if not default_base_weights.is_file():
        default_base_weights = ROOT / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
    parser.add_argument("--base-weights", default=str(default_base_weights), help="官方基础权重路径")
    parser.add_argument("--data-dir", default="data/scannet_semantic_gluestick", help="评测数据集路径")
    parser.add_argument("--split", default="val", help="评测集划分 (val 或 all)")
    parser.add_argument("--use-semantics", action="store_true", help="是否加载语义增强架构")
    parser.add_argument("--out-dir", default="outputs/gluestick_eval", help="评测结果与可视化保存目录")
    parser.add_argument("--max-visualize", type=int, default=5, help="最多生成多少个样本的可视化诊断对比图")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("  🔍 GlueStick 跨视角几何与语义匹配模型评测")
    print("=" * 70)

    # 1. 加载模型
    ckpt_path = Path(args.checkpoint).resolve()
    weights_to_use = str(ckpt_path) if ckpt_path.is_file() else args.base_weights

    model = SemanticGlueStick(
        weights=weights_to_use if not args.use_semantics else None,
        use_semantics=args.use_semantics,
    ).to(device)

    if ckpt_path.is_file():
        ckpt_data = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        model.load_state_dict(state_dict, strict=False)
        print(f"[✓] 成功载入指定权重检查点: {ckpt_path.name}")
    else:
        print(f"[!] 检查点 {ckpt_path} 不存在，已使用官方基线权重评测: {args.base_weights}")

    model.eval()

    # 2. 加载数据集
    val_ds = SemanticGlueStickDataset(args.data_dir, split=args.split)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=semantic_gluestick_collate_fn)
    print(f"[✓] 成功载入评测数据集 ({args.split}): 共 {len(val_ds)} 个样本")

    loss_fn = GlueStickLoss()

    all_metrics = []
    total_loss = 0.0

    print("[*] 正在执行全量批次前向推演与指标统计...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="评测进度")):
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(device)
                else:
                    batch_gpu[k] = v

            pred = model(batch_gpu)
            loss, stats = loss_fn(pred, batch_gpu)
            metrics = compute_matching_metrics(pred, batch_gpu)

            total_loss += stats["total_loss"]
            all_metrics.append(metrics)

            # 可视化输出
            if idx < args.max_visualize:
                sample_name = batch["name"][0]
                img0_u8 = (batch["image0"][0, 0].cpu().numpy() * 255.0).astype(np.uint8)
                img1_u8 = (batch["image1"][0, 0].cpu().numpy() * 255.0).astype(np.uint8)
                kpts0 = batch["keypoints0"][0].cpu().numpy()
                kpts1 = batch["keypoints1"][0].cpu().numpy()
                lines0 = batch["lines0"][0].cpu().numpy()
                lines1 = batch["lines1"][0].cpu().numpy()

                pred_m0 = pred["matches0"][0].cpu().numpy()
                gt_m0 = batch["gt_matches0"][0].cpu().numpy()
                pred_lm0 = pred["line_matches0"][0].cpu().numpy()
                gt_lm0 = batch["gt_line_matches0"][0].cpu().numpy()

                # 4 排诊断图
                r1 = draw_matches_side_by_side(img0_u8, img1_u8, kpts0, kpts1, pred_m0, title="Predicted Point Matches")
                r2 = draw_matches_side_by_side(img0_u8, img1_u8, kpts0, kpts1, gt_m0, title="Ground Truth Point Matches")
                r3 = draw_line_matches_side_by_side(img0_u8, img1_u8, lines0, lines1, pred_lm0, title="Predicted Line Matches")
                r4 = draw_line_matches_side_by_side(img0_u8, img1_u8, lines0, lines1, gt_lm0, title="Ground Truth Line Matches")

                vis_composite = np.concatenate([r1, r2, r3, r4], axis=0)
                save_img_path = out_dir / f"{sample_name}_eval_matches.jpg"
                cv2.imwrite(str(save_img_path), vis_composite)

    n = len(all_metrics)
    summary = {
        "dataset_size": n,
        "mean_loss": float(total_loss / max(n, 1)),
        "point_precision": float(np.mean([m["point_precision"] for m in all_metrics])),
        "point_recall": float(np.mean([m["point_recall"] for m in all_metrics])),
        "point_f1": float(np.mean([m["point_f1"] for m in all_metrics])),
        "line_precision": float(np.mean([m["line_precision"] for m in all_metrics])),
        "line_recall": float(np.mean([m["line_recall"] for m in all_metrics])),
        "line_f1": float(np.mean([m["line_f1"] for m in all_metrics])),
    }

    print("\n" + "=" * 70)
    print("  📊 全量定量评测结果报告 (Evaluation Summary Report)")
    print("=" * 70)
    print(f"  [Loss] 综合平均损失: {summary['mean_loss']:.4f}")
    print(f"  [Points] 关键点匹配: Precision={summary['point_precision']*100:.2f}%, Recall={summary['point_recall']*100:.2f}%, F1={summary['point_f1']:.4f}")
    print(f"  [Lines]  几何线匹配: Precision={summary['line_precision']*100:.2f}%, Recall={summary['line_recall']*100:.2f}%, F1={summary['line_f1']:.4f}")
    print("=" * 70)

    summary_file = out_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[✓] 详细评测指标已导出至: {summary_file}")
    print(f"[✓] 可视化对比图已保存至: {out_dir}")


if __name__ == "__main__":
    main()
