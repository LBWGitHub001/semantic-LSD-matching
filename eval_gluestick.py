"""GlueStick 几何与语义匹配模型最简评测脚本 (PyCharm 一键直接运行版).

核心特点:
1. 【零命令行参数依赖】：所有参数在下方【评测配置】区域集中直观配置，PyCharm 中右键点击 "Run" 即可直接执行；
2. 【工作目录自适应】：自动以脚本所在根目录定位，不受 PyCharm 运行路径设置影响；
3. 【模型智能感知】：自动探测并载入 best_model.pth，自动识别训练配置与语义增强开关；
4. 【线框底图渲染】：第 3、4 排不仅展示成功匹配的彩色线，更完整绘制所有候选线段作为几何底图；
5. 【全面量化评估】：自动输出特征点与几何直线 Precision, Recall, F1，导出 eval_summary.json 与诊断大图。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# ==============================================================================
# ⚙️ 评测配置区 (PyCharm 中直接修改下方参数即可，无需在终端输入命令)
# ==============================================================================
# 模型权重路径 (自动检查并加载，若文件不存在则自动回退至官方预训练基线)
CHECKPOINT = "checkpoints/gluestick_train/best_model.pth"

# 评测数据集目录
DATA_DIR = "data/scannet_semantic_gluestick"

# 评测子集划分: "val" (验证集 225 对样本，推荐) 或 "all" (全量 1500 对)
SPLIT = "val"

# 是否启用 SAM 语义增强 (设为 None 时将自动从检查点中读取训练时的设置)
USE_SEMANTICS = True

# 匹配置信度阈值 (默认 0.2，调低至 0.1 可显示更多弱纹理候选匹配)
FILTER_THRESHOLD = 0.2

# 诊断大图生成数量 (0 表示不保存图片，仅统计指标)
MAX_VISUALIZE = 5

# 评测指标与诊断大图输出目录
OUTPUT_DIR = "outputs/gluestick_eval"

# 计算设备 ("cuda" 或 "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================================================


# 确保能够正确引入第三方库与项目模块
ROOT = Path(__file__).resolve().parent
for cand in [ROOT / "third_party" / "GlueStick", ROOT / "GlueStick"]:
    if cand.is_dir() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

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

    # 1. 先用深灰色微点绘制所有检测到的特征点底图
    for pt in pts0:
        cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 1, (70, 70, 70), -1)
    for pt in pts1:
        cv2.circle(canvas, (int(round(pt[0])) + w0, int(round(pt[1]))), 1, (70, 70, 70), -1)

    # 2. 筛选匹配点对
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

    cv2.putText(
        canvas,
        f"{title} ({len(valid_pairs)} shown / {int((matches0 >= 0).sum())} total)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
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
    """并排绘制跨视角线段匹配连线 (含灰色几何线框底图 + 彩色匹配线段与中点连线)。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1

    # 1. 先用深灰色细线绘制两图所有检测到的候选线段 (展现线框几何底色)
    for l in lines0:
        p_s = (int(round(l[0, 0])), int(round(l[0, 1])))
        p_e = (int(round(l[1, 0])), int(round(l[1, 1])))
        cv2.line(canvas, p_s, p_e, (70, 70, 70), 1, lineType=cv2.LINE_AA)
    for l in lines1:
        p_s = (int(round(l[0, 0])) + w0, int(round(l[0, 1])))
        p_e = (int(round(l[1, 0])) + w0, int(round(l[1, 1])))
        cv2.line(canvas, p_s, p_e, (70, 70, 70), 1, lineType=cv2.LINE_AA)

    # 2. 筛选成功匹配的跨视角线段对
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

        # 加粗绘制匹配的两条线段
        cv2.line(canvas, p0_start, p0_end, col, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, p1_start, p1_end, col, 2, lineType=cv2.LINE_AA)

        # 绘制两条匹配线中点之间的连线
        mid0 = ((p0_start[0] + p0_end[0]) // 2, (p0_start[1] + p0_end[1]) // 2)
        mid1 = ((p1_start[0] + p1_end[0]) // 2, (p1_start[1] + p1_end[1]) // 2)
        cv2.line(canvas, mid0, mid1, col, 1, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{title} ({len(valid_pairs)} shown / {int((line_matches0 >= 0).sum())} total)",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def main():
    # 1. 路径自动解析与创建
    ckpt_path = (ROOT / CHECKPOINT).resolve() if not Path(CHECKPOINT).is_absolute() else Path(CHECKPOINT)
    data_dir = (ROOT / DATA_DIR).resolve() if not Path(DATA_DIR).is_absolute() else Path(DATA_DIR)
    out_dir = (ROOT / OUTPUT_DIR).resolve() if not Path(OUTPUT_DIR).is_absolute() else Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(DEVICE)

    print("=" * 75)
    print("  🔍 GlueStick 跨视角几何与语义匹配模型评测 (PyCharm 直接运行模式)")
    print("=" * 75)
    print(f"[*] 运行设备: {device}")
    print(f"[*] 评测数据: {data_dir} (子集: {SPLIT})")
    print(f"[*] 输出目录: {out_dir}")

    # 官方基线权重路径定位
    cand_base_weights = [
        ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar",
        ROOT / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar",
        ROOT / "weights" / "gluestick" / "checkpoint_GlueStick_MD.tar",
    ]
    base_weights_path = next((str(p.resolve()) for p in cand_base_weights if p.is_file()), None)

    # 2. 检查点读取与模型架构自适应
    use_semantics = USE_SEMANTICS
    ckpt_data = None

    if ckpt_path.is_file():
        print(f"[*] 检测到指定模型检查点: {ckpt_path.name}")
        ckpt_data = torch.load(ckpt_path, map_location=device)
        # 若用户设置为 None，则自动从检查点元数据中恢复 use_semantics
        if use_semantics is None and isinstance(ckpt_data, dict) and "conf" in ckpt_data:
            saved_conf = ckpt_data["conf"]
            if isinstance(saved_conf, dict):
                use_semantics = saved_conf.get("use_semantics", False)
                print(f"[*] 自动继承检查点语义配置: use_semantics={use_semantics}")
    else:
        print(f"[!] 警告: 未找到检查点 {ckpt_path}，将自动使用官方底座预训练权重评测！")
        if use_semantics is None:
            use_semantics = False

    if use_semantics is None:
        use_semantics = True

    # 3. 初始化模型
    model = SemanticGlueStick(
        weights=base_weights_path,
        use_semantics=use_semantics,
        filter_threshold=FILTER_THRESHOLD,
    ).to(device)

    # 4. 加载训练好的权重
    if ckpt_data is not None:
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[✓] 成功载入模型参数: {ckpt_path.name}")
        if missing:
            print(f"    - 未匹配的层数: {len(missing)}")
        if unexpected:
            print(f"    - 忽略的多余层数: {len(unexpected)}")

    model.eval()

    # 5. 加载数据集 (单线程无后台子进程，便于调试)
    val_ds = SemanticGlueStickDataset(data_dir, split=SPLIT)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 单线程加载
        collate_fn=semantic_gluestick_collate_fn,
    )
    total_samples = len(val_loader)
    print(f"[✓] 评测数据集装载完毕: 共 {total_samples} 个场景对 (单线程加载, 无 tqdm 进度条)")
    print("-" * 75)

    loss_fn = GlueStickLoss()
    all_metrics = []
    total_loss = 0.0

    print("[*] 正在执行批次前向推演与指标统计...")
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
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

            # 纯文本单行打印进度 (前 MAX_VISUALIZE 个样本以及每 20 个样本汇报一次)
            if idx < MAX_VISUALIZE or (idx + 1) % 20 == 0 or (idx + 1) == total_samples:
                print(
                    f"  -> [{idx + 1:3d}/{total_samples:3d}] 样本: {batch['name'][0]:<20} | "
                    f"Loss={stats['total_loss']:.3f} | Pt-F1={metrics['point_f1']:.3f}, Ln-F1={metrics['line_f1']:.3f}"
                )

            # 可视化输出前 N 对样本
            if idx < MAX_VISUALIZE:
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

                # 纵向拼接 4 排高清对比诊断大图
                r1 = draw_matches_side_by_side(img0_u8, img1_u8, kpts0, kpts1, pred_m0, title="Row 1: Predicted Point Matches")
                r2 = draw_matches_side_by_side(img0_u8, img1_u8, kpts0, kpts1, gt_m0, title="Row 2: Ground Truth Point Matches")
                r3 = draw_line_matches_side_by_side(img0_u8, img1_u8, lines0, lines1, pred_lm0, title="Row 3: Predicted Line Matches")
                r4 = draw_line_matches_side_by_side(img0_u8, img1_u8, lines0, lines1, gt_lm0, title="Row 4: Ground Truth Line Matches")

                vis_composite = np.concatenate([r1, r2, r3, r4], axis=0)
                save_img_path = out_dir / f"{sample_name}_eval_matches.jpg"
                cv2.imwrite(str(save_img_path), vis_composite)

    # 6. 计算全量平均指标
    n = len(all_metrics)
    summary = {
        "checkpoint": str(ckpt_path),
        "dataset_size": n,
        "mean_loss": float(total_loss / max(n, 1)),
        "point_precision": float(np.mean([m["point_precision"] for m in all_metrics])),
        "point_recall": float(np.mean([m["point_recall"] for m in all_metrics])),
        "point_f1": float(np.mean([m["point_f1"] for m in all_metrics])),
        "line_precision": float(np.mean([m["line_precision"] for m in all_metrics])),
        "line_recall": float(np.mean([m["line_recall"] for m in all_metrics])),
        "line_f1": float(np.mean([m["line_f1"] for m in all_metrics])),
    }

    print("\n" + "=" * 75)
    print("  📊 全量定量评测结果报告 (Evaluation Summary Report)")
    print("=" * 75)
    print(f"  [Loss] 综合平均损失: {summary['mean_loss']:.4f}")
    print(f"  [Points] 关键点匹配: Precision={summary['point_precision']*100:.2f}%, Recall={summary['point_recall']*100:.2f}%, F1={summary['point_f1']:.4f}")
    print(f"  [Lines]  几何线匹配: Precision={summary['line_precision']*100:.2f}%, Recall={summary['line_recall']*100:.2f}%, F1={summary['line_f1']:.4f}")
    print("=" * 75)

    summary_file = out_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[✓] 详细评测指标已导出至: {summary_file}")
    print(f"[✓] 可视化对比图已保存至: {out_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
