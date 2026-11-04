"""GlueStick 几何与语义匹配模型最简评测脚本 (PyCharm 一键直接运行版).

核心特点:
1. 【零命令行参数依赖】：所有参数在下方【评测配置】区域集中直观配置，PyCharm 中右键点击 "Run" 即可直接执行；
2. 【工作目录自适应】：自动以脚本所在根目录定位，不受 PyCharm 运行路径设置影响；
3. 【模型智能感知】：自动探测并载入 best_model.pth，自动识别训练配置与语义增强开关；
4. 【线框底图渲染】：第 3、4 排不仅展示成功匹配的彩色线，更完整绘制所有候选线段作为几何底图；
5. 【全面量化评估】：自动输出特征点与几何直线 Precision, Recall, F1，导出 eval_summary.json 与诊断大图。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# ==============================================================================
# ⚙️ 评测配置区 (PyCharm 中直接修改下方参数即可，无需在终端输入命令；亦支持命令行传参)
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

# 评测指标与诊断大图输出根目录 (每次评测将在其下新建独立的带时间戳运行目录)
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数 (若未提供参数则自动使用上方配置区的默认值，完美支持 PyCharm 右键一键运行)."""
    parser = argparse.ArgumentParser(description="GlueStick 跨视角几何与语义匹配模型评测")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="模型权重路径")
    parser.add_argument("--data-dir", default=DATA_DIR, help="评测数据集目录")
    parser.add_argument("--split", default=SPLIT, choices=["val", "train", "all"], help="评测子集划分")
    parser.add_argument("--use-semantics", dest="use_semantics", action="store_true", default=None, help="显式启用 SAM 语义增强")
    parser.add_argument("--no-semantics", dest="use_semantics", action="store_false", help="显式禁用 SAM 语义增强")
    parser.add_argument("--filter-threshold", type=float, default=FILTER_THRESHOLD, help="匹配置信度阈值")
    parser.add_argument("--max-visualize", type=int, default=MAX_VISUALIZE, help="诊断大图生成数量")
    parser.add_argument("--out-dir", "--output-dir", dest="output_dir", default=OUTPUT_DIR, help="评测结果根输出目录")
    parser.add_argument("--run-name", default=None, help="本次评测独立的子目录名称 (默认以 eval_YYYYMMDD_HHMMSS 自动命名)")
    parser.add_argument("--device", default=DEVICE, help="计算设备 ('cuda' 或 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 创建独立的运行输出目录 (每次运行独立存放)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = args.run_name if args.run_name else f"eval_{timestamp_str}"
    base_out_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    eval_run_dir = base_out_dir / run_dir_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    # 2. 在运行目录下创建专属 config 文件夹
    config_dir = eval_run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # 3. 解析与定位相关路径
    ckpt_path = (ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    data_dir = (ROOT / args.data_dir).resolve() if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    device = torch.device(args.device)

    print("=" * 75)
    print("  🔍 GlueStick 跨视角几何与语义匹配模型评测 (独立运行归档模式)")
    print("=" * 75)
    print(f"[*] 运行设备: {device}")
    print(f"[*] 评测数据: {data_dir} (子集: {args.split})")
    print(f"[*] 本次独立运行目录: {eval_run_dir}")
    print(f"[*] 配置与脚本归档目录: {config_dir}")

    # 官方基线权重路径定位
    cand_base_weights = [
        ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar",
        ROOT / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar",
        ROOT / "weights" / "gluestick" / "checkpoint_GlueStick_MD.tar",
    ]
    base_weights_path = next((str(p.resolve()) for p in cand_base_weights if p.is_file()), None)

    # 4. 检查点读取、训练参数提取与模型架构自适应
    use_semantics = args.use_semantics
    if use_semantics is None:
        use_semantics = USE_SEMANTICS

    ckpt_data = None
    training_params: dict[str, Any] = {
        "checkpoint_file": ckpt_path.name,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_exists": ckpt_path.is_file(),
    }

    if ckpt_path.is_file():
        print(f"[*] 检测到指定模型检查点: {ckpt_path.name}")
        try:
            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception:
            ckpt_data = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt_data, dict):
            # 提取检查点中保存的训练参数配置
            if "conf" in ckpt_data and isinstance(ckpt_data["conf"], dict):
                training_params["train_args"] = ckpt_data["conf"]
                # 若用户未显式指定，则自动从检查点元数据中恢复 use_semantics
                if args.use_semantics is None and "use_semantics" in ckpt_data["conf"]:
                    use_semantics = ckpt_data["conf"]["use_semantics"]
                    print(f"[*] 自动继承检查点语义配置: use_semantics={use_semantics}")
            if "epoch" in ckpt_data:
                training_params["checkpoint_epoch"] = ckpt_data["epoch"]
            if "val_loss" in ckpt_data:
                training_params["checkpoint_val_loss"] = ckpt_data["val_loss"]
            if "val_line_f1" in ckpt_data:
                training_params["checkpoint_val_line_f1"] = ckpt_data["val_line_f1"]
            if "record" in ckpt_data:
                training_params["checkpoint_record"] = ckpt_data["record"]

        # 检查检查点同级目录下是否存在 train_history.json
        hist_path = ckpt_path.parent / "train_history.json"
        if hist_path.is_file():
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    hist = json.load(f)
                if hist and isinstance(hist, list):
                    best_epoch_record = min(hist, key=lambda x: x.get("val_loss", float("inf")))
                    training_params["train_history_summary"] = {
                        "total_epochs": len(hist),
                        "best_epoch": best_epoch_record.get("epoch"),
                        "best_val_loss": best_epoch_record.get("val_loss"),
                        "last_epoch": hist[-1].get("epoch"),
                    }
            except Exception as e:
                training_params["train_history_error"] = str(e)
    else:
        print(f"[!] 警告: 未找到检查点 {ckpt_path}，将自动使用官方底座预训练权重评测！")
        training_params["source"] = "official_baseline_pretrained"
        if use_semantics is None:
            use_semantics = False

    if use_semantics is None:
        use_semantics = True

    # 5. 归档验证推理代码并写入配置文件
    current_script_path = Path(__file__).resolve()
    archived_script_name = current_script_path.name
    archived_script_path = config_dir / archived_script_name
    try:
        shutil.copy2(current_script_path, archived_script_path)
        print(f"[✓] 验证推理文件已归档至: {archived_script_path.name}")
    except Exception as e:
        print(f"[!] 警告: 归档推理文件时出错: {e}")

    # 组装配置文件字典
    config_dict = {
        "run_info": {
            "timestamp": timestamp_str,
            "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(eval_run_dir),
            "inference_script": {
                "file_name": archived_script_name,
                "original_path": str(current_script_path),
                "archived_path": str(archived_script_path),
                "description": "本次评测验证运行所使用的推理执行文件备份"
            }
        },
        "evaluation_parameters": {
            "checkpoint": str(ckpt_path),
            "data_dir": str(data_dir),
            "split": args.split,
            "use_semantics": use_semantics,
            "filter_threshold": args.filter_threshold,
            "max_visualize": args.max_visualize,
            "device": str(device),
            "batch_size": 1,
        },
        "training_parameters": training_params,
    }

    config_file = config_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"[✓] 评测与训练配置已记录至: config/config.json")

    # 创建指向最新一次评测结果的便捷软链接 (latest)
    try:
        latest_link = base_out_dir / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(eval_run_dir.name, target_is_directory=True)
    except Exception:
        pass

    # 6. 初始化模型
    model = SemanticGlueStick(
        weights=base_weights_path,
        use_semantics=use_semantics,
        filter_threshold=args.filter_threshold,
    ).to(device)

    # 7. 加载训练好的权重
    if ckpt_data is not None:
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[✓] 成功载入模型参数: {ckpt_path.name}")
        if missing:
            print(f"    - 未匹配的层数: {len(missing)}")
        if unexpected:
            print(f"    - 忽略的多余层数: {len(unexpected)}")

    model.eval()

    # 8. 加载数据集 (单线程无后台子进程，便于调试)
    val_ds = SemanticGlueStickDataset(data_dir, split=args.split)
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

            # 纯文本单行打印进度 (前 max_visualize 个样本以及每 20 个样本汇报一次)
            if idx < args.max_visualize or (idx + 1) % 20 == 0 or (idx + 1) == total_samples:
                print(
                    f"  -> [{idx + 1:3d}/{total_samples:3d}] 样本: {batch['name'][0]:<20} | "
                    f"Loss={stats['total_loss']:.3f} | Pt-F1={metrics['point_f1']:.3f}, Ln-F1={metrics['line_f1']:.3f}"
                )

            # 可视化输出前 N 对样本
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

                # 纵向拼接 4 排高清对比诊断大图
                r1 = draw_matches_side_by_side(img0_u8, img1_u8, kpts0, kpts1, pred_m0, title="Row 1: Predicted Point Matches")
                r2 = draw_matches_side_by_side(img0_u8, img1_u8, kpts0, kpts1, gt_m0, title="Row 2: Ground Truth Point Matches")
                r3 = draw_line_matches_side_by_side(img0_u8, img1_u8, lines0, lines1, pred_lm0, title="Row 3: Predicted Line Matches")
                r4 = draw_line_matches_side_by_side(img0_u8, img1_u8, lines0, lines1, gt_lm0, title="Row 4: Ground Truth Line Matches")

                vis_composite = np.concatenate([r1, r2, r3, r4], axis=0)
                save_img_path = eval_run_dir / f"{sample_name}_eval_matches.jpg"
                cv2.imwrite(str(save_img_path), vis_composite)

    # 9. 计算全量平均指标并导出
    n = len(all_metrics)
    summary = {
        "run_dir": str(eval_run_dir),
        "checkpoint": str(ckpt_path),
        "dataset_size": n,
        "mean_loss": float(total_loss / max(n, 1)),
        "point_precision": float(np.mean([m["point_precision"] for m in all_metrics])),
        "point_recall": float(np.mean([m["point_recall"] for m in all_metrics])),
        "point_f1": float(np.mean([m["point_f1"] for m in all_metrics])),
        "line_precision": float(np.mean([m["line_precision"] for m in all_metrics])),
        "line_recall": float(np.mean([m["line_recall"] for m in all_metrics])),
        "line_f1": float(np.mean([m["line_f1"] for m in all_metrics])),
        "config_file": str(config_file),
        "archived_inference_file": str(archived_script_path),
    }

    print("\n" + "=" * 75)
    print("  📊 全量定量评测结果报告 (Evaluation Summary Report)")
    print("=" * 75)
    print(f"  [Loss] 综合平均损失: {summary['mean_loss']:.4f}")
    print(f"  [Points] 关键点匹配: Precision={summary['point_precision']*100:.2f}%, Recall={summary['point_recall']*100:.2f}%, F1={summary['point_f1']:.4f}")
    print(f"  [Lines]  几何线匹配: Precision={summary['line_precision']*100:.2f}%, Recall={summary['line_recall']*100:.2f}%, F1={summary['line_f1']:.4f}")
    print("=" * 75)

    summary_file = eval_run_dir / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[✓] 详细评测指标已导出至: {summary_file}")
    print(f"[✓] 运行配置与推理脚本已归档至: {config_dir}")
    print(f"[✓] 可视化对比图已保存至: {eval_run_dir}")
    print("=" * 75)


if __name__ == "__main__":
    main()
