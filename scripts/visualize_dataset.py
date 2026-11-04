#!/usr/bin/env python3
"""ScanNet 语义增强 GlueStick 数据集综合可视化诊断工具 (Dataset Visualizer).

核心功能:
1. 几何特征检测看板: 并排展示 SuperPoint 关键点与 DeepLSD 直线段;
2. 特征点真值匹配看板: 跨视角彩虹连线展示几何反投影计算的 keypoint GT matches;
3. 直线段真值匹配看板: 对应颜色线段与中点连线展示 3D 反投影计算的 line GT matches;
4. SAM 语义实例掩膜图: 半透明多色实例分割 Masks 叠加，并融合几何线段展示物体边界贴合效果;
5. SAM 2D 密集语义特征图: SAM ViT-H 全图 256 维高分辨率语义嵌入通过跨视角联合 PCA 映射至连续 RGB 色彩空间;
6. 高性能缓存支持: 优先读取预生成的 SAM 语义缓存，兼顾毫秒级极速渲染与在线动态生成。

使用方式:
- PyCharm 中直接右键运行本脚本 (开箱即用，无需配置参数);
- 命令行运行: python scripts/visualize_dataset.py --pair-indices 0 1 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

# ==================== 可视化参数配置 (PyCharm 可直接修改此处) ====================
CONFIG = {
    "data_dir": "data/scannet_semantic_gluestick",  # npz 样本存放目录
    "out_dir": "outputs/dataset_visualizations",    # 可视化结果保存目录
    "sam_cache_dir": "data/scannet_semantic_gluestick/sam_cache",  # SAM 语义图缓存目录
    "sam_checkpoint": "weights/sam/sam_vit_h_4b8939.pth",          # SAM 权重路径
    "pair_indices": [0, 1, 15],                     # 默认可视化的样本索引 (设为 None 则按 max_samples 顺序处理)
    "max_samples": 3,                               # 当 pair_indices 为 None 时，最多可视化的样本数
    "max_draw_matches": 60,                         # 关键点匹配连线最大绘制数 (避免过度密集遮挡)
    "max_draw_line_matches": 45,                    # 直线匹配连线最大绘制数
    "show_sam_masks": True,                         # 是否可视化 SAM 语义实例掩膜图 (含线段轮廓贴合)
    "show_sam_dense_pca": True,                     # 是否可视化 SAM 2D 密集语义特征图 (256-d PCA)
    "enable_sam_generation": True,                  # 缓存缺失时，是否自动调用 SAM 现场生成
    "line_thickness": 2,                            # 几何线段绘制线宽
    "point_radius": 3,                              # 关键点绘制半径
}
# ==============================================================================

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 全局延迟加载 SAM 模型单例 (避免重复加载 2.4GB 权重)
_GLOBAL_SAM_EXTRACTOR = None
_GLOBAL_MASK_GENERATOR = None


def get_sam_models(checkpoint_path: str, device: str = "cuda"):
    """按需延迟初始化 SAM 提取器与 Mask 生成器单例。"""
    global _GLOBAL_SAM_EXTRACTOR, _GLOBAL_MASK_GENERATOR
    if _GLOBAL_SAM_EXTRACTOR is not None:
        return _GLOBAL_SAM_EXTRACTOR, _GLOBAL_MASK_GENERATOR

    ckpt_p = Path(checkpoint_path).resolve()
    if not ckpt_p.is_file():
        return None, None

    try:
        import torch
        from extractors.sam_extractor import SAMExtractor
        from segment_anything import SamAutomaticMaskGenerator

        if not torch.cuda.is_available():
            device = "cpu"

        print(f"[*] 正在加载 SAM 旗舰大模型 ({ckpt_p.name}) 用于动态生成语义图...")
        samext = SAMExtractor(checkpoint_path=ckpt_p, device=device)
        mask_gen = SamAutomaticMaskGenerator(
            model=samext.sam,
            points_per_side=16,
            points_per_batch=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            min_mask_region_area=100,
        )
        _GLOBAL_SAM_EXTRACTOR = samext
        _GLOBAL_MASK_GENERATOR = mask_gen
        print("[✓] SAM 语义生成引擎准备完毕！")
        return _GLOBAL_SAM_EXTRACTOR, _GLOBAL_MASK_GENERATOR
    except Exception as e:
        print(f"[!] 初始化 SAM 失败 ({e})，将跳过在线语义生成。")
        return None, None


def render_mask_overlay(
    img_rgb: np.ndarray,
    masks: list[dict[str, Any]],
    alpha: float = 0.45,
    seed: int = 42,
) -> np.ndarray:
    """渲染多色半透明实例掩膜及轮廓线。"""
    rng = np.random.RandomState(seed)
    h, w = img_rgb.shape[:2]
    if len(masks) == 0:
        return img_rgb.copy()

    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    mask_layer = np.zeros_like(img_rgb)
    weight_layer = np.zeros((h, w), dtype=np.float32)
    colors = [rng.randint(60, 240, size=3, dtype=np.uint8) for _ in sorted_masks]

    for m_dict, col in zip(sorted_masks, colors):
        seg = m_dict["segmentation"]
        mask_layer[seg] = col
        weight_layer[seg] = 1.0

    blended = np.where(
        weight_layer[..., None] > 0,
        (img_rgb.astype(np.float32) * (1.0 - alpha) + mask_layer.astype(np.float32) * alpha).astype(np.uint8),
        img_rgb,
    )

    for m_dict, col in zip(sorted_masks, colors):
        seg = m_dict["segmentation"].astype(np.uint8)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_col = np.clip(col.astype(int) + 40, 0, 255).tolist()
        cv2.drawContours(blended, cnts, -1, border_col, 1, lineType=cv2.LINE_AA)

    return blended


def compute_dense_pca(feat0: Any, feat1: Any) -> tuple[np.ndarray, np.ndarray]:
    """通过联合 SVD/PCA 将跨视角 256 维密集特征映射为连续 RGB 色彩图。"""
    f0_np = feat0.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 256)
    f1_np = feat1.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, 256)
    all_f = np.concatenate([f0_np, f1_np], axis=0)

    mean_f = np.mean(all_f, axis=0, keepdims=True)
    centered = all_f - mean_f
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    feat_3d = centered @ vt[:3].T

    f_min = feat_3d.min(axis=0, keepdims=True)
    f_max = feat_3d.max(axis=0, keepdims=True)
    rgb_all = ((feat_3d - f_min) / (f_max - f_min + 1e-6) * 220.0 + 25.0).astype(np.uint8)

    rgb0 = rgb_all[:len(f0_np)].reshape(64, 64, 3)
    rgb1 = rgb_all[len(f0_np):].reshape(64, 64, 3)

    rgb0_up = cv2.resize(rgb0, (640, 480), interpolation=cv2.INTER_CUBIC)
    rgb1_up = cv2.resize(rgb1, (640, 480), interpolation=cv2.INTER_CUBIC)
    return rgb0_up, rgb1_up


def get_or_generate_sam_maps(
    img0_rgb: np.ndarray,
    img1_rgb: np.ndarray,
    sample_name: str,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """优先从缓存加载 SAM 语义图，若不存在且允许则调用 SAM 动态计算并持久化缓存。"""
    cache_dir = Path(cfg.get("sam_cache_dir", "data/scannet_semantic_gluestick/sam_cache")).resolve()
    cache_file = cache_dir / f"{sample_name}_sam.npz"

    # 1. 尝试直接读取缓存
    if cache_file.is_file():
        try:
            cached = np.load(str(cache_file), allow_pickle=True)
            return {
                "mask_overlay0": cached["mask_overlay0"],
                "mask_overlay1": cached["mask_overlay1"],
                "dense_pca0": cached["dense_pca0"],
                "dense_pca1": cached["dense_pca1"],
                "num_masks0": int(cached["num_masks0"]),
                "num_masks1": int(cached["num_masks1"]),
            }
        except Exception:
            pass

    # 2. 若无缓存，检查是否允许现场调用 SAM 生成
    if not cfg.get("enable_sam_generation", True):
        return None

    samext, mask_gen = get_sam_models(cfg.get("sam_checkpoint", ""))
    if samext is None or mask_gen is None:
        return None

    import torch

    try:
        t0 = time.time()
        with torch.amp.autocast("cuda", enabled=(samext.device.type == "cuda")):
            m0 = mask_gen.generate(img0_rgb)
            m1 = mask_gen.generate(img1_rgb)
            f0, _ = samext.extract_image_embeddings(img0_rgb)
            f1, _ = samext.extract_image_embeddings(img1_rgb)

        overlay0 = render_mask_overlay(img0_rgb, m0)
        overlay1 = render_mask_overlay(img1_rgb, m1)
        pca0, pca1 = compute_dense_pca(f0, f1)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_file),
            mask_overlay0=overlay0,
            mask_overlay1=overlay1,
            dense_pca0=pca0,
            dense_pca1=pca1,
            num_masks0=len(m0),
            num_masks1=len(m1),
        )
        print(f"  [+] 已动态生成并保存 SAM 语义图缓存: {cache_file.name} (耗时: {time.time()-t0:.2f}s)")

        return {
            "mask_overlay0": overlay0,
            "mask_overlay1": overlay1,
            "dense_pca0": pca0,
            "dense_pca1": pca1,
            "num_masks0": len(m0),
            "num_masks1": len(m1),
        }
    except Exception as e:
        print(f"  [!] 动态生成 SAM 语义图出错: {e}")
        return None


def create_header_banner(
    sample_name: str,
    img_size: tuple[int, int],
    n_kpts0: int,
    n_kpts1: int,
    n_lines0: int,
    n_lines1: int,
    gt_pts: int,
    gt_lines: int,
    width: int,
    height: int = 76,
) -> np.ndarray:
    """创建专业学术风格的深色顶部信息标头。"""
    banner = np.full((height, width, 3), (28, 28, 32), dtype=np.uint8)

    cv2.line(banner, (0, 0), (width, 0), (255, 180, 50), 3)

    title = f"ScanNet-1500 Multi-Modal Semantic GlueStick | Sample: {sample_name} | Size: {img_size[1]}x{img_size[0]}"
    cv2.putText(banner, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    stats = (
        f"Keypoints: V0={n_kpts0}, V1={n_kpts1} (GT Matches: {gt_pts})   |   "
        f"Lines: V0={n_lines0}, V1={n_lines1} (GT Matches: {gt_lines})   |   "
        f"SAM ViT-H Semantics: Dual 2D Maps (Instance Masks & Dense Feature PCA)"
    )
    cv2.putText(banner, stats, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 240), 1, cv2.LINE_AA)

    return banner


def draw_detection_panel(
    img0: np.ndarray,
    img1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    point_radius: int = 3,
    line_thickness: int = 2,
) -> np.ndarray:
    """Panel 1: 几何特征检测看板 (展示 SuperPoint 关键点与 DeepLSD 直线段)。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    for l in lines0:
        p1 = (int(round(l[0, 0])), int(round(l[0, 1])))
        p2 = (int(round(l[1, 0])), int(round(l[1, 1])))
        cv2.line(bgr0, p1, p2, (0, 230, 0), line_thickness, cv2.LINE_AA)
        cv2.circle(bgr0, p1, 2, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr0, p2, 2, (0, 140, 255), -1, cv2.LINE_AA)

    for pt in kpts0:
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(bgr0, p, point_radius, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(bgr0, p, 1, (255, 255, 255), -1, cv2.LINE_AA)

    for l in lines1:
        p1 = (int(round(l[0, 0])), int(round(l[0, 1])))
        p2 = (int(round(l[1, 0])), int(round(l[1, 1])))
        cv2.line(bgr1, p1, p2, (0, 230, 0), line_thickness, cv2.LINE_AA)
        cv2.circle(bgr1, p1, 2, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr1, p2, 2, (0, 140, 255), -1, cv2.LINE_AA)

    for pt in kpts1:
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(bgr1, p, point_radius, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(bgr1, p, 1, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.rectangle(bgr0, (10, 10), (380, 42), (0, 0, 0), -1)
    cv2.putText(bgr0, f"View 0: SuperPoint ({len(kpts0)}) + DeepLSD ({len(lines0)})", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(bgr1, (10, 10), (380, 42), (0, 0, 0), -1)
    cv2.putText(bgr1, f"View 1: SuperPoint ({len(kpts1)}) + DeepLSD ({len(lines1)})", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1
    return canvas


def draw_kpt_matches_panel(
    img0: np.ndarray,
    img1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    gt_matches0: np.ndarray,
    max_draw: int = 60,
) -> np.ndarray:
    """Panel 2: 特征点真值匹配看板 (彩虹连线跨视角展示 gt_matches0)。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1

    valid_matches = []
    for i, j in enumerate(gt_matches0):
        if j >= 0 and j < len(kpts1):
            valid_matches.append((i, j))

    total_gt = len(valid_matches)
    if len(valid_matches) > max_draw:
        step = len(valid_matches) / max_draw
        sampled = [valid_matches[int(k * step)] for k in range(max_draw)]
    else:
        sampled = valid_matches

    cmap = cv2.applyColorMap(
        np.linspace(0, 255, max(len(sampled), 1), dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_RAINBOW,
    )

    for k, (i, j) in enumerate(sampled):
        col = tuple(int(c) for c in cmap[k, 0])
        p0 = (int(round(kpts0[i, 0])), int(round(kpts0[i, 1])))
        p1 = (int(round(kpts1[j, 0])) + w0, int(round(kpts1[j, 1])))

        cv2.circle(canvas, p0, 4, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, col, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (10, 10), (510, 42), (0, 0, 0), -1)
    label = f"GT Keypoint Matches (Total: {total_gt}, Showing: {len(sampled)})"
    cv2.putText(canvas, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 255, 100), 1, cv2.LINE_AA)

    return canvas


def draw_line_matches_panel(
    img0: np.ndarray,
    img1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    gt_line_matches0: np.ndarray,
    max_draw: int = 45,
) -> np.ndarray:
    """Panel 3: 直线段真值匹配看板 (对应同色线段与中心连线展示 gt_line_matches0)。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0.copy()
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1

    valid_line_matches = []
    for i, j in enumerate(gt_line_matches0):
        if j >= 0 and j < len(lines1):
            valid_line_matches.append((i, j))

    total_gt = len(valid_line_matches)
    if len(valid_line_matches) > max_draw:
        step = len(valid_line_matches) / max_draw
        sampled = [valid_line_matches[int(k * step)] for k in range(max_draw)]
    else:
        sampled = valid_line_matches

    cmap = cv2.applyColorMap(
        np.linspace(0, 255, max(len(sampled), 1), dtype=np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO,
    )

    for k, (i, j) in enumerate(sampled):
        col = tuple(int(c) for c in cmap[k, 0])
        l0 = lines0[i]
        l1 = lines1[j]

        p0_s = (int(round(l0[0, 0])), int(round(l0[0, 1])))
        p0_e = (int(round(l0[1, 0])), int(round(l0[1, 1])))
        p1_s = (int(round(l1[0, 0])) + w0, int(round(l1[0, 1])))
        p1_e = (int(round(l1[1, 0])) + w0, int(round(l1[1, 1])))

        cv2.line(canvas, p0_s, p0_e, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, p0_s, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p0_e, 3, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.line(canvas, p1_s, p1_e, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, p1_s, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1_e, 3, (255, 255, 255), -1, cv2.LINE_AA)

        mid0 = ((p0_s[0] + p0_e[0]) // 2, (p0_s[1] + p0_e[1]) // 2)
        mid1 = ((p1_s[0] + p1_e[0]) // 2, (p1_s[1] + p1_e[1]) // 2)
        cv2.line(canvas, mid0, mid1, col, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (10, 10), (480, 42), (0, 0, 0), -1)
    label = f"GT Line Matches (Total: {total_gt}, Showing: {len(sampled)})"
    cv2.putText(canvas, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 50), 1, cv2.LINE_AA)

    return canvas


def draw_sam_masks_panel(
    mask_overlay0: np.ndarray,
    mask_overlay1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    num_m0: int,
    num_m1: int,
) -> np.ndarray:
    """Panel 4: SAM 语义实例掩膜图 (半透明实例 Masks + DeepLSD 几何线段轮廓贴合展示)。"""
    h0, w0 = mask_overlay0.shape[:2]
    h1, w1 = mask_overlay1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    # 转换 RGB 为 BGR
    bgr0 = cv2.cvtColor(mask_overlay0, cv2.COLOR_RGB2BGR)
    bgr1 = cv2.cvtColor(mask_overlay1, cv2.COLOR_RGB2BGR)

    # 在掩膜上叠加亮黄色 DeepLSD 直线，直观检验直线与物体物理轮廓的契合度
    for l in lines0:
        p1 = (int(round(l[0, 0])), int(round(l[0, 1])))
        p2 = (int(round(l[1, 0])), int(round(l[1, 1])))
        cv2.line(bgr0, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

    for l in lines1:
        p1 = (int(round(l[0, 0])), int(round(l[0, 1])))
        p2 = (int(round(l[1, 0])), int(round(l[1, 1])))
        cv2.line(bgr1, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.rectangle(bgr0, (10, 10), (450, 42), (0, 0, 0), -1)
    cv2.putText(bgr0, f"View 0: SAM Masks ({num_m0} instances) + Boundary Lines", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(bgr1, (10, 10), (450, 42), (0, 0, 0), -1)
    cv2.putText(bgr1, f"View 1: SAM Masks ({num_m1} instances) + Boundary Lines", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1
    return canvas


def draw_sam_dense_pca_panel(
    dense_pca0: np.ndarray,
    dense_pca1: np.ndarray,
) -> np.ndarray:
    """Panel 5: SAM 2D 密集语义特征图 (ViT-H 256 维高分辨率语义嵌入联合 PCA 色彩图)。"""
    h0, w0 = dense_pca0.shape[:2]
    h1, w1 = dense_pca1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(dense_pca0, cv2.COLOR_RGB2BGR)
    bgr1 = cv2.cvtColor(dense_pca1, cv2.COLOR_RGB2BGR)

    cv2.rectangle(bgr0, (10, 10), (480, 42), (0, 0, 0), -1)
    cv2.putText(bgr0, "View 0: SAM ViT-H Dense Semantic PCA Feature Map", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(bgr1, (10, 10), (480, 42), (0, 0, 0), -1)
    cv2.putText(bgr1, "View 1: SAM ViT-H Dense Semantic PCA Feature Map", (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1
    return canvas


def render_composite_dashboard(
    data: dict[str, Any],
    sample_name: str,
    config: dict[str, Any],
) -> np.ndarray:
    """拼装多面板综合诊断大图 (包含几何、GT真值标注与 SAM 双语义图)。"""
    img0 = data["image0"]
    img1 = data["image1"]
    kpts0 = data["keypoints0"]
    kpts1 = data["keypoints1"]
    lines0 = data["lines0"]
    lines1 = data["lines1"]
    gt_m0 = data["gt_matches0"]
    gt_lm0 = data["gt_line_matches0"]

    w_total = img0.shape[1] + img1.shape[1]
    gt_pts_count = int(np.sum(gt_m0 >= 0))
    gt_lines_count = int(np.sum(gt_lm0 >= 0))

    # 1. 顶部 Header
    banner = create_header_banner(
        sample_name=sample_name,
        img_size=img0.shape[:2],
        n_kpts0=len(kpts0),
        n_kpts1=len(kpts1),
        n_lines0=len(lines0),
        n_lines1=len(lines1),
        gt_pts=gt_pts_count,
        gt_lines=gt_lines_count,
        width=w_total,
    )

    # 2. 面板 1: 特征检测
    p1 = draw_detection_panel(
        img0, img1, kpts0, kpts1, lines0, lines1,
        point_radius=config.get("point_radius", 3),
        line_thickness=config.get("line_thickness", 2),
    )

    # 3. 面板 2: 关键点真值匹配
    p2 = draw_kpt_matches_panel(
        img0, img1, kpts0, kpts1, gt_m0,
        max_draw=config.get("max_draw_matches", 60),
    )

    # 4. 面板 3: 直线真值匹配
    p3 = draw_line_matches_panel(
        img0, img1, lines0, lines1, gt_lm0,
        max_draw=config.get("max_draw_line_matches", 45),
    )

    panels = [banner, p1, p2, p3]

    # 5. 加载 / 生成 SAM 语义图 (实例 Masks 与 密集特征图)
    if config.get("show_sam_masks", True) or config.get("show_sam_dense_pca", True):
        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB) if img0.ndim == 2 else img0
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB) if img1.ndim == 2 else img1

        sam_maps = get_or_generate_sam_maps(img0_rgb, img1_rgb, sample_name, config)
        if sam_maps is not None:
            if config.get("show_sam_masks", True):
                p4 = draw_sam_masks_panel(
                    sam_maps["mask_overlay0"], sam_maps["mask_overlay1"],
                    lines0, lines1,
                    sam_maps["num_masks0"], sam_maps["num_masks1"],
                )
                panels.append(p4)

            if config.get("show_sam_dense_pca", True):
                p5 = draw_sam_dense_pca_panel(sam_maps["dense_pca0"], sam_maps["dense_pca1"])
                panels.append(p5)

    # 垂直拼装
    composite = np.concatenate(panels, axis=0)
    return composite


def main():
    parser = argparse.ArgumentParser(description="ScanNet 语义 GlueStick 数据集综合可视化工具")
    parser.add_argument("--data-dir", default=CONFIG["data_dir"], help="npz 数据集目录")
    parser.add_argument("--out-dir", default=CONFIG["out_dir"], help="输出保存目录")
    parser.add_argument("--pair-indices", nargs="+", type=int, default=CONFIG["pair_indices"],
                        help="指定需要可视化的样本索引 (如: 0 1 15)")
    parser.add_argument("--max-samples", type=int, default=CONFIG["max_samples"],
                        help="未指定 pair-indices 时处理的最大样本数")
    parser.add_argument("--no-sam-masks", action="store_true", help="禁用 SAM 语义实例掩膜图")
    parser.add_argument("--no-sam-pca", action="store_true", help="禁用 SAM 密集语义特征图")
    args = parser.parse_args()

    cfg = dict(CONFIG)
    cfg["data_dir"] = args.data_dir
    cfg["out_dir"] = args.out_dir
    if args.no_sam_masks:
        cfg["show_sam_masks"] = False
    if args.no_sam_pca:
        cfg["show_sam_dense_pca"] = False

    data_dir = Path(cfg["data_dir"]).resolve()
    out_dir = Path(cfg["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"[错误] 数据集目录不存在: {data_dir}")
        return

    npz_files = sorted(list(data_dir.glob("pair_*.npz")))
    if not npz_files:
        print(f"[错误] 在 {data_dir} 中未发现 pair_*.npz 样本文件！")
        return

    if args.pair_indices is not None and len(args.pair_indices) > 0:
        target_files = []
        for idx in args.pair_indices:
            candidate = data_dir / f"pair_{idx:05d}.npz"
            if candidate.is_file():
                target_files.append(candidate)
            elif idx < len(npz_files):
                target_files.append(npz_files[idx])
            else:
                print(f"[警告] 样本索引超出范围: {idx} (总数: {len(npz_files)})")
    else:
        target_files = npz_files[:cfg["max_samples"]]

    print("=" * 80)
    print(" ScanNet 语义 GlueStick 数据集全维度可视化诊断启动 (含几何检测、GT标注与SAM双语义图)")
    print(f" 数据集目录:   {data_dir}")
    print(f" 输出保存目录: {out_dir}")
    print(f" 待处理样本:   {len(target_files)} 对")
    print("=" * 80)

    t_start = time.time()
    for count, npz_path in enumerate(target_files, 1):
        sample_name = npz_path.stem
        raw_data = np.load(str(npz_path), allow_pickle=True)
        data = {k: raw_data[k] for k in raw_data.files}

        dashboard = render_composite_dashboard(data, sample_name, cfg)

        out_path = out_dir / f"{sample_name}_dataset_vis.jpg"
        cv2.imwrite(str(out_path), dashboard)

        n_pts = len(data.get("keypoints0", []))
        n_lines = len(data.get("lines0", []))
        gt_pts = int(np.sum(data.get("gt_matches0", np.array([])) >= 0))
        gt_lines = int(np.sum(data.get("gt_line_matches0", np.array([])) >= 0))

        print(
            f" [{count}/{len(target_files)}] 已渲染: {sample_name} | "
            f"点={n_pts} (GT={gt_pts}), 线={n_lines} (GT={gt_lines}) | "
            f"大图尺寸: {dashboard.shape[1]}x{dashboard.shape[0]} -> {out_path.name}"
        )

    t_total = time.time() - t_start
    print("-" * 80)
    print(f"[✓] 批量渲染完成！共处理 {len(target_files)} 对样本，总耗时 {t_total:.2f} 秒 (平均 {t_total / max(len(target_files), 1):.2f}s/对)")
    print(f"[✓] 包含 SAM 语义图的可视化大图请查看: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
