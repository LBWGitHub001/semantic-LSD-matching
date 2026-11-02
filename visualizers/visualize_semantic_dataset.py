"""高精度语义数据集多场景可视化与画廊生成工具 (Multi-Scene Semantic Visualizer).

功能:
1. 批量遍历数据集中不同场景的样本，渲染 3 行综合诊断看板 (几何线段、SAM ViT-H 语义着色、跨视角真值匹配);
2. 自动生成多场景全景对比看板 (multi_scene_overview.jpg)，并排展示不同室内场景中物体的语义线染色效果;
3. 保存各样本独立高清诊断图至 outputs/visualizations/{sample_name}.jpg。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.decomposition import PCA
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.semantic_gluestick_dataset import SemanticGlueStickDataset


def color_lines_by_semantic_pca(
    descs0: np.ndarray,
    descs1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """通过全局 PCA 将 256 维语义特征降维至 3 维 RGB 颜色空间。"""
    m0 = descs0.shape[1]
    m1 = descs1.shape[1]

    if m0 + m1 == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8)

    all_descs = np.concatenate([descs0.T, descs1.T], axis=0)  # (M0+M1, 256)

    if len(all_descs) < 3:
        c0 = np.full((m0, 3), 128, dtype=np.uint8)
        c1 = np.full((m1, 3), 128, dtype=np.uint8)
        return c0, c1

    pca = PCA(n_components=3, random_state=42)
    feat_3d = pca.fit_transform(all_descs)

    f_min = feat_3d.min(axis=0, keepdims=True)
    f_max = feat_3d.max(axis=0, keepdims=True)
    f_range = np.maximum(f_max - f_min, 1e-5)
    rgb = ((feat_3d - f_min) / f_range * 210.0 + 30.0).astype(np.uint8)

    bgr = rgb[:, [2, 1, 0]]
    colors0 = bgr[:m0]
    colors1 = bgr[m0:]

    return colors0, colors1


def draw_lines_on_image(
    image_gray: np.ndarray,
    lines: np.ndarray,
    colors: np.ndarray | None = None,
    default_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像上绘制几何线段。"""
    if image_gray.ndim == 2:
        vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        vis = image_gray.copy()

    for idx, line in enumerate(lines):
        pt1 = (int(round(line[0, 0])), int(round(line[0, 1])))
        pt2 = (int(round(line[1, 0])), int(round(line[1, 1])))
        color = tuple(int(c) for c in colors[idx]) if colors is not None else default_color
        cv2.line(vis, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt2, 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    return vis


def draw_cross_view_matches(
    img0: np.ndarray,
    img1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    gt_line_matches0: np.ndarray,
    max_draw: int = 40,
) -> np.ndarray:
    """并排绘制跨视角线段匹配连线图。"""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)

    bgr0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR) if img0.ndim == 2 else img0
    bgr1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1

    canvas[:h0, :w0] = bgr0
    canvas[:h1, w0:] = bgr1

    matched_pairs = []
    for i, j in enumerate(gt_line_matches0):
        if j >= 0 and j < len(lines1):
            matched_pairs.append((i, j))

    if len(matched_pairs) > max_draw:
        step = len(matched_pairs) // max_draw
        matched_pairs = matched_pairs[::step][:max_draw]

    cmap = cv2.applyColorMap(np.linspace(0, 255, max(len(matched_pairs), 1), dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)

    for k, (i, j) in enumerate(matched_pairs):
        col = tuple(int(c) for c in cmap[k, 0])
        l0 = lines0[i]
        l1 = lines1[j]

        p0_start = (int(l0[0, 0]), int(l0[0, 1]))
        p0_end = (int(l0[1, 0]), int(l0[1, 1]))
        cv2.line(canvas, p0_start, p0_end, col, 2, cv2.LINE_AA)

        p1_start = (int(l1[0, 0]) + w0, int(l1[0, 1]))
        p1_end = (int(l1[1, 0]) + w0, int(l1[1, 1]))
        cv2.line(canvas, p1_start, p1_end, col, 2, cv2.LINE_AA)

        mid0 = ((p0_start[0] + p0_end[0]) // 2, (p0_start[1] + p0_end[1]) // 2)
        mid1 = ((p1_start[0] + p1_end[0]) // 2, (p1_start[1] + p1_end[1]) // 2)
        cv2.line(canvas, mid0, mid1, col, 1, cv2.LINE_AA)

    return canvas


def render_sample_board(data: dict[str, Any], sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    """为单对样本渲染完整的 3 行综合看板与语义对比切片。

    返回:
        composite: 3 行全量复合诊断看板
        semantic_slice: 包含原图与 PCA 语义线染色的紧凑横条 (供生成多场景合辑)
    """
    img0 = data["image0"]
    img1 = data["image1"]
    lines0 = data["lines0"]
    lines1 = data["lines1"]
    descs0 = data["line_semantic_descs0"]
    descs1 = data["line_semantic_descs1"]
    gt_lm0 = data["gt_line_matches0"]

    # 1. 计算 PCA 语义色彩
    colors0, colors1 = color_lines_by_semantic_pca(descs0, descs1)

    # Panel 1: View 0 原图 + 绿色几何线
    p1 = draw_lines_on_image(img0, lines0, default_color=(0, 255, 0))
    cv2.putText(p1, f"View 0: DeepLSD ({len(lines0)} lines)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Panel 2: View 1 原图 + 绿色几何线
    p2 = draw_lines_on_image(img1, lines1, default_color=(0, 255, 0))
    cv2.putText(p2, f"View 1: DeepLSD ({len(lines1)} lines)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Panel 3: View 0 SAM ViT-H 语义着色线段
    p3 = draw_lines_on_image(img0, lines0, colors=colors0)
    cv2.putText(p3, f"{sample_name} | V0: SAM ViT-H PCA", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Panel 4: View 1 SAM ViT-H 语义着色线段
    p4 = draw_lines_on_image(img1, lines1, colors=colors1)
    cv2.putText(p4, f"{sample_name} | V1: SAM ViT-H PCA", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Panel 5: 跨视角真值匹配
    p5 = draw_cross_view_matches(img0, img1, lines0, lines1, gt_lm0)
    num_matches = int(np.sum(gt_lm0 >= 0))
    cv2.putText(p5, f"Cross-View GT Matches ({num_matches} matched lines)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    row1 = np.concatenate([p1, p2], axis=1)
    row2 = np.concatenate([p3, p4], axis=1)
    row3 = p5

    composite = np.concatenate([row1, row2, row3], axis=0)
    return composite, row2


def main():
    parser = argparse.ArgumentParser(description="多场景语义增强数据集可视化诊断工具")
    parser.add_argument("--data-dir", default="data/test_semantic_gluestick", help="数据集路径")
    parser.add_argument("--out-dir", default="outputs/visualizations", help="独立样本保存目录")
    parser.add_argument("--overview-path", default="outputs/multi_scene_overview.jpg", help="多场景全景合辑路径")
    parser.add_argument("--preview-path", default="outputs/semantic_dataset_preview.jpg", help="主预览图路径")
    parser.add_argument("--max-overview-scenes", type=int, default=6, help="全景大图最多展示几个场景")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overview_path = Path(args.overview_path).resolve()
    overview_path.parent.mkdir(parents=True, exist_ok=True)

    ds = SemanticGlueStickDataset(args.data_dir)
    print("=" * 70)
    print(f"[*] 启动多场景可视化生成: 数据集共有 {len(ds)} 个样本")
    print(f"[*] 独立看板输出目录: {out_dir}")
    print(f"[*] 全景合辑保存路径: {overview_path}")
    print("=" * 70)

    scene_slices: dict[str, np.ndarray] = {}

    for idx in range(len(ds)):
        name = ds.sample_names[idx]
        data = ds.get_numpy_data(idx)
        composite, row_semantic = render_sample_board(data, name)

        sample_out = out_dir / f"{name}.jpg"
        cv2.imwrite(str(sample_out), composite)
        print(f"  [✓] 已生成样本看板: {sample_out.name}")

        # 记录每个场景的第 1 个样本作为多场景全景切片
        scene = name.split("_pair")[0] if "_pair" in name else name
        if scene not in scene_slices and len(scene_slices) < args.max_overview_scenes:
            scene_slices[scene] = row_semantic

        # 默认使用第 1 个样本更新单图主预览
        if idx == 0:
            cv2.imwrite(str(Path(args.preview_path).resolve()), composite)

    # 拼装多场景全景对比大图 (垂直堆叠各场景的语义对比切片)
    if len(scene_slices) > 0:
        slices = list(scene_slices.values())
        overview_img = np.concatenate(slices, axis=0)
        cv2.imwrite(str(overview_path), overview_img)
        print(f"\n[✓] 包含 {len(scene_slices)} 个场景的全景对比大图已生成: {overview_path}")

    print("\n[✓] 所有多场景可视化渲染完毕！")


if __name__ == "__main__":
    main()
