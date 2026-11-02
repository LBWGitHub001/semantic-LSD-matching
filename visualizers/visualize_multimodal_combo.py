"""多模态几何与语义特征四合一综合可视化工具 (Multi-Modal Combo Visualizer).

核心功能:
将以下 4 项核心视觉内容合成单张高清综合对比看板:
1. 原图 (Original RGB Image);
2. DeepLSD 输出的直线场 (仅几何距离场 Distance Field);
3. SAM 输出的语义实例掩膜 (Semantic Masks) 叠加在原图上;
4. 原图 + SAM 语义掩膜 + DeepLSD 几何线段 (全要素融合).

输出格式:
- 经典 2x2 四宫格画板 (附带专业学术标头与图例)
- 1x4 横向全景对比带 (便于论文与网页并排比对展示)
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractors.deeplsd_extractor import DeepLSDExtractor
from extractors.sam_extractor import SAMExtractor


def render_distance_field(
    df: np.ndarray,
    max_dist: float = 5.0,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """渲染 DeepLSD 几何距离场 (仅距离，DF).

    距离越小 (越靠近直线，0px) 强度越高，距离越大 (背景区域，5px) 呈现深蓝/冷色。
    """
    # 归一化并反转: 0px 对应 255 (高亮), 5px 对应 0 (暗背景)
    norm_df = np.clip(df / max_dist, 0.0, 1.0)
    inverted = ((1.0 - norm_df) * 255.0).astype(np.uint8)
    color_df = cv2.applyColorMap(inverted, colormap)

    # 绘制距离刻度标尺 (Legend) 于右下角
    h, w = color_df.shape[:2]
    bar_w, bar_h = 140, 14
    margin_x, margin_y = 20, 20
    x1, y1 = w - bar_w - margin_x, h - bar_h - margin_y
    x2, y2 = w - margin_x, h - margin_y

    # 绘制半透明黑底背景盒
    sub_bg = color_df[y1 - 20:y2 + 8, x1 - 10:x2 + 10]
    color_df[y1 - 20:y2 + 8, x1 - 10:x2 + 10] = (sub_bg * 0.25).astype(np.uint8)

    # 生成色彩渐变条
    grad = np.linspace(255, 0, bar_w, dtype=np.uint8).reshape(1, bar_w)
    grad_color = cv2.applyColorMap(grad, colormap)
    grad_color = cv2.resize(grad_color, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)
    color_df[y1:y2, x1:x2] = grad_color
    cv2.rectangle(color_df, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 刻度文字
    cv2.putText(color_df, "0px (Line)", (x1 - 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(color_df, "5px (Far)", (x2 - 50, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    return color_df


def render_sam_masks_overlay(
    image_rgb: np.ndarray,
    masks: list[dict[str, Any]],
    alpha: float = 0.45,
    draw_contours: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """将 SAM 预测的语义实例掩膜以半透明方式叠加在原图上。"""
    np.random.seed(seed)
    h, w = image_rgb.shape[:2]
    if len(masks) == 0:
        return image_rgb.copy()

    # 面积从大到小排序，确保小实例在上层
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    colors = []
    for _ in range(len(sorted_masks)):
        col = np.random.randint(50, 240, size=3, dtype=np.uint8)
        colors.append(col)

    mask_layer = np.zeros_like(image_rgb)
    weight_layer = np.zeros((h, w), dtype=np.float32)

    for mask_dict, col in zip(sorted_masks, colors):
        seg = mask_dict["segmentation"]
        mask_layer[seg] = col
        weight_layer[seg] = 1.0

    blended = np.where(
        weight_layer[..., None] > 0,
        (image_rgb.astype(np.float32) * (1.0 - alpha) + mask_layer.astype(np.float32) * alpha).astype(np.uint8),
        image_rgb,
    )

    if draw_contours:
        for mask_dict, col in zip(sorted_masks, colors):
            seg = mask_dict["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border_color = np.clip(col.astype(int) + 35, 0, 255).tolist()
            cv2.drawContours(blended, contours, -1, border_color, 1, lineType=cv2.LINE_AA)

    return blended


def render_overlay_with_lines(
    base_img_rgb: np.ndarray,
    lines: np.ndarray,
    line_color: tuple[int, int, int] = (0, 255, 255),  # 高亮黄色
    point_color: tuple[int, int, int] = (255, 50, 50), # 红色端点
    thickness: int = 2,
    radius: int = 3,
) -> np.ndarray:
    """在底图之上叠加绘制高精度亚像素 DeepLSD 直线段。"""
    res = base_img_rgb.copy()
    for l in lines:
        p1 = (int(round(l[0, 0])), int(round(l[0, 1])))
        p2 = (int(round(l[1, 0])), int(round(l[1, 1])))
        cv2.line(res, p1, p2, line_color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(res, p1, radius, point_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(res, p2, radius, point_color, -1, lineType=cv2.LINE_AA)
    return res


def add_panel_label(
    panel_bgr: np.ndarray,
    title: str,
    subtitle: str | None = None,
    badge_color: tuple[int, int, int] = (0, 120, 215),
) -> np.ndarray:
    """为单张子图添加规范的信息标签卡。"""
    out = panel_bgr.copy()
    h, w = out.shape[:2]

    card_h = 36 if subtitle is None else 50
    card_w = 360
    # 左上角半透明信息条
    overlay = out.copy()
    cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # 色彩装饰边条
    cv2.rectangle(out, (12, 12), (16, 12 + card_h), badge_color, -1)

    # 标题文字
    cv2.putText(out, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def build_combo_visualizations(
    img_rgb: np.ndarray,
    df: np.ndarray,
    masks: list[dict[str, Any]],
    lines: np.ndarray,
    scene_name: str = "ScanNet Scene",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """生成 4 合 1 组合图 (2x2 宫格与 1x4 横条).

    返回:
        grid_2x2: 2x2 宫格大图
        strip_1x4: 1x4 横向排布大图
        panels_dict: 包含 4 个独立面板图像的字典
    """
    # 1. 原图 (RGB -> BGR 供 OpenCV 输出)
    p1 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    p1 = add_panel_label(p1, "(a) Original RGB Image", "Raw Scene Camera Capture", badge_color=(0, 150, 255))

    # 2. DeepLSD 几何距离场 (仅距离)
    p2 = render_distance_field(df, max_dist=5.0)
    p2 = add_panel_label(p2, "(b) DeepLSD Distance Field", "Distance to Nearest Line (DF: 0-5px)", badge_color=(255, 100, 0))

    # 3. SAM 语义掩膜叠加在原图上
    masked_rgb = render_sam_masks_overlay(img_rgb, masks, alpha=0.45)
    p3 = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
    p3 = add_panel_label(p3, "(c) SAM Semantic Masks Overlay", f"Instance Segmentation ({len(masks)} masks)", badge_color=(180, 50, 220))

    # 4. 原图 + 语义掩膜 + DeepLSD 线段
    fusion_rgb = render_overlay_with_lines(masked_rgb, lines, line_color=(0, 255, 255), point_color=(0, 0, 255))
    p4 = cv2.cvtColor(fusion_rgb, cv2.COLOR_RGB2BGR)
    p4 = add_panel_label(p4, "(d) Original + SAM Masks + LSD", f"Multi-Modal Fusion ({len(lines)} detected lines)", badge_color=(0, 220, 100))

    # 组装 2x2 布局
    row_top = np.concatenate([p1, p2], axis=1)
    row_bottom = np.concatenate([p3, p4], axis=1)
    grid_body = np.concatenate([row_top, row_bottom], axis=0)

    # 顶部专业标题栏
    gh, gw = grid_body.shape[:2]
    header_h = 60
    header = np.zeros((header_h, gw, 3), dtype=np.uint8)
    header[:] = (28, 28, 30)

    cv2.putText(
        header,
        f"Multi-Modal Feature Fusion Board | {scene_name}",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "DeepLSD (Geometric Distance Field & Lines) + SAM ViT-H (Semantic Instance Masks)",
        (gw - 660, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    grid_2x2 = np.concatenate([header, grid_body], axis=0)

    # 组装 1x4 横条布局
    strip_body = np.concatenate([p1, p2, p3, p4], axis=1)
    sh, sw = strip_body.shape[:2]
    strip_header = np.zeros((50, sw, 3), dtype=np.uint8)
    strip_header[:] = (28, 28, 30)
    cv2.putText(
        strip_header,
        f"Multi-Modal Visual Strip | {scene_name} [Original -> DeepLSD DF -> SAM Masks -> Fusion]",
        (25, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    strip_1x4 = np.concatenate([strip_header, strip_body], axis=0)

    panels = {
        "original": p1,
        "deeplsd_df": p2,
        "sam_masks": p3,
        "fusion": p4,
    }

    return grid_2x2, strip_1x4, panels


def main():
    parser = argparse.ArgumentParser(description="多模态几何与语义特征四合一组合可视化工具")
    parser.add_argument("--image-path", default=None, help="输入图片路径 (若未提供，则自动从数据集选取)")
    parser.add_argument("--npz-path", default=None, help="可选: 预先提取的 .npz 路径 (包含 df 与 lines)")
    parser.add_argument("--sample-name", default=None, help="样本名称 (如 pair_00000 或 scene0707_00_pair00)")
    parser.add_argument("--dataset-dir", default="data/scannet_semantic_gluestick", help="数据集根目录")
    parser.add_argument("--out-dir", default="outputs/multimodal_combo", help="合成大图输出目录")
    parser.add_argument("--sam-checkpoint", default="weights/sam/sam_vit_h_4b8939.pth", help="SAM ViT-H 权重路径")
    parser.add_argument("--deeplsd-checkpoint", default="third_party/DeepLSD/weights/deeplsd_wireframe.tar", help="DeepLSD 权重")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--batch-samples", nargs="+", default=None, help="批量运行的样本名列表")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir).resolve()

    # 1. 确定待处理的样本列表
    samples_to_process = []
    if args.batch_samples:
        samples_to_process = args.batch_samples
    elif args.sample_name:
        samples_to_process = [args.sample_name]
    elif args.image_path:
        samples_to_process = [Path(args.image_path).stem]
    else:
        # 自动从 dataset_dir 中读取前 3 个样本
        cand_npzs = sorted(list(dataset_dir.glob("*.npz")) + list((dataset_dir / "npz").glob("*.npz")))
        if cand_npzs:
            samples_to_process = [p.stem for p in cand_npzs[:3]]
        else:
            samples_to_process = ["pair_00000"]

    print("=" * 70)
    print(f"[*] 启动多模态四合一可视化引擎: 待处理 {len(samples_to_process)} 个样本")
    print(f"[*] 导出目录: {out_dir}")
    print("=" * 70)

    # 2. 初始化核心引擎
    deeplsd = DeepLSDExtractor(checkpoint_path=args.deeplsd_checkpoint, device=args.device, detect_lines=True)
    samext = SAMExtractor(model_type="vit_h", checkpoint_path=args.sam_checkpoint, device=args.device)

    # 初始化 SAM 自动 Mask 生成器
    from segment_anything import SamAutomaticMaskGenerator
    mask_gen = SamAutomaticMaskGenerator(
        model=samext.sam,
        points_per_side=16,
        points_per_batch=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    for sample_name in samples_to_process:
        print(f"\n[*] 正在处理样本: {sample_name}...")

        # 寻找输入图像与已有 npz
        img_rgb = None
        df = None
        lines = None

        if args.image_path and Path(args.image_path).is_file():
            raw_bgr = cv2.imread(args.image_path)
            if raw_bgr is not None:
                img_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
                if img_rgb.shape[:2] != (480, 640):
                    img_rgb = cv2.resize(img_rgb, (640, 480), interpolation=cv2.INTER_AREA)

        if img_rgb is None:
            # 从数据集候选目录中查找
            cand_imgs = [
                dataset_dir / "images" / f"{sample_name}_0.png",
                dataset_dir / "images" / f"{sample_name}_0.jpg",
                dataset_dir / f"{sample_name}_0.png",
            ]
            for p in cand_imgs:
                if p.is_file():
                    bgr = cv2.imread(str(p))
                    if bgr is not None:
                        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        break

        # 若仍然未找到图像，尝试从 ScanNet 原始数据或 npz 读取
        npz_file = dataset_dir / "npz" / f"{sample_name}.npz"
        if not npz_file.is_file():
            npz_file = dataset_dir / f"{sample_name}.npz"

        if npz_file.is_file():
            npz_data = np.load(str(npz_file), allow_pickle=True)
            if "df0" in npz_data:
                df = npz_data["df0"]
            if "lines0" in npz_data:
                lines = npz_data["lines0"]
            if img_rgb is None and "image0" in npz_data:
                g = npz_data["image0"]
                img_rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

        if img_rgb is None:
            print(f"[!] 错误: 未能找到样本 {sample_name} 的图像，跳过。")
            continue

        # 若未预存 df 或 lines，则现场使用 DeepLSD 提取
        if df is None or lines is None:
            print("  [-] 运行 DeepLSD 提取亚像素几何线段与距离场...")
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            dlsd_out = deeplsd.detect(gray)
            df = dlsd_out["df"]
            lines = dlsd_out["lines"]
        else:
            print(f"  [✓] 已载入预提取几何特征: {len(lines)} 条线段, 距离场尺寸 {df.shape}")

        # 运行 SAM 生成语义掩膜
        print("  [-] 运行 SAM ViT-H 生成语义实例掩膜...")
        with torch.amp.autocast("cuda", enabled=samext.use_fp16):
            masks = mask_gen.generate(img_rgb)
        print(f"  [✓] SAM 生成 {len(masks)} 个语义实例掩膜")

        # 拼装 2x2 宫格与 1x4 横条
        grid_2x2, strip_1x4, panels = build_combo_visualizations(
            img_rgb=img_rgb,
            df=df,
            masks=masks,
            lines=lines,
            scene_name=sample_name,
        )

        # 保存结果
        grid_out = out_dir / f"{sample_name}_combo_2x2.jpg"
        strip_out = out_dir / f"{sample_name}_combo_strip.jpg"
        cv2.imwrite(str(grid_out), grid_2x2)
        cv2.imwrite(str(strip_out), strip_1x4)

        # 保存 4 个单独子图方便单独使用
        sample_sub_dir = out_dir / sample_name
        sample_sub_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_sub_dir / "1_original.jpg"), panels["original"])
        cv2.imwrite(str(sample_sub_dir / "2_deeplsd_df.jpg"), panels["deeplsd_df"])
        cv2.imwrite(str(sample_sub_dir / "3_sam_masks.jpg"), panels["sam_masks"])
        cv2.imwrite(str(sample_sub_dir / "4_fusion_all.jpg"), panels["fusion"])

        print(f"  [✓] 已生成四合一 2x2 看板: {grid_out}")
        print(f"  [✓] 已生成 1x4 横向排布图: {strip_out}")
        print(f"  [✓] 独立子图已保存至: {sample_sub_dir}")

    print("\n[✓] 所有多模态四合一综合可视化生成完毕！")


if __name__ == "__main__":
    main()
