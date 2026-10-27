"""
High-Performance Matching and Diagnosis Visualization Module.
Refactored and unified visualizers for:
1. Dual-row Semantic Region Mask & Centroid Correspondence with TP/FP/FN Error Diagnostics.
2. Side-by-side Object Bipartite Matcher Canvas ("连连看").
3. Sub-pixel Straight Line Correspondence Canvas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# High-contrast distinctive color palette for GT matching pairs
DISTINCT_COLORS: list[tuple[int, int, int]] = [
    (0, 215, 255),    # Bright Gold (BGR)
    (255, 144, 30),   # Electric Dodger Blue
    (50, 205, 50),    # Lime Green
    (238, 130, 238),  # Violet
    (0, 165, 255),    # Bright Orange
    (208, 224, 64),   # Cyan Turquoise
    (180, 105, 255),  # Hot Pink
    (0, 255, 127),    # Spring Green
    (255, 191, 0),    # Deep Sky Blue
    (147, 20, 255),   # Bright Deep Pink
    (128, 255, 0),    # Bright Chartreuse
    (255, 0, 255),    # Magenta
    (0, 250, 154),    # Medium Spring Green
    (240, 140, 180),  # Soft Violet
    (80, 180, 240),   # Warm Amber
]


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.50) -> np.ndarray:
    """Vectorized alpha-blending of a binary mask over a 3-channel image."""
    m_bool = mask > 0.5
    if not np.any(m_bool):
        return image
    color_arr = np.array(color, dtype=np.float32)
    image[m_bool] = np.clip(image[m_bool] * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)
    return image


def draw_mask_boundary(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draws smooth anti-aliased contour boundary around a binary mask."""
    m_u8 = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def draw_dashed_line(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 8,
) -> None:
    """Draws smooth dashed line connecting two coordinates."""
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 1e-3:
        return
    steps = max(int(dist / dash_length), 2)
    for s in range(0, steps, 2):
        p_a = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (s / steps)),
            int(pt1[1] + (pt2[1] - pt1[1]) * (s / steps)),
        )
        p_b = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (min(s + 1, steps) / steps)),
            int(pt1[1] + (pt2[1] - pt1[1]) * (min(s + 1, steps) / steps)),
        )
        cv2.line(canvas, p_a, p_b, color, thickness, cv2.LINE_AA)


def draw_text_with_pill(
    canvas: np.ndarray,
    text: str,
    pos: tuple[int, int],
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (20, 20, 20),
    font_scale: float = 0.42,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draws text with a rounded dark background pill for crystal-clear legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = min(canvas.shape[1], x + tw + padding)
    y2 = min(canvas.shape[0], y + baseline + padding)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _to_numpy_mask(mask: Any) -> np.ndarray:
    """Safely converts a PyTorch Tensor or NumPy array to a NumPy mask."""
    if TORCH_AVAILABLE and isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    elif hasattr(mask, "detach"):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def visualize_semantic_region_matches(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    masks0: Any,
    masks1: Any,
    regs0: list[dict[str, Any]],
    regs1: list[dict[str, Any]],
    pred_matches: list[tuple[int, int]],
    gt_matches: list[tuple[int, int]],
    pred_un0: list[int],
    pred_un1: list[int],
    out_path: Path | str,
    title: str = "Semantic Region Mask Matching",
    connect_tp: bool = False,
) -> None:
    """
    Renders a unified dual-pair comparison canvas containing TWO PAIRS of images stacked vertically:
      - Top Pair (Row 1): Model Prediction & Error Diagnostics
          * Green mask: correctly matched regions (TP).
          * Red mask & red lines: incorrectly matched regions (FP).
          * Yellow dashed lines: missed ground truth target correspondences (FN).
          * Blue lines: region boundaries across all semantic segments.
      - Bottom Pair (Row 2): Ground Truth Target Matching
          * Matching regions highlighted with identical distinct colors per GT pair.
          * Solid colored lines: connect corresponding centroids across views.
          * Blue lines: region boundaries across all semantic segments.
    """
    m0_arr = _to_numpy_mask(masks0)
    m1_arr = _to_numpy_mask(masks1)

    H0, W0 = bgr0.shape[:2]
    H1, W1 = bgr1.shape[:2]
    H = max(H0, H1)
    header_h = 50
    gap = 30
    W_total = W0 + W1 + gap

    # Layout geometry: Top row (Model Pred) + Middle banner + Bottom row (Ground Truth)
    row1_y = header_h
    mid_header_y = row1_y + H
    row2_y = mid_header_y + header_h
    total_h = row2_y + H + 12

    canvas = np.full((total_h, W_total, 3), 26, dtype=np.uint8)

    # Palette
    GREEN = (40, 200, 40)
    RED = (40, 40, 235)
    YELLOW = (0, 215, 255)
    BLUE = (255, 140, 0)  # BGR

    gt_set = set(gt_matches)
    pred_set = set(pred_matches)
    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    # =========================================================================
    # ROW 1: MODEL PREDICTION & DIAGNOSIS (上层：模型预测与诊断)
    # =========================================================================
    overlay0_pred = bgr0.copy()
    overlay1_pred = bgr1.copy()

    # 1. 区域之间使用蓝色的线划分出来 (Blue boundary lines for all regions)
    for i in range(len(m0_arr)):
        draw_mask_boundary(overlay0_pred, m0_arr[i], BLUE, thickness=2)
    for j in range(len(m1_arr)):
        draw_mask_boundary(overlay1_pred, m1_arr[j], BLUE, thickness=2)

    # 2. 预测正确的染成绿色 (TP -> Green)
    for i, j in tp_set:
        if i < len(m0_arr):
            blend_mask(overlay0_pred, m0_arr[i], GREEN, alpha=0.55)
        if j < len(m1_arr):
            blend_mask(overlay1_pred, m1_arr[j], GREEN, alpha=0.55)

    # 3. 预测错误的染成红色 (FP -> Red)
    for i, j in fp_set:
        if i < len(m0_arr):
            blend_mask(overlay0_pred, m0_arr[i], RED, alpha=0.60)
        if j < len(m1_arr):
            blend_mask(overlay1_pred, m1_arr[j], RED, alpha=0.60)

    canvas[row1_y : row1_y + H0, :W0] = overlay0_pred
    canvas[row1_y : row1_y + H1, W0 + gap :] = overlay1_pred

    # 4. 预测错误连红线 (FP centroid error vectors)
    for i, j in fp_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row1_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row1_y))

        cv2.circle(canvas, pt0, 6, RED, -1)
        cv2.circle(canvas, pt1, 6, RED, -1)
        cv2.line(canvas, pt0, pt1, RED, 2, cv2.LINE_AA)

        lbl0 = regs0[i].get("label", regs0[i].get("category", f"obj_{i}")) if i < len(regs0) else ""
        lbl1 = regs1[j].get("label", regs1[j].get("category", f"obj_{j}")) if j < len(regs1) else ""
        draw_text_with_pill(canvas, f"FP: {lbl0}", (pt0[0] + 5, pt0[1] - 8), RED, bg_color=(20, 20, 20))
        draw_text_with_pill(canvas, f"FP: {lbl1}", (pt1[0] + 5, pt1[1] - 8), RED, bg_color=(20, 20, 20))

    # 5. 漏配真值连黄色虚线 (FN yellow dashed line)
    for i, j in fn_set:
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row1_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row1_y))
        draw_dashed_line(canvas, pt0, pt1, YELLOW, thickness=1, dash_length=8)

    # 6. 可选连接 TP 匹配线
    if connect_tp:
        for i, j in tp_set:
            c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
            c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
            pt0 = (int(c0[0]), int(c0[1] + row1_y))
            pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row1_y))
            cv2.line(canvas, pt0, pt1, GREEN, 1, cv2.LINE_AA)

    # Row 1 Banner
    n_tp, n_fp, n_fn = len(tp_set), len(fp_set), len(fn_set)
    pair_prec = n_tp / (n_tp + n_fp) * 100.0 if (n_tp + n_fp) > 0 else 0.0
    pair_rec = n_tp / (n_tp + n_fn) * 100.0 if (n_tp + n_fn) > 0 else 0.0
    status_str = f"TP={n_tp} (Green Mask) | FP={n_fp} (Red Mask & Red Line) | FN={n_fn} (Yellow Dashed) | Prec={pair_prec:.1f}% | Rec={pair_rec:.1f}%"
    cv2.rectangle(canvas, (0, 0), (W_total, header_h), (34, 34, 34), -1)
    cv2.putText(canvas, f"[TOP: MODEL PREDICTION & DIAGNOSIS] {title} [{status_str}]", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    # =========================================================================
    # ROW 2: GROUND TRUTH TARGET CORRESPONDENCES (下层：真值匹配对)
    # =========================================================================
    overlay0_gt = bgr0.copy()
    overlay1_gt = bgr1.copy()

    # 1. 对应区域染上相同的高对比度辨识色 (Identical distinct color per GT pair)
    for m_idx, (i, j) in enumerate(gt_matches):
        col = DISTINCT_COLORS[m_idx % len(DISTINCT_COLORS)]
        if i < len(m0_arr):
            blend_mask(overlay0_gt, m0_arr[i], col, alpha=0.55)
        if j < len(m1_arr):
            blend_mask(overlay1_gt, m1_arr[j], col, alpha=0.55)

    # 2. 区域边界轮廓线
    for i in range(len(m0_arr)):
        draw_mask_boundary(overlay0_gt, m0_arr[i], BLUE, thickness=2)
    for j in range(len(m1_arr)):
        draw_mask_boundary(overlay1_gt, m1_arr[j], BLUE, thickness=2)

    canvas[row2_y : row2_y + H0, :W0] = overlay0_gt
    canvas[row2_y : row2_y + H1, W0 + gap :] = overlay1_gt

    # 3. 绘制真值连接线与目标名称 (Solid matching lines between centroids)
    matched_0_set = {i for i, _ in gt_matches}
    matched_1_set = {j for _, j in gt_matches}
    un_0_count = len(regs0) - len(matched_0_set)
    un_1_count = len(regs1) - len(matched_1_set)

    for m_idx, (i, j) in enumerate(gt_matches):
        col = DISTINCT_COLORS[m_idx % len(DISTINCT_COLORS)]
        c0 = regs0[i]["centroid_2d"] if i < len(regs0) else [W0 // 2, H0 // 2]
        c1 = regs1[j]["centroid_2d"] if j < len(regs1) else [W1 // 2, H1 // 2]
        pt0 = (int(c0[0]), int(c0[1] + row2_y))
        pt1 = (int(c1[0] + W0 + gap), int(c1[1] + row2_y))

        cv2.circle(canvas, pt0, 5, col, -1)
        cv2.circle(canvas, pt1, 5, col, -1)
        cv2.line(canvas, pt0, pt1, col, 2, cv2.LINE_AA)

        lbl = regs0[i].get("label", regs0[i].get("category", f"match_{m_idx}")) if i < len(regs0) else ""
        draw_text_with_pill(canvas, lbl, (pt0[0] + 5, pt0[1] - 6), (255, 255, 255), (20, 20, 20))

    # Row 2 Middle Banner
    gt_banner = f"[BOTTOM: GROUND TRUTH CORRESPONDENCES] Target Matches: {len(gt_matches)} pairs | Unmatched: View0={un_0_count}, View1={un_1_count} | Matching regions share identical color & connecting lines"
    cv2.rectangle(canvas, (0, mid_header_y), (W_total, mid_header_y + header_h), (34, 34, 34), -1)
    cv2.putText(canvas, gt_banner, (20, mid_header_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas)


def visualize_object_matches(
    raw_img0: np.ndarray,
    raw_img1: np.ndarray,
    objs0: list[dict[str, Any]],
    objs1: list[dict[str, Any]],
    pred_matches: list[tuple[int, int]],
    gt_matches: list[tuple[int, int]],
    pred_un0: list[int],
    pred_un1: list[int],
    out_path: Path | str,
    title: str = "Object Bipartite Matching",
) -> None:
    """Draws side-by-side '连连看' visualization with green match lines and bounding boxes."""
    H0, W0, _ = raw_img0.shape
    H1, W1, _ = raw_img1.shape
    canvas_h = max(H0, H1) + 60
    canvas_w = W0 + W1 + 40
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

    y_offset = 50
    x0_offset = 10
    x1_offset = W0 + 30

    canvas[y_offset : y_offset + H0, x0_offset : x0_offset + W0] = raw_img0
    canvas[y_offset : y_offset + H1, x1_offset : x1_offset + W1] = raw_img1

    cv2.putText(canvas, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)

    gt_set = set(gt_matches)

    # Unmatched objects
    for i in pred_un0:
        if i < len(objs0):
            bx = objs0[i]["bbox"]
            p1 = (int(bx[0] + x0_offset), int(bx[1] + y_offset))
            p2 = (int(bx[2] + x0_offset), int(bx[3] + y_offset))
            cv2.rectangle(canvas, p1, p2, (40, 40, 230), 2)
            cv2.putText(canvas, f"{objs0[i]['category']} (unmatched)", (p1[0], max(p1[1] - 6, y_offset + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    for j in pred_un1:
        if j < len(objs1):
            bx = objs1[j]["bbox"]
            p1 = (int(bx[0] + x1_offset), int(bx[1] + y_offset))
            p2 = (int(bx[2] + x1_offset), int(bx[3] + y_offset))
            cv2.rectangle(canvas, p1, p2, (40, 40, 230), 2)
            cv2.putText(canvas, f"{objs1[j]['category']} (unmatched)", (p1[0], max(p1[1] - 6, y_offset + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    # Matched pairs
    for i, j in pred_matches:
        if i >= len(objs0) or j >= len(objs1):
            continue
        b0 = objs0[i]["bbox"]
        b1 = objs1[j]["bbox"]
        c0 = objs0[i].get("centroid", [(b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0])
        c1 = objs1[j].get("centroid", [(b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0])

        is_correct = (i, j) in gt_set if gt_set else True
        color = (40, 190, 40) if is_correct else (200, 30, 220)

        p0_min = (int(b0[0] + x0_offset), int(b0[1] + y_offset))
        p0_max = (int(b0[2] + x0_offset), int(b0[3] + y_offset))
        cv2.rectangle(canvas, p0_min, p0_max, color, 2)
        cv2.putText(canvas, objs0[i]["category"], (p0_min[0], max(p0_min[1] - 6, y_offset + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        p1_min = (int(b1[0] + x1_offset), int(b1[1] + y_offset))
        p1_max = (int(b1[2] + x1_offset), int(b1[3] + y_offset))
        cv2.rectangle(canvas, p1_min, p1_max, color, 2)
        cv2.putText(canvas, objs1[j]["category"], (p1_min[0], max(p1_min[1] - 6, y_offset + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        pt_from = (int(c0[0] + x0_offset), int(c0[1] + y_offset))
        pt_to = (int(c1[0] + x1_offset), int(c1[1] + y_offset))
        cv2.line(canvas, pt_from, pt_to, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_from, 5, color, -1)
        cv2.circle(canvas, pt_to, 5, color, -1)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), canvas)
