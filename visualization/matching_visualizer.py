"""
High-Performance Matching and Diagnosis Visualization Module.
Refactored and unified visualizers for:
1. Dual-row Semantic Region Mask & Centroid Correspondence with TP/FP/FN Error Diagnostics.
2. Side-by-side Object Bipartite Matcher Canvas ("连连看").
3. Sub-pixel Straight Line Correspondence Canvas.
"""

from __future__ import annotations

import math
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


def _as_numpy(value: Any) -> np.ndarray | None:
    """Detach tensors and drop a leading batch dimension of size 1."""
    if value is None:
        return None
    if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] == 1 and arr.ndim >= 3:
        arr = arr[0]
    return arr


def _as_bbox_list(bboxes: Any) -> list[Sequence[float]]:
    """Normalize (M, 4) / (1, M, 4) boxes or a list of xyxy boxes."""
    if bboxes is None:
        return []
    if isinstance(bboxes, (list, tuple)) and (not bboxes or isinstance(bboxes[0], (list, tuple, np.ndarray))):
        return [list(map(float, b[:4])) for b in bboxes]
    arr = _as_numpy(bboxes)
    if arr is None or arr.size == 0:
        return []
    if arr.ndim == 1 and arr.shape[0] >= 4:
        return [arr[:4].astype(float).tolist()]
    if arr.ndim >= 2:
        if arr.shape[-1] < 4:
            return []
        flat = arr.reshape(-1, arr.shape[-1])
        return [row[:4].astype(float).tolist() for row in flat]
    return []


def line5d_to_patch_endpoints(
    feat: Sequence[float],
    patch_size: int = 112,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Invert [cu, cv, L, sin2θ, cos2θ] into undirected endpoints on the square patch."""
    cu, cv, length_norm, sin2, cos2 = (float(v) for v in list(feat)[:5])
    if length_norm <= 1e-6:
        return None
    theta = 0.5 * math.atan2(sin2, cos2)
    cx = cu * float(patch_size)
    cy = cv * float(patch_size)
    half = 0.5 * length_norm * float(patch_size)
    dx = half * math.cos(theta)
    dy = half * math.sin(theta)
    return (cx - dx, cy - dy), (cx + dx, cy + dy)


def patch_point_to_image(
    u: float,
    v: float,
    bbox: Sequence[float],
    patch_size: int = 112,
) -> tuple[float, float]:
    """Map a patch-square pixel back through the same letterbox as extract_region_patch_from_crop."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    w_c = max(x2 - x1, 1.0)
    h_c = max(y2 - y1, 1.0)
    scale = float(patch_size) / max(h_c, w_c)
    new_w = max(1, min(patch_size, int(round(w_c * scale))))
    new_h = max(1, min(patch_size, int(round(h_c * scale))))
    pad_x = (patch_size - new_w) // 2
    pad_y = (patch_size - new_h) // 2
    local_x = (float(u) - pad_x) / scale
    local_y = (float(v) - pad_y) / scale
    return x1 + local_x, y1 + local_y


def segment_from_line5d(
    feat: Sequence[float],
    bbox: Sequence[float],
    patch_size: int = 112,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Recover one image-space segment from a 5D patch line and its parent region box."""
    ends = line5d_to_patch_endpoints(feat, patch_size=patch_size)
    if ends is None:
        return None
    p0 = patch_point_to_image(ends[0][0], ends[0][1], bbox, patch_size=patch_size)
    p1 = patch_point_to_image(ends[1][0], ends[1][1], bbox, patch_size=patch_size)
    return p0, p1


def collect_line_segments(
    patch_lines: Any,
    line_mask: Any,
    bboxes: Any,
    patch_size: int = 112,
) -> list[tuple[tuple[float, float], tuple[float, float]] | None]:
    """
    Flatten valid (region, slot) lines in the same order as the line Sinkhorn.

    Invalid or zero-length slots are omitted. Returned index i is the flat slot.
    """
    lines = _as_numpy(patch_lines)
    if lines is None or lines.size == 0:
        return []
    if lines.ndim == 2 and lines.shape[-1] >= 5:
        lines = lines.reshape(1, lines.shape[0], lines.shape[-1])
    if lines.ndim != 3 or lines.shape[-1] < 5:
        return []
    regions, slots, _ = lines.shape
    mask = _as_numpy(line_mask)
    if mask is None:
        mask = np.ones((regions, slots), dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)
        if mask.ndim == 1:
            mask = mask.reshape(regions, slots) if mask.size == regions * slots else np.ones((regions, slots), dtype=bool)
        elif mask.shape[:2] != (regions, slots):
            mask = np.ones((regions, slots), dtype=bool)
    boxes = _as_bbox_list(bboxes)
    segments: list[tuple[tuple[float, float], tuple[float, float]] | None] = []
    for region_idx in range(regions):
        if region_idx >= len(boxes):
            break
        for slot in range(slots):
            if not bool(mask[region_idx, slot]):
                continue
            segments.append(segment_from_line5d(lines[region_idx, slot], boxes[region_idx], patch_size=patch_size))
    return segments


def _segment_midpoint(seg: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, float]:
    return ((seg[0][0] + seg[1][0]) * 0.5, (seg[0][1] + seg[1][1]) * 0.5)


def _canvas_point(pt: tuple[float, float], x_off: int, y_off: int) -> tuple[int, int]:
    return int(round(pt[0] + x_off)), int(round(pt[1] + y_off))


def _draw_segment(
    canvas: np.ndarray,
    seg: tuple[tuple[float, float], tuple[float, float]] | None,
    x_off: int,
    y_off: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    dashed: bool = False,
) -> tuple[int, int] | None:
    if seg is None:
        return None
    p0 = _canvas_point(seg[0], x_off, y_off)
    p1 = _canvas_point(seg[1], x_off, y_off)
    if dashed:
        draw_dashed_line(canvas, p0, p1, color, thickness=max(1, thickness - 1), dash_length=7)
    else:
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
    mid = _canvas_point(_segment_midpoint(seg), x_off, y_off)
    cv2.circle(canvas, mid, 3, color, -1, cv2.LINE_AA)
    return mid


def visualize_line_matches(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    patch_lines0: Any,
    patch_lines1: Any,
    line_mask0: Any,
    line_mask1: Any,
    bboxes0: Any,
    bboxes1: Any,
    pred_matches: list[tuple[int, int]],
    gt_matches: list[tuple[int, int]],
    out_path: Path | str,
    title: str = "Line Matching",
    patch_size: int = 112,
    masks0: Any = None,
    masks1: Any = None,
    has_gt: bool = True,
) -> None:
    """
    Dual-row line correspondence canvas.

    Top: predicted line matches (TP distinct color, FP red, FN yellow dashed).
    Bottom: ground-truth line pairs in shared colors, or a note when CAD ids are absent.
    Region masks, when given, are drawn only as faint contours.
    """
    segs0 = collect_line_segments(patch_lines0, line_mask0, bboxes0, patch_size=patch_size)
    segs1 = collect_line_segments(patch_lines1, line_mask1, bboxes1, patch_size=patch_size)
    gt_pairs = [(int(i), int(j)) for i, j in gt_matches] if has_gt else []
    pred_pairs = [(int(i), int(j)) for i, j in pred_matches]

    H0, W0 = bgr0.shape[:2]
    H1, W1 = bgr1.shape[:2]
    H = max(H0, H1)
    header_h = 50
    gap = 30
    W_total = W0 + W1 + gap
    row1_y = header_h
    mid_header_y = row1_y + H
    row2_y = mid_header_y + header_h
    total_h = row2_y + H + 12
    canvas = np.full((total_h, W_total, 3), 26, dtype=np.uint8)

    RED = (40, 40, 235)
    YELLOW = (0, 215, 255)
    GRAY = (150, 150, 150)
    CONTOUR = (90, 70, 50)

    def _paint_base(image: np.ndarray, masks: Any) -> np.ndarray:
        overlay = image.copy()
        mask_arr = _to_numpy_mask(masks) if masks is not None else None
        if mask_arr is not None and mask_arr.ndim >= 2:
            if mask_arr.ndim == 2:
                mask_arr = mask_arr[None, ...]
            for idx in range(len(mask_arr)):
                draw_mask_boundary(overlay, mask_arr[idx], CONTOUR, thickness=1)
        return overlay

    def _paste_row(overlay0: np.ndarray, overlay1: np.ndarray, y0: int) -> None:
        canvas[y0 : y0 + H0, :W0] = overlay0
        canvas[y0 : y0 + H1, W0 + gap :] = overlay1

    def _draw_all_gray(segs: list, x_off: int, y_off: int) -> None:
        for seg in segs:
            _draw_segment(canvas, seg, x_off, y_off, GRAY, thickness=1)

    gt_set = set(gt_pairs)
    pred_set = set(pred_pairs)
    tp_set = pred_set & gt_set
    fp_set = pred_set - gt_set
    fn_set = gt_set - pred_set

    base0 = _paint_base(bgr0, masks0)
    base1 = _paint_base(bgr1, masks1)
    _paste_row(base0, base1, row1_y)
    _draw_all_gray(segs0, 0, row1_y)
    _draw_all_gray(segs1, W0 + gap, row1_y)

    def _pair_ok(i: int, j: int) -> bool:
        return 0 <= i < len(segs0) and 0 <= j < len(segs1) and segs0[i] is not None and segs1[j] is not None

    for color_idx, (i, j) in enumerate(sorted(tp_set)):
        if not _pair_ok(i, j):
            continue
        color = DISTINCT_COLORS[color_idx % len(DISTINCT_COLORS)]
        mid0 = _draw_segment(canvas, segs0[i], 0, row1_y, color, thickness=2)
        mid1 = _draw_segment(canvas, segs1[j], W0 + gap, row1_y, color, thickness=2)
        if mid0 is not None and mid1 is not None:
            cv2.line(canvas, mid0, mid1, color, 1, cv2.LINE_AA)

    for i, j in fp_set:
        if not _pair_ok(i, j):
            continue
        mid0 = _draw_segment(canvas, segs0[i], 0, row1_y, RED, thickness=2)
        mid1 = _draw_segment(canvas, segs1[j], W0 + gap, row1_y, RED, thickness=2)
        if mid0 is not None and mid1 is not None:
            cv2.line(canvas, mid0, mid1, RED, 2, cv2.LINE_AA)

    for i, j in fn_set:
        if not _pair_ok(i, j):
            continue
        mid0 = _draw_segment(canvas, segs0[i], 0, row1_y, YELLOW, thickness=2, dashed=True)
        mid1 = _draw_segment(canvas, segs1[j], W0 + gap, row1_y, YELLOW, thickness=2, dashed=True)
        if mid0 is not None and mid1 is not None:
            draw_dashed_line(canvas, mid0, mid1, YELLOW, thickness=1, dash_length=8)

    n_tp, n_fp, n_fn = len(tp_set), len(fp_set), len(fn_set)
    cv2.rectangle(canvas, (0, 0), (W_total, header_h), (34, 34, 34), -1)
    cv2.putText(
        canvas,
        f"[TOP: LINE PREDICTION] {title}  TP={n_tp}  FP={n_fp} (red)  FN={n_fn} (yellow dashed)  lines={len(segs0)}/{len(segs1)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    _paste_row(base0, base1, row2_y)
    _draw_all_gray(segs0, 0, row2_y)
    _draw_all_gray(segs1, W0 + gap, row2_y)
    if has_gt:
        for color_idx, (i, j) in enumerate(gt_pairs):
            if not _pair_ok(i, j):
                continue
            color = DISTINCT_COLORS[color_idx % len(DISTINCT_COLORS)]
            mid0 = _draw_segment(canvas, segs0[i], 0, row2_y, color, thickness=2)
            mid1 = _draw_segment(canvas, segs1[j], W0 + gap, row2_y, color, thickness=2)
            if mid0 is not None and mid1 is not None:
                cv2.line(canvas, mid0, mid1, color, 2, cv2.LINE_AA)
        gt_banner = f"[BOTTOM: LINE GROUND TRUTH] pairs={len(gt_pairs)}  same color = one correspondence"
    else:
        gt_banner = "[BOTTOM: LINE GROUND TRUTH] no CAD line ids; line GT disabled"
    cv2.rectangle(canvas, (0, mid_header_y), (W_total, mid_header_y + header_h), (34, 34, 34), -1)
    cv2.putText(
        canvas,
        gt_banner,
        (16, mid_header_y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

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


def _flag(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _line_probability(out: dict[str, Any]) -> Any:
    """Line Sinkhorn matrix, or None when the model has no line head."""
    p_line = out.get("P_line")
    if isinstance(p_line, (list, tuple)):
        return p_line[0] if p_line else None
    return p_line


def _line_ground_truth(batch: dict[str, Any]) -> tuple[list[tuple[int, int]], bool]:
    """Flattened line pairs. Empty when the sample has no CAD line ids."""
    has_ids = _flag(batch.get("has_line_ids", False))
    raw = batch.get("gt_line_pairs_flat", [])
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
        first = raw[0]
        if not first or not isinstance(first[0], (int, np.integer)):
            raw = first
    pairs: list[tuple[int, int]] = []
    if has_ids:
        for rec in raw or []:
            if isinstance(rec, (list, tuple)) and len(rec) >= 2 and not isinstance(rec[0], (list, tuple)):
                pairs.append((int(rec[0]), int(rec[1])))
    return pairs, has_ids


def _load_region_views(
    batch: dict[str, Any],
    data_dir: Path | str | None,
    scene_name: str | None,
) -> tuple[Any, Any, Any, Any]:
    """Images and masks from the batch, falling back to the pair directory on disk."""
    bgr0 = batch.get("raw_img0")
    bgr1 = batch.get("raw_img1")
    masks0 = batch.get("masks0")
    masks1 = batch.get("masks1")
    if (bgr0 is not None and masks0 is not None) or data_dir is None or not scene_name:
        return bgr0, bgr1, masks0, masks1

    p_dir = Path(data_dir) / str(scene_name)
    loaded0 = cv2.imread(str(p_dir / "image_0.png"))
    loaded1 = cv2.imread(str(p_dir / "image_1.png"))
    if loaded0 is not None:
        bgr0 = loaded0
    if loaded1 is not None:
        bgr1 = loaded1

    def _masks(regs: list, folder: str) -> np.ndarray:
        shape = bgr0.shape[:2] if bgr0 is not None else (480, 640)
        stacked = []
        for reg in regs:
            mask = cv2.imread(str(p_dir / folder / f"mask_{reg['idx']:02d}.png"), cv2.IMREAD_GRAYSCALE)
            stacked.append((mask > 127).astype(np.float32) if mask is not None else np.zeros(shape, dtype=np.float32))
        if not stacked:
            return np.zeros((0, shape[0], shape[1]), dtype=np.float32)
        return np.stack(stacked, axis=0)

    return bgr0, bgr1, _masks(batch.get("regs0", []), "masks_0"), _masks(batch.get("regs1", []), "masks_1")


def save_pair_visualization(
    batch: dict[str, Any],
    out: dict[str, Any],
    out_path: Path | str,
    *,
    title: str,
    pred_matches: list[tuple[int, int]],
    pred_un0: list[int],
    pred_un1: list[int],
    data_dir: Path | str | None = None,
    scene_name: str | None = None,
    use_region_ds: bool = True,
) -> bool:
    """
    Write one diagnostic canvas for a collated pair.

    A line head (P_line plus patch lines) draws line correspondences.
    Otherwise region masks or object boxes use the existing canvases.
    """
    from datasets.object_dataset import predict_bipartite_matches

    if use_region_ds:
        bgr0, bgr1, masks0, masks1 = _load_region_views(batch, data_dir, scene_name)
        if bgr0 is None or bgr1 is None:
            return False
        p_line = _line_probability(out)
        if p_line is not None and batch.get("patch_lines0") is not None and batch.get("patch_lines1") is not None:
            line_pred, _, _ = predict_bipartite_matches(p_line, thresh=0.2, min_area_ratio=0.0)
            gt_lines, has_line_gt = _line_ground_truth(batch)
            bboxes0 = batch.get("bboxes0")
            bboxes1 = batch.get("bboxes1")
            if bboxes0 is None:
                bboxes0 = [reg["bbox"] for reg in batch.get("regs0", [])]
            if bboxes1 is None:
                bboxes1 = [reg["bbox"] for reg in batch.get("regs1", [])]
            visualize_line_matches(
                bgr0=bgr0,
                bgr1=bgr1,
                patch_lines0=batch.get("patch_lines0"),
                patch_lines1=batch.get("patch_lines1"),
                line_mask0=batch.get("line_mask0"),
                line_mask1=batch.get("line_mask1"),
                bboxes0=bboxes0,
                bboxes1=bboxes1,
                pred_matches=line_pred,
                gt_matches=gt_lines,
                out_path=out_path,
                title=title,
                masks0=masks0,
                masks1=masks1,
                has_gt=has_line_gt,
            )
            return True
        if masks0 is None or masks1 is None:
            return False
        visualize_semantic_region_matches(
            bgr0=bgr0,
            bgr1=bgr1,
            masks0=masks0,
            masks1=masks1,
            regs0=batch.get("regs0", []),
            regs1=batch.get("regs1", []),
            pred_matches=pred_matches,
            gt_matches=batch["gt_matches"][0],
            pred_un0=pred_un0,
            pred_un1=pred_un1,
            out_path=out_path,
            title=title,
        )
        return True

    if batch.get("raw_img0") is None or batch.get("raw_img1") is None:
        return False
    visualize_object_matches(
        batch["raw_img0"],
        batch["raw_img1"],
        batch["objs0"],
        batch["objs1"],
        pred_matches,
        batch["gt_matches"][0],
        pred_un0,
        pred_un1,
        out_path,
        title=title,
    )
    return True


def _side_by_side(img0: np.ndarray, img1: np.ndarray, title: str) -> tuple[np.ndarray, int, int]:
    """Place two views side by side under a title bar. Returns canvas, y offset, view1 x offset."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    header, gap = 32, 12
    canvas = np.full((header + max(h0, h1), w0 + gap + w1, 3), 18, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header), (28, 32, 40), -1)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    canvas[header : header + h0, :w0] = img0
    canvas[header : header + h1, w0 + gap :] = img1
    return canvas, header, w0 + gap


def render_dataset_ground_truth(
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    regs0: list[dict[str, Any]],
    regs1: list[dict[str, Any]],
    masks0: np.ndarray,
    masks1: np.ndarray,
    lines0: list[dict[str, Any]],
    lines1: list[dict[str, Any]],
    correspondence: dict[str, Any],
    out_dir: Path | str,
    stem: str,
) -> dict[str, Path]:
    """
    Write separate Ground Truth panels: raw photos, region labels, and CAD line labels.

    Matched pairs share one color across the two views.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gray = (110, 110, 110)
    paths: dict[str, Path] = {}

    raw, _, _ = _side_by_side(bgr0, bgr1, "原图  View0  |  View1")
    raw_path = out / f"{stem}__raw.jpg"
    cv2.imwrite(str(raw_path), raw)
    paths["raw"] = raw_path

    region_matches = correspondence.get("matches") or []
    color0: dict[int, tuple[int, int, int]] = {}
    color1: dict[int, tuple[int, int, int]] = {}
    for idx, rec in enumerate(region_matches):
        color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
        color0[int(rec.get("mask0_idx", -1))] = color
        color1[int(rec.get("mask1_idx", -1))] = color

    def _paint_regions(image: np.ndarray, regs: list[dict[str, Any]], masks: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
        painted = image.copy()
        for i, reg in enumerate(regs):
            color = colors.get(i, gray)
            if i < len(masks):
                blend_mask(painted, masks[i], color, alpha=0.45)
                draw_mask_boundary(painted, masks[i], color, thickness=2)
            centroid = reg.get("centroid_2d") or [0, 0]
            label = str(reg.get("label") or reg.get("category") or f"r{i}")
            draw_text_with_pill(painted, label, (int(centroid[0]), int(centroid[1])), (255, 255, 255), (20, 20, 20))
        return painted

    region_canvas, y0, x1 = _side_by_side(
        _paint_regions(bgr0, regs0, masks0, color0),
        _paint_regions(bgr1, regs1, masks1, color1),
        f"区域标签 Ground Truth  同色 = 同一对  ({len(region_matches)} pairs)",
    )
    for rec in region_matches:
        i, j = int(rec.get("mask0_idx", -1)), int(rec.get("mask1_idx", -1))
        if i < 0 or j < 0 or i >= len(regs0) or j >= len(regs1):
            continue
        c0 = regs0[i].get("centroid_2d") or [0, 0]
        c1 = regs1[j].get("centroid_2d") or [0, 0]
        color = color0.get(i, gray)
        cv2.line(
            region_canvas,
            (int(c0[0]), int(c0[1]) + y0),
            (int(c1[0]) + x1, int(c1[1]) + y0),
            color,
            1,
            cv2.LINE_AA,
        )
    region_path = out / f"{stem}__regions.jpg"
    cv2.imwrite(str(region_path), region_canvas)
    paths["regions"] = region_path

    if lines0 or lines1:
        line_matches = correspondence.get("line_matches") or []
        lcolor0: dict[int, tuple[int, int, int]] = {}
        lcolor1: dict[int, tuple[int, int, int]] = {}
        for idx, rec in enumerate(line_matches):
            color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]
            lcolor0[int(rec.get("line0_idx", -1))] = color
            lcolor1[int(rec.get("line1_idx", -1))] = color

        unmatched = (255, 255, 255)

        def _paint_lines(image: np.ndarray, records: list[dict[str, Any]], colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
            painted = image.copy()
            for i, rec in enumerate(records):
                coords = rec.get("coords") or []
                if len(coords) < 4:
                    continue
                matched = i in colors
                color = colors[i] if matched else unmatched
                p0 = (int(round(coords[0])), int(round(coords[1])))
                p1 = (int(round(coords[2])), int(round(coords[3])))
                cv2.line(painted, p0, p1, (0, 0, 0), 4 if matched else 3, cv2.LINE_AA)
                cv2.line(painted, p0, p1, color, 2 if matched else 1, cv2.LINE_AA)
                if matched:
                    tag = f"{rec.get('semantic', 'line')}#{rec.get('line_id', i)}"
                    draw_text_with_pill(painted, tag, ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2), (255, 255, 255), (16, 16, 16), font_scale=0.35)
            return painted

        line_canvas, ly, lx = _side_by_side(
            _paint_lines(bgr0, lines0, lcolor0),
            _paint_lines(bgr1, lines1, lcolor1),
            f"直线 Ground Truth  白线=本视角全部CAD线 {len(lines0)}|{len(lines1)}  彩色=跨视角同一 line_id 仅 {len(line_matches)} 对",
        )
        for rec in line_matches:
            i, j = int(rec.get("line0_idx", -1)), int(rec.get("line1_idx", -1))
            if not (0 <= i < len(lines0) and 0 <= j < len(lines1)):
                continue
            c0, c1 = lines0[i].get("coords") or [], lines1[j].get("coords") or []
            if len(c0) < 4 or len(c1) < 4:
                continue
            color = lcolor0.get(i, gray)
            cv2.line(
                line_canvas,
                (int((c0[0] + c0[2]) / 2), int((c0[1] + c0[3]) / 2) + ly),
                (int((c1[0] + c1[2]) / 2) + lx, int((c1[1] + c1[3]) / 2) + ly),
                color,
                1,
                cv2.LINE_AA,
            )
        line_path = out / f"{stem}__lines.jpg"
        cv2.imwrite(str(line_path), line_canvas)
        paths["lines"] = line_path
    return paths
