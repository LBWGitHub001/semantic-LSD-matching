"""SuperPoint 评估与诊断可视化套件 (Evaluation & Diagnostic Visualizer for SuperPoint).

包含:
1. match_accuracy: 最近邻特征匹配准确率与匹配索引。
2. match_accuracy_mma: 多阈值平均匹配精度 (MMA @ 1px, 2px, 3px)。
3. calculate_corner_repeatability: 跨视角角点重复率 (Repeatability) 与亚像素重定位误差。
4. extract_detected_lines: 基于对数距离场与角度场的直线拓扑提取与孤立端点识别。
5. save_point_match_figure: 经典绿红线特征匹配图 (向后兼容)。
6. save_superpoint_diagnostic_sheet: 四格全景诊断大图 (热力图辉光 + 直线拓扑 + 跨视角匹配 + 指标横幅)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def match_accuracy(
    descriptor0: torch.Tensor,
    descriptor1: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Nearest-neighbor accuracy when points0[i] is the true match of points1[i]."""
    if points0.shape[0] == 0 or points1.shape[0] == 0:
        return 0.0, torch.zeros(0, dtype=torch.long, device=points0.device)
    sampled0 = _sample_descriptors(descriptor0, points0)
    sampled1 = _sample_descriptors(descriptor1, points1)
    choice = torch.matmul(sampled0, sampled1.transpose(0, 1)).argmax(dim=1)
    target = torch.arange(choice.shape[0], device=choice.device)
    accuracy = float((choice == target).float().mean())
    return accuracy, choice


def match_accuracy_mma(
    descriptor0: torch.Tensor,
    descriptor1: torch.Tensor,
    points0: torch.Tensor,
    points1: torch.Tensor,
    homography: torch.Tensor | np.ndarray,
    thresholds: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict[str, float]:
    """计算多阈值平均匹配精度 (Mean Matching Accuracy @ 1px, 2px, 3px)."""
    if points0.shape[0] == 0 or points1.shape[0] == 0:
        return {f"mma_{t:.1f}px": 0.0 for t in thresholds} | {"nn_acc": 0.0}

    sampled0 = _sample_descriptors(descriptor0, points0)
    sampled1 = _sample_descriptors(descriptor1, points1)
    sim = torch.matmul(sampled0, sampled1.transpose(0, 1))
    choice = sim.argmax(dim=1)

    # 1. 对应同名点匹配准确率 (若 points0 与 points1 按行对齐)
    target = torch.arange(choice.shape[0], device=choice.device)
    nn_acc = float((choice == target).float().mean())

    # 2. 单应重投影物理几何误差
    p0_np = points0.detach().cpu().numpy()
    p1_np = points1.detach().cpu().numpy()
    h_np = homography.detach().cpu().numpy() if isinstance(homography, torch.Tensor) else np.asarray(homography)
    if h_np.ndim == 3:
        h_np = h_np[0]

    # 将 points0 投影到 View 1
    ones = np.ones((len(p0_np), 1), dtype=np.float32)
    p0_homo = np.hstack([p0_np, ones])
    warped = (h_np @ p0_homo.T).T
    w = np.clip(warped[:, 2:3], 1e-6, None)
    warped_p0 = warped[:, :2] / w

    # 获取匹配选中的 points1 坐标
    choice_np = choice.detach().cpu().numpy()
    chosen_p1 = p1_np[choice_np]

    reproj_errors = np.linalg.norm(warped_p0 - chosen_p1, axis=-1)

    results = {"nn_acc": nn_acc}
    for t in thresholds:
        results[f"mma_{t:.1f}px"] = float((reproj_errors <= t).mean())
    return results


def calculate_corner_repeatability(
    points0: torch.Tensor | np.ndarray,
    points1: torch.Tensor | np.ndarray,
    homography: torch.Tensor | np.ndarray,
    img_h: int,
    img_w: int,
    distance_thresh: float = 3.0,
) -> tuple[float, float]:
    """计算跨视角角点重复率 (Corner Repeatability) 与平均重定位误差 (px).

    公式:
        Repeatability = 成功重现的角点数 / 在有效视野内的角点总数
    """
    p0 = points0.detach().cpu().numpy() if isinstance(points0, torch.Tensor) else np.asarray(points0)
    p1 = points1.detach().cpu().numpy() if isinstance(points1, torch.Tensor) else np.asarray(points1)
    h_mat = homography.detach().cpu().numpy() if isinstance(homography, torch.Tensor) else np.asarray(homography)
    if h_mat.ndim == 3:
        h_mat = h_mat[0]

    if len(p0) == 0 or len(p1) == 0:
        return 0.0, 0.0

    # 投影 p0 到 View 1
    ones = np.ones((len(p0), 1), dtype=np.float32)
    p0_homo = np.hstack([p0, ones])
    warped = (h_mat @ p0_homo.T).T
    w = np.clip(warped[:, 2:3], 1e-6, None)
    warped_p0 = warped[:, :2] / w

    # 过滤出落在 View 1 视场内的点
    valid_mask = (
        (warped_p0[:, 0] >= 0)
        & (warped_p0[:, 0] < img_w)
        & (warped_p0[:, 1] >= 0)
        & (warped_p0[:, 1] < img_h)
    )
    if not valid_mask.any():
        return 0.0, 0.0

    valid_warped = warped_p0[valid_mask]
    # 计算与 View 1 中所有检测点的两两距离
    dists = np.linalg.norm(valid_warped[:, None, :] - p1[None, :, :], axis=-1)
    min_dists = dists.min(axis=1)

    hits = min_dists <= distance_thresh
    repeatability = float(hits.mean())
    mean_err = float(min_dists[hits].mean()) if hits.any() else 0.0
    return repeatability, mean_err


def extract_detected_lines(
    heatmap: torch.Tensor,
    df_map: torch.Tensor,
    af_map: torch.Tensor,
    loss_module: Any,
    score_thresh: float = 0.05,
    line_thresh: float = 0.50,
) -> dict[str, Any]:
    """基于对数距离场与角度场提取合法直线拓扑，并识别孤立端点。"""
    device = heatmap.device
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
    if df_map.dim() == 3:
        df_map = df_map.unsqueeze(0)
    if af_map.dim() == 3:
        af_map = af_map.unsqueeze(0)
    if df_map.shape[1] == 5:
        df_map = df_map[:, 3:4]
    if af_map.shape[1] == 5:
        af_map = af_map[:, 4:5]

    if loss_module is None or not hasattr(loss_module, "extract_candidate_endpoints"):
        from models.superpoint.loss import SuperPointDetectorLoss
        loss_module = SuperPointDetectorLoss()

    with torch.no_grad():
        scores, points = loss_module.extract_candidate_endpoints(heatmap)
        s_df, s_af, dist, diff, p1, p2 = loss_module.interpolate_and_sample_lines(points, df_map, af_map)
        v_matrix = loss_module.compute_line_validity_matrix(s_df, s_af, dist)

    sc = scores[0].detach().cpu().numpy()
    pts = points[0].detach().cpu().numpy()
    v_mat = v_matrix[0].detach().cpu().numpy()
    d_mat = dist[0].detach().cpu().numpy()
    df_mean = s_df[0].mean(dim=-1).detach().cpu().numpy()

    num_pts = len(pts)
    lines: list[dict[str, Any]] = []
    has_line = np.zeros(num_pts, dtype=bool)

    for i in range(num_pts):
        for j in range(i + 1, num_pts):
            if v_mat[i, j] >= line_thresh and sc[i] >= score_thresh and sc[j] >= score_thresh:
                has_line[i] = True
                has_line[j] = True
                lines.append({
                    "p1": pts[i],
                    "p2": pts[j],
                    "score1": sc[i],
                    "score2": sc[j],
                    "length": d_mat[i, j],
                    "validity": v_mat[i, j],
                    "mean_df": df_mean[i, j],
                })

    orphan_indices = np.where(~has_line)[0]
    orphan_rate = float(len(orphan_indices) / max(num_pts, 1))
    mean_length = float(np.mean([item["length"] for item in lines])) if lines else 0.0
    max_length = float(np.max([item["length"] for item in lines])) if lines else 0.0
    mean_df_val = float(np.mean([item["mean_df"] for item in lines])) if lines else 0.0

    return {
        "lines": lines,
        "endpoints": pts,
        "scores": sc,
        "num_lines": len(lines),
        "orphan_count": len(orphan_indices),
        "orphan_rate": orphan_rate,
        "mean_length": mean_length,
        "max_length": max_length,
        "mean_df": mean_df_val,
    }


def save_detection_mask_lines_figure(
    path: Path | str,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    pred0: dict[str, torch.Tensor],
    pred1: dict[str, torch.Tensor],
    df0: torch.Tensor,
    df1: torch.Tensor,
    af0: torch.Tensor,
    af1: torch.Tensor,
    loss_module: Any,
    pair_id: str = "",
    metrics_summary: dict[str, Any] | None = None,
    points0: torch.Tensor | np.ndarray | None = None,
    points1: torch.Tensor | np.ndarray | None = None,
) -> None:
    """渲染【检测验证: 原始端点与重构直线 + 预测热力图】图 (*_detection.jpg)。

    核心元素:
    1. 上排主图: 原始图像上清晰绘制真实端点 (绿色圆环)、预测端点 (黄色成线/红色孤立/青色候选) 与重构直线 (亮绿实线);
    2. 下排副图: 模型实际输出的高分辨率端点概率热力图 (pred['heatmap']) TURBO 伪彩色叠加显示;
    3. 底部状态栏: 标注直线条数、平均线长、端点最大置信度、孤立率与图例。
    """
    left_raw = np.ascontiguousarray(bgr0)
    right_raw = np.ascontiguousarray(bgr1)
    if left_raw.ndim == 2:
        left_raw = cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR)
    if right_raw.ndim == 2:
        right_raw = cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR)
    img_h, img_w = left_raw.shape[:2]

    geom0 = extract_detected_lines(pred0["heatmap"], df0, af0, loss_module)
    geom1 = extract_detected_lines(pred1["heatmap"], df1, af1, loss_module)

    def _render_reconstructed_lines(
        raw: np.ndarray,
        geom: dict[str, Any],
        pts_gt: Any,
        view_idx: int,
    ) -> np.ndarray:
        canvas = raw.copy()

        # 1. 绘制真实 GT 角点 (绿色空心圆环)
        if pts_gt is not None:
            gt_arr = pts_gt.detach().cpu().numpy() if torch.is_tensor(pts_gt) else np.asarray(pts_gt)
            if gt_arr.ndim == 3:
                gt_arr = gt_arr[0]
            for pt in gt_arr:
                gx, gy = int(round(float(pt[0]))), int(round(float(pt[1])))
                cv2.circle(canvas, (gx, gy), 4, (0, 255, 0), 1, cv2.LINE_AA)

        # 2. 绘制合法长直线
        for line in geom["lines"]:
            p1 = (int(round(float(line["p1"][0]))), int(round(float(line["p1"][1]))))
            p2 = (int(round(float(line["p2"][0]))), int(round(float(line["p2"][1]))))
            cv2.line(canvas, p1, p2, (0, 255, 60), 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, 4, (0, 255, 255), -1, cv2.LINE_AA)
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                canvas,
                f"{line['length']:.0f}px",
                (mid_x + 3, mid_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. 标注候选与孤立端点
        sc = geom["scores"]
        pts = geom["endpoints"]
        for i, pt in enumerate(pts):
            cx, cy = int(round(float(pt[0]))), int(round(float(pt[1])))
            score_val = float(sc[i])
            is_line_pt = any(
                np.allclose(line["p1"], pt, atol=1.0) or np.allclose(line["p2"], pt, atol=1.0)
                for line in geom["lines"]
            )
            if is_line_pt:
                continue

            # 孤立端点
            if score_val >= 0.05:
                cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 7, 2, cv2.LINE_AA)
                cv2.putText(
                    canvas, f"{score_val:.2f}", (cx + 4, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 165, 255), 1, cv2.LINE_AA
                )
            elif i < 8:
                # 初始阶段低分候选点 (前 8 个)
                cv2.drawMarker(canvas, (cx, cy), (255, 200, 0), cv2.MARKER_TILTED_CROSS, 5, 1, cv2.LINE_AA)
                cv2.putText(
                    canvas, f"{score_val:.2f}", (cx + 3, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1, cv2.LINE_AA
                )

        _draw_panel_title(canvas, f"[View {view_idx}] Reconstructed Lines ({geom['num_lines']}) & Endpoints")
        return canvas

    def _render_heatmap(raw: np.ndarray, heatmap: torch.Tensor, view_idx: int) -> np.ndarray:
        if heatmap.dim() == 4 and heatmap.shape[1] == 1:
            h_np = heatmap[0, 0].detach().cpu().numpy()
        elif heatmap.dim() == 3:
            h_np = heatmap[0].detach().cpu().numpy()
        else:
            h_np = heatmap.detach().cpu().numpy()

        h_min, h_max = float(h_np.min()), float(h_np.max())
        h_norm = (h_np - h_min) / max(h_max - h_min, 1e-6)
        h_u8 = (np.clip(h_norm, 0.0, 1.0) * 255).astype(np.uint8)
        h_color = cv2.applyColorMap(h_u8, cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(raw, 0.38, h_color, 0.62, 0)
        _draw_panel_title(vis, f"[View {view_idx}] Predicted Heatmap (max: {h_max:.4f}, mean: {h_np.mean():.4f})")
        return vis

    view0_vis = _render_reconstructed_lines(left_raw, geom0, points0, 0)
    view1_vis = _render_reconstructed_lines(right_raw, geom1, points1, 1)
    heat0_vis = _render_heatmap(left_raw, pred0["heatmap"], 0)
    heat1_vis = _render_heatmap(right_raw, pred1["heatmap"], 1)

    top_row = np.concatenate([view0_vis, view1_vis], axis=1)
    bottom_row = np.concatenate([heat0_vis, heat1_vis], axis=1)
    main_canvas = np.vstack([top_row, bottom_row])

    # 底部状态栏
    banner_h = 56
    banner = np.zeros((banner_h, img_w * 2, 3), dtype=np.uint8) + 24
    cv2.rectangle(banner, (0, 0), (img_w * 2 - 1, banner_h - 1), (60, 60, 60), 1)

    gt_info0 = f"GT: {len(points0)}" if points0 is not None else ""
    gt_info1 = f"GT: {len(points1)}" if points1 is not None else ""
    title_str = (
        f"Detection & Heatmap: {pair_id} | Lines: V0={geom0['num_lines']} (avg {geom0['mean_length']:.1f}px), "
        f"V1={geom1['num_lines']} (avg {geom1['mean_length']:.1f}px) | {gt_info0} {gt_info1}".strip()
    )
    max_s0 = float(geom0['scores'].max()) if len(geom0['scores']) > 0 else 0.0
    max_s1 = float(geom1['scores'].max()) if len(geom1['scores']) > 0 else 0.0
    sub_str = (
        f"Endpoints MaxScore: V0={max_s0:.3f}, V1={max_s1:.3f} | "
        f"Orphans: V0={geom0['orphan_count']} ({geom0['orphan_rate']*100:.1f}%), "
        f"V1={geom1['orphan_count']} ({geom1['orphan_rate']*100:.1f}%)"
    )
    if metrics_summary and "loss" in metrics_summary:
        sub_str += f" | Loss={metrics_summary['loss']:.4f}"

    legend_str = "Legend: [Green Ring]=GT Corner | [Yellow Dot]=Line Endpoint | [Red/Cyan Cross]=Orphan | [Green Line]=Valid Line"

    cv2.putText(banner, title_str, (16, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(banner, sub_str, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(banner, legend_str, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 180), 1, cv2.LINE_AA)

    combined = np.vstack([main_canvas, banner])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), combined)


def save_descriptor_matching_figure(
    path: Path | str,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    descriptor0: torch.Tensor,
    descriptor1: torch.Tensor,
    points0: torch.Tensor | np.ndarray,
    points1: torch.Tensor | np.ndarray,
    homography: torch.Tensor | np.ndarray,
    pair_id: str = "",
    metrics_summary: dict[str, Any] | None = None,
) -> None:
    """渲染【描述验证: 配对情况】图 (*_matching.jpg)。

    核心元素:
    1. 左右视角双图拼接;
    2. 基于 256 维局部描述子最近邻检索跨视角对应点;
    3. 利用真实单应变换 (Ground-Truth H) 计算重投影几何误差:
       - 投影误差 <= 3.0px: 正确配对 (亮绿色连线与端点);
       - 投影误差 > 3.0px: 误配 (红色连线与端点);
    4. 底部横幅: 标注同名点最近邻匹配准确率 (NN Acc %)、MMA@1/2/3px 精度。
    """
    left = np.ascontiguousarray(bgr0)
    right = np.ascontiguousarray(bgr1)
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    img_h, img_w = left.shape[:2]

    canvas = np.concatenate([left.copy(), right.copy()], axis=1)

    pts0_t = torch.as_tensor(points0, dtype=torch.float32, device=descriptor0.device)
    pts1_t = torch.as_tensor(points1, dtype=torch.float32, device=descriptor1.device)
    if pts0_t.dim() == 3:
        pts0_t = pts0_t[0]
    if pts1_t.dim() == 3:
        pts1_t = pts1_t[0]

    mma_res = match_accuracy_mma(descriptor0, descriptor1, pts0_t, pts1_t, homography)
    acc, choice = match_accuracy(descriptor0, descriptor1, pts0_t, pts1_t)

    pts0_np = pts0_t.detach().cpu().numpy()
    pts1_np = pts1_t.detach().cpu().numpy()
    choice_np = choice.detach().cpu().numpy()

    h_np = homography.detach().cpu().numpy() if isinstance(homography, torch.Tensor) else np.asarray(homography)
    if h_np.ndim == 3:
        h_np = h_np[0]

    if len(pts0_np) > 0 and len(pts1_np) > 0:
        ones = np.ones((len(pts0_np), 1), dtype=np.float32)
        p0_homo = np.hstack([pts0_np, ones])
        warped = (h_np @ p0_homo.T).T
        w = np.clip(warped[:, 2:3], 1e-6, None)
        warped_p0 = warped[:, :2] / w

        for idx, p0 in enumerate(pts0_np):
            c_idx = int(choice_np[idx]) if idx < len(choice_np) else -1
            if 0 <= c_idx < len(pts1_np):
                chosen_p1 = pts1_np[c_idx]
                reproj_err = float(np.linalg.norm(warped_p0[idx] - chosen_p1))
                is_correct = reproj_err <= 3.0
                color = (0, 230, 80) if is_correct else (0, 40, 230)

                start = (int(round(float(p0[0]))), int(round(float(p0[1]))))
                end = (int(round(float(chosen_p1[0]))) + img_w, int(round(float(chosen_p1[1]))))

                cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, start, 3, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, end, 3, color, -1, cv2.LINE_AA)

    _draw_panel_title(canvas[:, :img_w], "[View 0] Source Keypoints")
    _draw_panel_title(canvas[:, img_w:], "[View 1] Target Keypoints")

    # 底部指标横幅
    banner_h = 48
    banner = np.zeros((banner_h, img_w * 2, 3), dtype=np.uint8) + 24
    cv2.rectangle(banner, (0, 0), (img_w * 2 - 1, banner_h - 1), (60, 60, 60), 1)

    matched_cnt = int(round(acc * len(pts0_np)))
    title_str = (
        f"Descriptor Validation: {pair_id} | NN Match Accuracy: {acc * 100:.1f}% "
        f"({matched_cnt}/{len(pts0_np)} matched)"
    )
    mma_str = (
        f"MMA: @1.0px={mma_res.get('mma_1.0px', 0.0)*100:.1f}%, "
        f"@2.0px={mma_res.get('mma_2.0px', 0.0)*100:.1f}%, "
        f"@3.0px={mma_res.get('mma_3.0px', 0.0)*100:.1f}%"
    )
    if metrics_summary and "ld" in metrics_summary:
        mma_str += f" | Desc Hinge Loss={metrics_summary['ld']:.4f}"

    cv2.putText(banner, title_str, (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(banner, mma_str, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 180), 1, cv2.LINE_AA)

    combined = np.vstack([canvas, banner])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), combined)


def save_superpoint_diagnostic_sheet(
    path: Path | str,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    pred0: dict[str, torch.Tensor],
    pred1: dict[str, torch.Tensor],
    df0: torch.Tensor,
    df1: torch.Tensor,
    af0: torch.Tensor,
    af1: torch.Tensor,
    points0: np.ndarray | torch.Tensor,
    points1: np.ndarray | torch.Tensor,
    homography: np.ndarray | torch.Tensor,
    pair_id: str,
    metrics_summary: dict[str, Any],
    loss_module: Any,
) -> None:
    """渲染四格全景诊断大图 (Heatmap Overlay + Valid Lines + Cross-View Matching + Status Banner)."""
    left_raw = np.ascontiguousarray(bgr0)
    right_raw = np.ascontiguousarray(bgr1)
    if left_raw.ndim == 2:
        left_raw = cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR)
    if right_raw.ndim == 2:
        right_raw = cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR)
    img_h, img_w = left_raw.shape[:2]

    # 提取直线拓扑与端点
    geom0 = extract_detected_lines(pred0["heatmap"], df0, af0, loss_module)
    geom1 = extract_detected_lines(pred1["heatmap"], df1, af1, loss_module)

    # -------------------------------------------------------------------------
    # Panel 1: 端点热力图辉光叠加 (Heatmap Overlay)
    # -------------------------------------------------------------------------
    def _render_heatmap_panel(img: np.ndarray, heatmap: torch.Tensor, pts: np.ndarray) -> np.ndarray:
        hm = heatmap[0, 0].detach().cpu().numpy()
        hm_u8 = (np.clip(hm, 0.0, 1.0) * 255).astype(np.uint8)
        colored_hm = cv2.applyColorMap(hm_u8, cv2.COLORMAP_MAGMA)
        overlay = cv2.addWeighted(img, 0.65, colored_hm, 0.35, 0)
        for pt in pts:
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            cv2.drawMarker(overlay, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 3, (255, 200, 0), 1, cv2.LINE_AA)
        return overlay

    p1_left = _render_heatmap_panel(left_raw, pred0["heatmap"], geom0["endpoints"])
    p1_right = _render_heatmap_panel(right_raw, pred1["heatmap"], geom1["endpoints"])
    panel1 = np.concatenate([p1_left, p1_right], axis=1)
    _draw_panel_title(panel1, "[Panel 1] Endpoint Probability Heatmaps & Detected Corners")

    # -------------------------------------------------------------------------
    # Panel 2: 重构的合法直线段 (Extracted Valid Lines)
    # -------------------------------------------------------------------------
    def _render_lines_panel(img: np.ndarray, geom: dict[str, Any]) -> np.ndarray:
        canvas = img.copy()
        # 绘制合法直线
        for line in geom["lines"]:
            p1 = (int(round(float(line["p1"][0]))), int(round(float(line["p1"][1]))))
            p2 = (int(round(float(line["p2"][0]))), int(round(float(line["p2"][1]))))
            cv2.line(canvas, p1, p2, (0, 230, 80), 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, 3, (0, 255, 255), -1, cv2.LINE_AA)
            # 标注长度
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            cv2.putText(
                canvas, f"{line['length']:.0f}px", (mid_x + 3, mid_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
            )
        # 标记孤立端点为红色小叉
        for i, pt in enumerate(geom["endpoints"]):
            if i in np.where(geom["scores"] >= 0.05)[0]:
                is_orphan = not any(
                    np.allclose(line["p1"], pt, atol=1.0) or np.allclose(line["p2"], pt, atol=1.0)
                    for line in geom["lines"]
                )
                if is_orphan:
                    cx, cy = int(round(float(pt[0]))), int(round(float(pt[1])))
                    cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA)
        return canvas

    p2_left = _render_lines_panel(left_raw, geom0)
    p2_right = _render_lines_panel(right_raw, geom1)
    panel2 = np.concatenate([p2_left, p2_right], axis=1)
    _draw_panel_title(
        panel2,
        f"[Panel 2] Extracted Valid Straight Lines (View0: {geom0['num_lines']}, View1: {geom1['num_lines']})"
    )

    # -------------------------------------------------------------------------
    # Panel 3: 跨视角同名点特征匹配 (Cross-View Matching with Ground Truth H)
    # -------------------------------------------------------------------------
    pts0_np = points0.detach().cpu().numpy() if isinstance(points0, torch.Tensor) else np.asarray(points0)
    pts1_np = points1.detach().cpu().numpy() if isinstance(points1, torch.Tensor) else np.asarray(points1)
    panel3 = np.concatenate([left_raw.copy(), right_raw.copy()], axis=1)

    if len(pts0_np) > 0 and len(pts1_np) > 0:
        _, choice = match_accuracy(
            pred0["descriptor"], pred1["descriptor"],
            torch.from_numpy(pts0_np).to(pred0["descriptor"].device),
            torch.from_numpy(pts1_np).to(pred1["descriptor"].device),
        )
        choice_np = choice.detach().cpu().numpy()
        for idx, p0 in enumerate(pts0_np):
            c_idx = int(choice_np[idx]) if idx < len(choice_np) else -1
            hit = c_idx == idx
            color = (0, 230, 80) if hit else (0, 40, 230)
            target_pt = pts1_np[c_idx] if 0 <= c_idx < len(pts1_np) else pts1_np[idx]
            pt_left = (int(round(float(p0[0]))), int(round(float(p0[1]))))
            pt_right = (int(round(float(target_pt[0]))) + img_w, int(round(float(target_pt[1]))))
            cv2.line(panel3, pt_left, pt_right, color, 1, cv2.LINE_AA)
            cv2.circle(panel3, pt_left, 3, color, -1, cv2.LINE_AA)
            cv2.circle(panel3, pt_right, 3, color, -1, cv2.LINE_AA)

    _draw_panel_title(panel3, "[Panel 3] Cross-View Keypoint Feature Matching (Green=Correct, Red=Error)")

    # -------------------------------------------------------------------------
    # Panel 4: 底部指标横幅 (Status Banner)
    # -------------------------------------------------------------------------
    banner_h = 56
    banner = np.zeros((banner_h, img_w * 2, 3), dtype=np.uint8) + 24
    cv2.rectangle(banner, (0, 0), (img_w * 2 - 1, banner_h - 1), (60, 60, 60), 1)

    line1 = (
        f"Pair: {pair_id} | Total Loss: {metrics_summary.get('loss', 0.0):.4f} "
        f"[desc={metrics_summary.get('ld', 0.0):.3f}, det={metrics_summary.get('l_det', 0.0):.3f}, "
        f"orph={metrics_summary.get('l_orphan', 0.0):.3f}, len={metrics_summary.get('l_len', 0.0):.3f}, "
        f"prem={metrics_summary.get('l_prem', 0.0):.3f}]"
    )
    line2 = (
        f"Corner Rep@3px: {metrics_summary.get('repeatability', 0.0) * 100:.1f}% | "
        f"Valid Lines: {geom0['num_lines']} (mean_len={geom0['mean_length']:.1f}px) | "
        f"Orphan Rate: {geom0['orphan_rate'] * 100:.1f}% | "
        f"Match Acc: {metrics_summary.get('match_acc', 0.0) * 100:.1f}%"
    )
    cv2.putText(banner, line1, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(banner, line2, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 230, 255), 1, cv2.LINE_AA)

    # 垂直拼装 4 个面板
    full_sheet = np.vstack([panel1, panel2, panel3, banner])

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), full_sheet)


def save_point_match_figure(
    path: Path | str,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    choice: np.ndarray,
) -> None:
    """向后兼容的简版特征点配对图。"""
    left = np.ascontiguousarray(bgr0)
    right = np.ascontiguousarray(bgr1)
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    _, width = left.shape[:2]
    canvas = np.concatenate([left, right], axis=1)
    correct = 0
    for index, (source, target) in enumerate(zip(points0, points1)):
        predicted = int(choice[index]) if index < len(choice) else -1
        hit = predicted == index
        correct += int(hit)
        color = (40, 200, 40) if hit else (40, 40, 220)
        start = (int(round(float(source[0]))), int(round(float(source[1]))))
        end_point = points1[predicted] if 0 <= predicted < len(points1) else target
        end = (int(round(float(end_point[0]))) + width, int(round(float(end_point[1]))))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (end[0], end[1]), 3, color, -1, cv2.LINE_AA)
    total = len(points0)
    cv2.rectangle(canvas, (0, 0), (168, 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        f"{correct}/{total}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), canvas)


def _draw_panel_title(canvas: np.ndarray, title: str) -> None:
    """在面板左上角绘制半透明标志牌。"""
    cv2.rectangle(canvas, (6, 6), (len(title) * 8 + 18, 28), (20, 20, 20), -1)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _sample_descriptors(descriptor: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """descriptor is (1, C, H, W) or (C, H, W). points is (K, 2) in pixels."""
    if descriptor.dim() == 3:
        descriptor = descriptor.unsqueeze(0)
    _, _, height, width = descriptor.shape
    x = points[:, 0] / max(width - 1, 1) * 2.0 - 1.0
    y = points[:, 1] / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(descriptor, grid, align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def save_line_descriptor_pca_figure(
    path: Path | str,
    bgr0: np.ndarray,
    bgr1: np.ndarray,
    descriptor0: torch.Tensor,
    descriptor1: torch.Tensor,
    lines0: torch.Tensor | np.ndarray | None = None,
    lines1: torch.Tensor | np.ndarray | None = None,
    pair_id: str = "",
    metrics_summary: dict[str, Any] | None = None,
) -> None:
    """渲染【线一致性像素描述子 PCA 伪彩流形图】(*_line_pca.jpg)。

    核心功能:
    1. 将两视角 256 维稠密特征场通过联合 PCA 降维投影至 3 维 RGB 空间;
    2. 直观检验用户核心需求:
       - 同一条线上的各像素: 特征相近 -> RGB 颜色纯净均一;
       - 跨视角同名物理直线: 特征相同 -> 左右两图同一直线呈现完全相同的 RGB 颜色;
       - 不同的直线段: 特征正交推离 -> 呈现截然不同的 RGB 颜色;
       - 背景像素: 处于不同特征簇，与直线形成鲜明色差对比。
    """
    left_raw = np.ascontiguousarray(bgr0)
    right_raw = np.ascontiguousarray(bgr1)
    if left_raw.ndim == 2:
        left_raw = cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR)
    if right_raw.ndim == 2:
        right_raw = cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR)
    img_h, img_w = left_raw.shape[:2]

    # 获取 (1, C, H, W) 特征
    d0 = descriptor0 if descriptor0.dim() == 4 else descriptor0.unsqueeze(0)
    d1 = descriptor1 if descriptor1.dim() == 4 else descriptor1.unsqueeze(0)

    # 统一降维: 展平为 (2*H*W, C)
    c = d0.shape[1]
    f0 = d0[0].permute(1, 2, 0).reshape(-1, c)  # (H*W, C)
    f1 = d1[0].permute(1, 2, 0).reshape(-1, c)
    combined_f = torch.cat([f0, f1], dim=0)     # (2*H*W, C)

    with torch.no_grad():
        # 联合 PCA 降至 3 维
        _, _, v = torch.pca_lowrank(combined_f, q=3, center=True)
        proj = torch.matmul(combined_f, v[:, :3])  # (2*H*W, 3)

        # 归一化到 [0, 255] RGB
        p_min = proj.min(dim=0, keepdim=True)[0]
        p_max = proj.max(dim=0, keepdim=True)[0]
        norm_proj = (proj - p_min) / (p_max - p_min + 1e-6)
        rgb_combined = (norm_proj.clamp(0.0, 1.0) * 255.0).byte().cpu().numpy()

    rgb0 = rgb_combined[:img_h * img_w].reshape(img_h, img_w, 3)
    rgb1 = rgb_combined[img_h * img_w:].reshape(img_h, img_w, 3)
    # 转为 BGR 供 OpenCV 保存
    bgr_pca0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR)
    bgr_pca1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)

    def _draw_lines_on_pca(pca_img: np.ndarray, lines_data: Any, raw_img: np.ndarray) -> np.ndarray:
        vis = cv2.addWeighted(raw_img, 0.25, pca_img, 0.75, 0)
        if lines_data is not None:
            ln_arr = lines_data.detach().cpu().numpy() if torch.is_tensor(lines_data) else np.asarray(lines_data)
            if ln_arr.ndim == 3:
                ln_arr = ln_arr[0]
            for idx, l in enumerate(ln_arr):
                p1 = (int(round(float(l[0]))), int(round(float(l[1]))))
                p2 = (int(round(float(l[2]))), int(round(float(l[3]))))
                cv2.line(vis, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(vis, p1, 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(vis, p2, 3, (0, 255, 255), -1, cv2.LINE_AA)
        return vis

    pca_vis0 = _draw_lines_on_pca(bgr_pca0, lines0, left_raw)
    pca_vis1 = _draw_lines_on_pca(bgr_pca1, lines1, right_raw)

    _draw_panel_title(pca_vis0, "[View 0] Line-Consistent PCA Descriptors")
    _draw_panel_title(pca_vis1, "[View 1] Line-Consistent PCA Descriptors")

    main_canvas = np.concatenate([pca_vis0, pca_vis1], axis=1)

    banner_h = 42
    banner = np.zeros((banner_h, img_w * 2, 3), dtype=np.uint8) + 24
    cv2.rectangle(banner, (0, 0), (img_w * 2 - 1, banner_h - 1), (60, 60, 60), 1)

    title_str = f"Descriptor PCA Color Space: {pair_id} | Same Line = Same Color | Cross-Image Line Match = Same Color"
    cv2.putText(banner, title_str, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 240, 255), 1, cv2.LINE_AA)

    combined = np.vstack([main_canvas, banner])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), combined)


__all__ = [
    "match_accuracy",
    "match_accuracy_mma",
    "calculate_corner_repeatability",
    "extract_detected_lines",
    "save_detection_mask_lines_figure",
    "save_descriptor_matching_figure",
    "save_line_descriptor_pca_figure",
    "save_superpoint_diagnostic_sheet",
    "save_point_match_figure",
]
