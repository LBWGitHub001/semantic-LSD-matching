"""SuperPoint 解耦损失函数体系 (Decoupled Loss System for SuperPoint).

包含两大完全独立的损失类以及统一的联合损失类:
1. SuperPointDescriptorLoss:
   纯描述子度量合页损失 (同名点推近至 1.0，异名点推离至 0.2 以下，250 倍正样本加权)。
2. SuperPointDetectorLoss:
   端点检测与 4 项直线几何场联合损失:
   - detector_loss: 端点加权 BCE 正向引导;
   - orphan_endpoint_loss: 孤立未连线端点置信度惩罚;
   - max_length_line_loss: 超线性长线收益奖励 ((d / D_max)^1.5);
   - premature_ray_probing_loss: 端点外推射线探针过早早退惩罚.
3. SuperPointJointLoss (别名 SuperPointLoss, SuperPointLineLoss):
   组合描述子损失与端点/几何损失，支持联合端到端训练与向后兼容。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 描述子度量与线级一致性损失类 (Descriptor Metric & Line-Consistent Loss)
# =============================================================================

def sample_descriptors_along_lines(
    descriptors: torch.Tensor,
    lines: Any,
    num_samples: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿成组直线段密集采样像素描述子，并计算每条线的单位特征原型。

    参数:
        descriptors: (B, C, H, W) 全分辨率密集描述子图
        lines: (B, K, 4) 或 (K, 4) 直线段坐标 [x1, y1, x2, y2]
        num_samples: 每条线段沿线均匀采样点数

    返回:
        sampled_desc: (B, K, num_samples, C) 沿线采样的单位归一化描述子
        line_prototypes: (B, K, C) 每条线段的归一化原型特征向量
    """
    if isinstance(lines, np.ndarray):
        lines = torch.from_numpy(lines)
    if isinstance(lines, (list, tuple)):
        lines = torch.as_tensor(lines, dtype=torch.float32, device=descriptors.device)
    lines = lines.to(device=descriptors.device, dtype=torch.float32)

    if lines.dim() == 2:
        lines = lines.unsqueeze(0).expand(descriptors.shape[0], -1, -1)
    batch_size, num_lines, _ = lines.shape
    c = descriptors.shape[1]
    if num_lines == 0:
        return (
            torch.zeros((batch_size, 0, num_samples, c), device=descriptors.device, dtype=descriptors.dtype),
            torch.zeros((batch_size, 0, c), device=descriptors.device, dtype=descriptors.dtype),
        )

    _, _, height, width = descriptors.shape
    device = descriptors.device
    dtype = descriptors.dtype

    p1 = lines[..., :2]  # (B, K, 2)
    p2 = lines[..., 2:]  # (B, K, 2)
    diff = p2 - p1       # (B, K, 2)

    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=dtype).view(1, 1, num_samples, 1)
    pts = p1.unsqueeze(2) + t * diff.unsqueeze(2)  # (B, K, num_samples, 2)

    # 归一化至 [-1, 1] 供 grid_sample
    gx = 2.0 * pts[..., 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * pts[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (B, K, num_samples, 2)

    sampled = F.grid_sample(
        descriptors,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, K, num_samples)
    sampled = sampled.permute(0, 2, 3, 1)  # (B, K, num_samples, C)
    sampled = F.normalize(sampled, p=2, dim=-1)

    # 计算线原型向量
    proto = sampled.mean(dim=2)  # (B, K, C)
    proto = F.normalize(proto, p=2, dim=-1)

    return sampled, proto


def sample_background_descriptors(
    descriptors: torch.Tensor,
    df_or_mask: torch.Tensor | None = None,
    num_samples: int = 64,
) -> torch.Tensor:
    """在背景区域 (非直线区域) 采样像素描述子，用于方案 A 的对比推离。"""
    batch_size, c, height, width = descriptors.shape
    device = descriptors.device
    dtype = descriptors.dtype

    if df_or_mask is not None and df_or_mask.shape[-2:] == (height, width):
        # 判定背景: 若距离场数值小 (< 0.25) 或掩膜为 0
        bg_prob = (df_or_mask < 0.25).float().view(batch_size, -1)
    else:
        # 默认均匀采样背景概率
        bg_prob = torch.ones((batch_size, height * width), device=device, dtype=dtype)

    bg_indices = torch.multinomial(bg_prob.clamp(min=1e-5), num_samples=num_samples, replacement=True)
    flat_desc = descriptors.permute(0, 2, 3, 1).reshape(batch_size, height * width, c)
    idx_expanded = bg_indices.unsqueeze(-1).expand(-1, -1, c)
    bg_desc = torch.gather(flat_desc, 1, idx_expanded)  # (B, num_samples, C)
    return F.normalize(bg_desc, p=2, dim=-1)


class SuperPointDescriptorLoss(nn.Module):
    """SuperPoint 描述子度量与线级一致性损失模块 (集成方案 A 线级一致性与对比推离约束).

    双重模式:
    1. 线级一致模式 (当提供 lines0 与 lines1 时触发):
       - L_intra: 同一条线上的各像素描述子余弦相似度拉近至 1.0 (同一条线上值相似);
       - L_cross: 跨图对应同名直线段的原型描述子余弦相似度拉近至 1.0 (不同图上同一直线描述相同);
       - L_inter: 异线原型相互推离至 <= line_neg_margin (0.2);
       - L_bg: 背景像素样本与线原型推离至 <= bg_neg_margin (0.1) (方案 A 负样本推离).
    2. 经典网格模式 (向后兼容):
       - 距离单应投影位置 <= 8px 判定为同名正样本对，拉近至 positive_margin (1.0);
       - 其余判定为负样本对，推远至 negative_margin (0.2) 以下。
    """

    def __init__(
        self,
        positive_weight: float = 250.0,
        positive_margin: float = 1.0,
        negative_margin: float = 0.2,
        weight_desc: float = 1.0,
        weight_intra: float = 1.0,
        weight_cross: float = 1.5,
        weight_inter: float = 0.5,
        weight_bg: float = 0.3,
        line_neg_margin: float = 0.2,
        bg_neg_margin: float = 0.1,
        num_line_points: int = 12,
        num_bg_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.positive_weight = float(positive_weight)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.weight_desc = float(weight_desc)
        self.weight_intra = float(weight_intra)
        self.weight_cross = float(weight_cross)
        self.weight_inter = float(weight_inter)
        self.weight_bg = float(weight_bg)
        self.line_neg_margin = float(line_neg_margin)
        self.bg_neg_margin = float(bg_neg_margin)
        self.num_line_points = int(num_line_points)
        self.num_bg_samples = int(num_bg_samples)

    def descriptor_loss(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        homography: torch.Tensor,
    ) -> torch.Tensor:
        """计算两视角半稠密网格描述子之间的合页度量损失 (经典点级 SuperPoint 损失)。"""
        batch, channels, cells_h, cells_w = desc0.shape
        num_cells = cells_h * cells_w

        flat0 = desc0.permute(0, 2, 3, 1).reshape(batch, num_cells, channels)
        flat1 = desc1.permute(0, 2, 3, 1).reshape(batch, num_cells, channels)

        dot = torch.matmul(flat0, flat1.transpose(1, 2))

        centers = _cell_centers(cells_h, cells_w, desc0.device, desc0.dtype)
        centers = centers.reshape(1, num_cells, 2).expand(batch, -1, -1)

        warped = _apply_homography(homography, centers)
        distance = torch.cdist(warped, centers)

        positive = (distance <= 8.0).to(dot.dtype)

        hinge_pos = self.positive_weight * positive * F.relu(self.positive_margin - dot)
        hinge_neg = (1.0 - positive) * F.relu(dot - self.negative_margin)

        return (hinge_pos + hinge_neg).mean()

    def line_descriptor_loss(
        self,
        dense0: torch.Tensor,
        dense1: torch.Tensor,
        lines0: Any,
        lines1: Any,
        image0: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """计算线级一致性与对比推离损失 (方案 A)。"""
        sampled0, proto0 = sample_descriptors_along_lines(dense0, lines0, self.num_line_points)
        sampled1, proto1 = sample_descriptors_along_lines(dense1, lines1, self.num_line_points)

        batch_size, k, num_samples, _ = sampled0.shape
        device = dense0.device

        if k == 0:
            zero = torch.tensor(0.0, device=device)
            return {"loss": zero, "l_intra": zero, "l_cross": zero, "l_inter": zero, "l_bg": zero}

        # 1. 线内一致性损失 (同一条线上值相似)
        cos0 = (sampled0 * proto0.unsqueeze(2)).sum(dim=-1)  # (B, K, num_samples)
        cos1 = (sampled1 * proto1.unsqueeze(2)).sum(dim=-1)
        l_intra = 0.5 * ((1.0 - cos0).mean() + (1.0 - cos1).mean())

        # 2. 跨图同名线一致性 (不同图上同一个直线的描述应该相同)
        cos_proto = (proto0 * proto1).sum(dim=-1)  # (B, K)
        cos_pts = (sampled0 * sampled1).sum(dim=-1)  # (B, K, num_samples)
        l_cross = 0.5 * ((1.0 - cos_proto).mean() + (1.0 - cos_pts).mean())

        # 3. 异线原型推离 (防止特征坍塌)
        dot_proto = torch.matmul(proto0, proto1.transpose(-1, -2))  # (B, K, K)
        eye = torch.eye(k, device=device, dtype=torch.bool).unsqueeze(0)
        if k > 1:
            off_diag = dot_proto[~eye]
            l_inter = F.relu(off_diag - self.line_neg_margin).mean()
        else:
            l_inter = torch.tensor(0.0, device=device)

        # 4. 背景像素推离 (方案 A)
        df0 = image0[:, 3:4] if (image0 is not None and image0.shape[1] >= 4) else None
        bg_desc = sample_background_descriptors(dense0, df0, self.num_bg_samples)  # (B, N_bg, C)
        bg_sim = torch.matmul(bg_desc, proto0.transpose(-1, -2))  # (B, N_bg, K)
        l_bg = F.relu(bg_sim - self.bg_neg_margin).mean()

        total = (
            self.weight_intra * l_intra
            + self.weight_cross * l_cross
            + self.weight_inter * l_inter
            + self.weight_bg * l_bg
        )

        return {
            "loss": total,
            "l_intra": l_intra,
            "l_cross": l_cross,
            "l_inter": l_inter,
            "l_bg": l_bg,
        }

    def forward(
        self,
        pred0: dict[str, torch.Tensor],
        pred1: dict[str, torch.Tensor],
        homography: torch.Tensor,
        lines0: Any = None,
        lines1: Any = None,
        image0: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """独立描述子损失前向传播。支持自动检测线级一致损失或经典合页损失。"""
        if lines0 is None:
            lines0 = kwargs.get("lines0")
        if lines1 is None:
            lines1 = kwargs.get("lines1")
        if image0 is None:
            image0 = kwargs.get("image0")
        if image1 is None:
            image1 = kwargs.get("image1")

        has_lines = lines0 is not None and lines1 is not None and len(lines0) > 0
        if has_lines:
            dense0 = pred0.get("descriptor", pred0["desc"])
            dense1 = pred1.get("descriptor", pred1["desc"])
            line_res = self.line_descriptor_loss(dense0, dense1, lines0, lines1, image0, image1)
            total_loss = self.weight_desc * line_res["loss"]
            return {
                "loss": total_loss,
                "ld": total_loss.detach(),
                "l_intra": line_res["l_intra"].detach(),
                "l_cross": line_res["l_cross"].detach(),
                "l_inter": line_res["l_inter"].detach(),
                "l_bg": line_res["l_bg"].detach(),
            }
        else:
            ld = self.descriptor_loss(pred0["desc"], pred1["desc"], homography)
            total_loss = self.weight_desc * ld
            return {
                "loss": total_loss,
                "ld": ld.detach(),
            }


# =============================================================================
# 2. 端点检测与直线几何场损失类 (Detector & Line Geometry Loss)
# =============================================================================

class SuperPointDetectorLoss(nn.Module):
    """SuperPoint 端点检测与直线几何场综合损失模块。

    包含 4 项约束:
    1. detector_loss: 锚定 GT 端点的加权 BCE 损失;
    2. orphan_endpoint_loss: 孤立未连线端点几何惩罚;
    3. max_length_line_loss: 超线性长线收益奖励;
    4. premature_ray_probing_loss: 端点外推射线探针过早早退惩罚。
    """

    def __init__(
        self,
        weight_det: float = 1.0,
        weight_orphan: float = 1.0,
        weight_len: float = 0.5,
        weight_premature: float = 0.5,
        tau_df: float = 0.55,
        tau_ang: float = 0.10,
        min_length: float = 8.0,
        num_candidates: int = 16,
        num_line_samples: int = 10,
        probe_dist: float = 4.0,
        det_pos_weight: float = 50.0,
        length_exponent: float = 1.5,
        nms_radius: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.weight_det = float(weight_det)
        self.weight_orphan = float(weight_orphan)
        self.weight_len = float(weight_len)
        self.weight_premature = float(weight_premature)

        self.tau_df = float(tau_df)
        self.tau_ang = float(tau_ang)
        self.min_length = float(min_length)
        self.num_candidates = int(num_candidates)
        self.num_line_samples = int(num_line_samples)
        self.probe_dist = float(probe_dist)
        self.det_pos_weight = float(det_pos_weight)
        self.length_exponent = float(length_exponent)
        self.nms_radius = int(nms_radius)

    # -------------------------------------------------------------------------
    # 损失函数类成员
    # -------------------------------------------------------------------------

    def detector_loss(
        self,
        semi_or_heatmap: torch.Tensor,
        points_gt: Any,
    ) -> torch.Tensor:
        """【类成员损失项 1】端点检测损失。若传入 65 维 Logits (semi)，使用 SuperPoint 经典网格多分类交叉熵；若为单通道 heatmap，降级为加权 BCE。"""
        if points_gt is None:
            return torch.tensor(0.0, device=semi_or_heatmap.device, dtype=semi_or_heatmap.dtype)

        batch_size = semi_or_heatmap.shape[0]
        device = semi_or_heatmap.device

        if isinstance(points_gt, torch.Tensor):
            if points_gt.dim() == 2:
                pts_list = [points_gt]
            elif points_gt.dim() == 3:
                pts_list = [points_gt[b] for b in range(points_gt.shape[0])]
            else:
                pts_list = []
        elif isinstance(points_gt, list):
            pts_list = points_gt
        else:
            pts_list = []

        has_any_gt = any(pts is not None and len(pts) > 0 for pts in pts_list)
        if not has_any_gt:
            return torch.tensor(0.0, device=device, dtype=semi_or_heatmap.dtype)

        # 1. 经典 SuperPoint 65 维 Logits 网格交叉熵 (8x8 cell 分类)
        if semi_or_heatmap.shape[1] == 65:
            _, _, hc, wc = semi_or_heatmap.shape
            labels = torch.full((batch_size, hc, wc), 64, dtype=torch.long, device=device)
            for b, pts in enumerate(pts_list):
                if b >= batch_size or pts is None or len(pts) == 0:
                    continue
                coords = pts.round().long()
                for pt in coords:
                    px = int(pt[0].item())
                    py = int(pt[1].item())
                    cx, cy = px // 8, py // 8
                    dx, dy = px % 8, py % 8
                    if 0 <= cx < wc and 0 <= cy < hc:
                        labels[b, cy, cx] = dy * 8 + dx

            weights = torch.ones(65, device=device, dtype=semi_or_heatmap.dtype)
            weights[:-1] = self.det_pos_weight
            return F.cross_entropy(semi_or_heatmap, labels, weight=weights)

        # 2. 单通道 Heatmap 加权 BCE 降级方案
        _, _, height, width = semi_or_heatmap.shape
        heatmap = semi_or_heatmap
        target = torch.zeros_like(heatmap)
        for b, pts in enumerate(pts_list):
            if b >= batch_size or pts is None or len(pts) == 0:
                continue
            coords = pts.round().long()
            for pt in coords:
                px = int(pt[0].clamp(0, width - 1).item())
                py = int(pt[1].clamp(0, height - 1).item())
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            val = 1.0 if (dy == 0 and dx == 0) else 0.5
                            if val > target[b, 0, ny, nx]:
                                target[b, 0, ny, nx] = val

        bce = -(
            self.det_pos_weight * target * torch.log(heatmap.clamp(min=1e-6))
            + (1.0 - target) * torch.log((1.0 - heatmap).clamp(min=1e-6))
        ).mean()
        return bce

    def orphan_endpoint_loss(
        self,
        scores: torch.Tensor,
        v_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """【类成员损失项 2】孤立端点几何合法性惩罚 (Orphan Endpoint Penalty)."""
        s_support = v_matrix.max(dim=-1)[0]  # (B, K)
        return (scores * (1.0 - s_support)).mean()

    def max_length_line_loss(
        self,
        scores: torch.Tensor,
        dist: torch.Tensor,
        v_matrix: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """【类成员损失项 3】超线性长线收益奖励损失 (Super-linear Max Length Reward Loss)."""
        diag = math.sqrt(float(height**2 + width**2))
        reward_len = (dist / max(diag, 1.0)) ** self.length_exponent
        pair_scores = scores.unsqueeze(2) * scores.unsqueeze(1)  # (B, K, K)

        denom = v_matrix.sum(dim=(-1, -2)) + 1e-4
        return - ((v_matrix * pair_scores * reward_len).sum(dim=(-1, -2)) / denom).mean()

    def premature_ray_probing_loss(
        self,
        scores: torch.Tensor,
        dist: torch.Tensor,
        diff: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        v_matrix: torch.Tensor,
        df_map: torch.Tensor,
    ) -> torch.Tensor:
        """【类成员损失项 4】端点外推射线探针过早终止惩罚 (Premature Ray Probing Penalty)."""
        batch_size, k = scores.shape
        _, _, height, width = df_map.shape

        dir_vec = diff / (dist.unsqueeze(-1) + 1e-6)
        probe_j = p2 + self.probe_dist * dir_vec  # (B, K, K, 2)
        probe_i = p1 - self.probe_dist * dir_vec  # (B, K, K, 2)

        probe_j_g = torch.stack(
            [2.0 * probe_j[..., 0] / max(width - 1, 1) - 1.0, 2.0 * probe_j[..., 1] / max(height - 1, 1) - 1.0],
            dim=-1,
        ).view(batch_size, k * k, 1, 2)
        probe_i_g = torch.stack(
            [2.0 * probe_i[..., 0] / max(width - 1, 1) - 1.0, 2.0 * probe_i[..., 1] / max(height - 1, 1) - 1.0],
            dim=-1,
        ).view(batch_size, k * k, 1, 2)

        df_pj = F.grid_sample(df_map, probe_j_g, mode="bilinear", padding_mode="zeros", align_corners=True).view(
            batch_size, k, k
        )
        df_pi = F.grid_sample(df_map, probe_i_g, mode="bilinear", padding_mode="zeros", align_corners=True).view(
            batch_size, k, k
        )

        ext_j = torch.sigmoid((df_pj - self.tau_df) / 0.05)
        ext_i = torch.sigmoid((df_pi - self.tau_df) / 0.05)
        pair_scores = scores.unsqueeze(2) * scores.unsqueeze(1)

        denom = v_matrix.sum(dim=(-1, -2)) + 1e-4
        return ((v_matrix * pair_scores * (ext_j + ext_i)).sum(dim=(-1, -2)) / denom).mean()

    # -------------------------------------------------------------------------
    # 直线几何辅助类成员
    # -------------------------------------------------------------------------

    def extract_candidate_endpoints(
        self,
        heatmap: torch.Tensor,
        nms_radius: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """提取 Top-K 候选端点，保留对 heatmap 的可求导梯度置信度 scores。支持 NMS 抑制临近重复聚簇。"""
        batch_size, _, height, width = heatmap.shape
        dtype = heatmap.dtype
        radius = self.nms_radius if nms_radius is None else nms_radius

        k = min(self.num_candidates, height * width)
        if radius > 0:
            kernel_size = 2 * radius + 1
            max_pooled = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=radius)
            is_peak = (heatmap == max_pooled)
            suppressed = heatmap * is_peak.to(dtype)
            flat_suppressed = suppressed.view(batch_size, height * width)
            _, topk_idx = torch.topk(flat_suppressed.detach(), k=k, dim=-1)
        else:
            flat_heat = heatmap.view(batch_size, height * width)
            _, topk_idx = torch.topk(flat_heat.detach(), k=k, dim=-1)

        scores = torch.gather(heatmap.view(batch_size, height * width), 1, topk_idx)  # (B, K)
        coord_y = (topk_idx // width).to(dtype)
        coord_x = (topk_idx % width).to(dtype)
        points = torch.stack([coord_x, coord_y], dim=-1)  # (B, K, 2)

        return scores, points

    def interpolate_and_sample_lines(
        self,
        points: torch.Tensor,
        df_map: torch.Tensor,
        af_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """在成对端点之间密集插值采样，并双线性采样对数距离场与角度场。"""
        batch_size, k, _ = points.shape
        _, _, height, width = df_map.shape
        device = points.device
        dtype = points.dtype

        p1 = points.unsqueeze(2)  # (B, K, 1, 2)
        p2 = points.unsqueeze(1)  # (B, 1, K, 2)
        diff = p2 - p1            # (B, K, K, 2)
        dist = torch.norm(diff, p=2, dim=-1)  # (B, K, K)

        t = torch.linspace(0.0, 1.0, self.num_line_samples, device=device, dtype=dtype).view(1, 1, 1, -1, 1)
        samples = p1.unsqueeze(3) + t * diff.unsqueeze(3)  # (B, K, K, M, 2)

        grid_x = 2.0 * samples[..., 0] / max(width - 1, 1) - 1.0
        grid_y = 2.0 * samples[..., 1] / max(height - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(batch_size, k * k, self.num_line_samples, 2)

        sampled_df = F.grid_sample(
            df_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).view(batch_size, k, k, self.num_line_samples)
        sampled_af = F.grid_sample(
            af_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).view(batch_size, k, k, self.num_line_samples)

        return sampled_df, sampled_af, dist, diff, p1, p2

    def compute_line_validity_matrix(
        self,
        sampled_df: torch.Tensor,
        sampled_af: torch.Tensor,
        dist: torch.Tensor,
    ) -> torch.Tensor:
        """判定两两端点构成的线段几何合法度矩阵 V_ij in [0, 1]。"""
        batch_size, k, _, _ = sampled_df.shape
        device = dist.device
        dtype = dist.dtype

        # 条件 1: 距离场均值 >= tau_df
        mean_df = sampled_df.mean(dim=-1)  # (B, K, K)
        v_df = torch.sigmoid((mean_df - self.tau_df) / 0.05)

        # 条件 2: 角度场极差 <= tau_ang (处理 0/π 周期性)
        af_spread = sampled_af.max(dim=-1)[0] - sampled_af.min(dim=-1)[0]
        af_spread_wrap = torch.minimum(af_spread, 1.0 - af_spread)
        v_ang = torch.sigmoid((self.tau_ang - af_spread_wrap) / 0.02)

        # 条件 3: 最小长度与排除自连
        min_len_mask = torch.sigmoid((dist - self.min_length) / 2.0)
        eye_mask = (1.0 - torch.eye(k, device=device, dtype=dtype)).unsqueeze(0)

        return v_df * v_ang * min_len_mask * eye_mask

    def line_geometric_loss(
        self,
        heatmap: torch.Tensor,
        df_map: torch.Tensor,
        af_map: torch.Tensor,
        points_gt: Any = None,
        semi: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """单视角端点与长线几何联合损失调度。"""
        _, _, height, width = heatmap.shape

        scores, points = self.extract_candidate_endpoints(heatmap)
        sampled_df, sampled_af, dist, diff, p1, p2 = self.interpolate_and_sample_lines(
            points, df_map, af_map
        )
        v_matrix = self.compute_line_validity_matrix(sampled_df, sampled_af, dist)

        l_orphan = self.orphan_endpoint_loss(scores, v_matrix)
        l_len = self.max_length_line_loss(scores, dist, v_matrix, height, width)
        l_prem = self.premature_ray_probing_loss(scores, dist, diff, p1, p2, v_matrix, df_map)
        l_det = self.detector_loss(semi if semi is not None else heatmap, points_gt)

        return {
            "l_orphan": l_orphan,
            "l_len": l_len,
            "l_prem": l_prem,
            "l_det": l_det,
        }

    def forward(
        self,
        pred0: dict[str, torch.Tensor],
        pred1: dict[str, torch.Tensor],
        homography: torch.Tensor | None = None,
        image0: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        points0: Any = None,
        points1: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """独立端点与直线几何损失前向传播。返回 {"loss": ..., "l_det": ..., "l_orphan": ..., "l_len": ..., "l_prem": ...}。"""
        del homography
        if image0 is None:
            image0 = kwargs.get("image0")
        if image1 is None:
            image1 = kwargs.get("image1")
        if points0 is None:
            points0 = kwargs.get("points0")
        if points1 is None:
            points1 = kwargs.get("points1")

        device = pred0["heatmap"].device

        if image0 is not None and image1 is not None:
            df0, af0 = image0[:, 3:4], image0[:, 4:5]
            df1, af1 = image1[:, 3:4], image1[:, 4:5]

            semi0 = pred0.get("semi")
            semi1 = pred1.get("semi")
            geom0 = self.line_geometric_loss(pred0["heatmap"], df0, af0, points0, semi=semi0)
            geom1 = self.line_geometric_loss(pred1["heatmap"], df1, af1, points1, semi=semi1)

            l_det = 0.5 * (geom0["l_det"] + geom1["l_det"])
            l_orphan = 0.5 * (geom0["l_orphan"] + geom1["l_orphan"])
            l_len = 0.5 * (geom0["l_len"] + geom1["l_len"])
            l_prem = 0.5 * (geom0["l_prem"] + geom1["l_prem"])
        else:
            l_det = torch.tensor(0.0, device=device)
            l_orphan = torch.tensor(0.0, device=device)
            l_len = torch.tensor(0.0, device=device)
            l_prem = torch.tensor(0.0, device=device)

        total_loss = (
            self.weight_det * l_det
            + self.weight_orphan * l_orphan
            + self.weight_len * l_len
            + self.weight_premature * l_prem
        )

        return {
            "loss": total_loss,
            "l_det": l_det.detach(),
            "l_orphan": l_orphan.detach(),
            "l_len": l_len.detach(),
            "l_prem": l_prem.detach(),
        }


# =============================================================================
# 3. 联合损失类 (Joint Loss Module: Descriptor + Detector/Line)
# =============================================================================

class SuperPointJointLoss(nn.Module):
    """SuperPoint 描述子与端点几何场统一联合损失模块。

    内部组合 SuperPointDescriptorLoss 与 SuperPointDetectorLoss，
    支持联合端到端训练与全套诊断指标输出。
    """

    def __init__(
        self,
        positive_weight: float = 250.0,
        positive_margin: float = 1.0,
        negative_margin: float = 0.2,
        weight_desc: float = 1.0,
        weight_det: float = 1.0,
        weight_orphan: float = 1.0,
        weight_len: float = 0.5,
        weight_premature: float = 0.5,
        tau_df: float = 0.55,
        tau_ang: float = 0.10,
        min_length: float = 8.0,
        num_candidates: int = 16,
        num_line_samples: int = 10,
        probe_dist: float = 4.0,
        det_pos_weight: float = 50.0,
        length_exponent: float = 1.5,
        nms_radius: int = 4,
        weight_intra: float = 1.0,
        weight_cross: float = 1.5,
        weight_inter: float = 0.5,
        weight_bg: float = 0.3,
        line_neg_margin: float = 0.2,
        bg_neg_margin: float = 0.1,
        num_line_points: int = 12,
        num_bg_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.desc_loss_module = SuperPointDescriptorLoss(
            positive_weight=positive_weight,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            weight_desc=weight_desc,
            weight_intra=weight_intra,
            weight_cross=weight_cross,
            weight_inter=weight_inter,
            weight_bg=weight_bg,
            line_neg_margin=line_neg_margin,
            bg_neg_margin=bg_neg_margin,
            num_line_points=num_line_points,
            num_bg_samples=num_bg_samples,
            **kwargs,
        )
        self.det_loss_module = SuperPointDetectorLoss(
            weight_det=weight_det,
            weight_orphan=weight_orphan,
            weight_len=weight_len,
            weight_premature=weight_premature,
            tau_df=tau_df,
            tau_ang=tau_ang,
            min_length=min_length,
            num_candidates=num_candidates,
            num_line_samples=num_line_samples,
            probe_dist=probe_dist,
            det_pos_weight=det_pos_weight,
            length_exponent=length_exponent,
            nms_radius=nms_radius,
            **kwargs,
        )

    # 代理属性
    @property
    def positive_weight(self) -> float:
        return self.desc_loss_module.positive_weight

    @property
    def positive_margin(self) -> float:
        return self.desc_loss_module.positive_margin

    @property
    def negative_margin(self) -> float:
        return self.desc_loss_module.negative_margin

    @property
    def weight_desc(self) -> float:
        return self.desc_loss_module.weight_desc

    @property
    def weight_det(self) -> float:
        return self.det_loss_module.weight_det

    @property
    def weight_orphan(self) -> float:
        return self.det_loss_module.weight_orphan

    @property
    def weight_len(self) -> float:
        return self.det_loss_module.weight_len

    @property
    def weight_premature(self) -> float:
        return self.det_loss_module.weight_premature

    @property
    def tau_df(self) -> float:
        return self.det_loss_module.tau_df

    @property
    def tau_ang(self) -> float:
        return self.det_loss_module.tau_ang

    @property
    def min_length(self) -> float:
        return self.det_loss_module.min_length

    @property
    def num_candidates(self) -> int:
        return self.det_loss_module.num_candidates

    @property
    def num_line_samples(self) -> int:
        return self.det_loss_module.num_line_samples

    @property
    def probe_dist(self) -> float:
        return self.det_loss_module.probe_dist

    @property
    def det_pos_weight(self) -> float:
        return self.det_loss_module.det_pos_weight

    @property
    def length_exponent(self) -> float:
        return self.det_loss_module.length_exponent

    # 代理成员方法
    def descriptor_loss(self, desc0: torch.Tensor, desc1: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        return self.desc_loss_module.descriptor_loss(desc0, desc1, homography)

    def detector_loss(self, heatmap: torch.Tensor, points_gt: Any) -> torch.Tensor:
        return self.det_loss_module.detector_loss(heatmap, points_gt)

    def orphan_endpoint_loss(self, scores: torch.Tensor, v_matrix: torch.Tensor) -> torch.Tensor:
        return self.det_loss_module.orphan_endpoint_loss(scores, v_matrix)

    def max_length_line_loss(
        self, scores: torch.Tensor, dist: torch.Tensor, v_matrix: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        return self.det_loss_module.max_length_line_loss(scores, dist, v_matrix, height, width)

    def premature_ray_probing_loss(
        self, scores: torch.Tensor, dist: torch.Tensor, diff: torch.Tensor,
        p1: torch.Tensor, p2: torch.Tensor, v_matrix: torch.Tensor, df_map: torch.Tensor
    ) -> torch.Tensor:
        return self.det_loss_module.premature_ray_probing_loss(scores, dist, diff, p1, p2, v_matrix, df_map)

    def extract_candidate_endpoints(self, heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.det_loss_module.extract_candidate_endpoints(heatmap)

    def interpolate_and_sample_lines(
        self, points: torch.Tensor, df_map: torch.Tensor, af_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.det_loss_module.interpolate_and_sample_lines(points, df_map, af_map)

    def compute_line_validity_matrix(
        self, sampled_df: torch.Tensor, sampled_af: torch.Tensor, dist: torch.Tensor
    ) -> torch.Tensor:
        return self.det_loss_module.compute_line_validity_matrix(sampled_df, sampled_af, dist)

    def line_geometric_loss(
        self, heatmap: torch.Tensor, df_map: torch.Tensor, af_map: torch.Tensor, points_gt: Any = None, semi: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        return self.det_loss_module.line_geometric_loss(heatmap, df_map, af_map, points_gt, semi=semi)

    # 别名映射
    loss_descriptor = descriptor_loss
    loss_detector = detector_loss
    loss_orphan = orphan_endpoint_loss
    loss_max_length = max_length_line_loss
    loss_premature = premature_ray_probing_loss

    def forward(
        self,
        pred0: dict[str, torch.Tensor],
        pred1: dict[str, torch.Tensor],
        homography: torch.Tensor,
        image0: torch.Tensor | None = None,
        image1: torch.Tensor | None = None,
        points0: Any = None,
        points1: Any = None,
        lines0: Any = None,
        lines1: Any = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """联合损失前向计算。同时返回各分项数值与总和 loss。"""
        if lines0 is None:
            lines0 = kwargs.get("lines0")
        if lines1 is None:
            lines1 = kwargs.get("lines1")
        desc_res = self.desc_loss_module(
            pred0, pred1, homography, lines0=lines0, lines1=lines1, image0=image0, image1=image1, **kwargs
        )
        det_res = self.det_loss_module(
            pred0, pred1, homography, image0=image0, image1=image1, points0=points0, points1=points1, **kwargs
        )

        total_loss = desc_res["loss"] + det_res["loss"]

        out = {
            "loss": total_loss,
            "ld": desc_res["ld"],
            "l_det": det_res["l_det"],
            "l_orphan": det_res["l_orphan"],
            "l_len": det_res["l_len"],
            "l_prem": det_res["l_prem"],
        }
        for k in ("l_intra", "l_cross", "l_inter", "l_bg"):
            if k in desc_res:
                out[k] = desc_res[k]
        return out


# 别名导出与向后兼容
SuperPointLoss = SuperPointJointLoss
SuperPointLineLoss = SuperPointJointLoss


# =============================================================================
# 模块级向后兼容快捷函数 (Forwarding to SuperPointJointLoss)
# =============================================================================

def descriptor_hinge_loss(
    desc0: torch.Tensor,
    desc1: torch.Tensor,
    homography: torch.Tensor,
    positive_weight: float = 250.0,
    positive_margin: float = 1.0,
    negative_margin: float = 0.2,
) -> torch.Tensor:
    module = SuperPointDescriptorLoss(
        positive_weight=positive_weight,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )
    return module.descriptor_loss(desc0, desc1, homography)


def detector_bce_loss(
    heatmap: torch.Tensor,
    points_gt: Any,
    pos_weight: float = 50.0,
) -> torch.Tensor:
    module = SuperPointDetectorLoss(det_pos_weight=pos_weight)
    return module.detector_loss(heatmap, points_gt)


def orphan_endpoint_loss(
    scores: torch.Tensor,
    v_matrix: torch.Tensor,
) -> torch.Tensor:
    module = SuperPointDetectorLoss()
    return module.orphan_endpoint_loss(scores, v_matrix)


def max_length_line_loss(
    scores: torch.Tensor,
    dist: torch.Tensor,
    v_matrix: torch.Tensor,
    height: int,
    width: int,
    exponent: float = 1.5,
) -> torch.Tensor:
    module = SuperPointDetectorLoss(length_exponent=exponent)
    return module.max_length_line_loss(scores, dist, v_matrix, height, width)


def premature_ray_probing_loss(
    scores: torch.Tensor,
    dist: torch.Tensor,
    diff: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    v_matrix: torch.Tensor,
    df_map: torch.Tensor,
    tau_df: float = 0.55,
    probe_dist: float = 4.0,
) -> torch.Tensor:
    module = SuperPointDetectorLoss(tau_df=tau_df, probe_dist=probe_dist)
    return module.premature_ray_probing_loss(scores, dist, diff, p1, p2, v_matrix, df_map)


def extract_candidate_endpoints(
    heatmap: torch.Tensor,
    num_candidates: int = 16,
    nms_radius: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = SuperPointDetectorLoss(num_candidates=num_candidates, nms_radius=nms_radius)
    return module.extract_candidate_endpoints(heatmap)


def interpolate_and_sample_lines(
    points: torch.Tensor,
    df_map: torch.Tensor,
    af_map: torch.Tensor,
    num_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    module = SuperPointDetectorLoss(num_line_samples=num_samples)
    return module.interpolate_and_sample_lines(points, df_map, af_map)


def compute_line_validity_matrix(
    sampled_df: torch.Tensor,
    sampled_af: torch.Tensor,
    dist: torch.Tensor,
    tau_df: float = 0.55,
    tau_ang: float = 0.10,
    min_length: float = 8.0,
) -> torch.Tensor:
    module = SuperPointDetectorLoss(tau_df=tau_df, tau_ang=tau_ang, min_length=min_length)
    return module.compute_line_validity_matrix(sampled_df, sampled_af, dist)


# =============================================================================
# 基础坐标与单应投影辅助函数
# =============================================================================

def _cell_centers(cells_h: int, cells_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """计算 8x8 网格中心在原图像素坐标系下的绝对坐标 (x, y)。"""
    rows = (torch.arange(cells_h, device=device, dtype=dtype) + 0.5) * 8.0 - 0.5
    cols = (torch.arange(cells_w, device=device, dtype=dtype) + 0.5) * 8.0 - 0.5
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def _apply_homography(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """将 2D 坐标点集通过 3x3 单应矩阵投影到目标图像坐标系。"""
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=-1)
    mapped = torch.einsum("bij,bmj->bmi", homography, homo)
    w = mapped[..., 2:3].clamp(min=1e-6)
    return mapped[..., :2] / w


__all__ = [
    "SuperPointDescriptorLoss",
    "SuperPointDetectorLoss",
    "SuperPointJointLoss",
    "SuperPointLoss",
    "SuperPointLineLoss",
    "sample_descriptors_along_lines",
    "sample_background_descriptors",
    "descriptor_hinge_loss",
    "detector_bce_loss",
    "orphan_endpoint_loss",
    "max_length_line_loss",
    "premature_ray_probing_loss",
    "extract_candidate_endpoints",
    "interpolate_and_sample_lines",
    "compute_line_validity_matrix",
]
