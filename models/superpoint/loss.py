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

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 描述子度量合页损失类 (Descriptor Metric Hinge Loss)
# =============================================================================

class SuperPointDescriptorLoss(nn.Module):
    """SuperPoint 描述子度量合页损失模块。

    针对跨视角半稠密特征图计算余弦相似度合页损失:
    - 距离单应投影位置 <= 8px 判定为同名正样本对，拉近至 positive_margin (1.0)，加权 positive_weight (250x);
    - 其余判定为负样本对，推远至 negative_margin (0.2) 以下。
    """

    def __init__(
        self,
        positive_weight: float = 250.0,
        positive_margin: float = 1.0,
        negative_margin: float = 0.2,
        weight_desc: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.positive_weight = float(positive_weight)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.weight_desc = float(weight_desc)

    def descriptor_loss(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        homography: torch.Tensor,
    ) -> torch.Tensor:
        """计算两视角半稠密网格描述子之间的合页度量损失。"""
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

    def forward(
        self,
        pred0: dict[str, torch.Tensor],
        pred1: dict[str, torch.Tensor],
        homography: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """独立描述子损失前向传播。返回 {"loss": ..., "ld": ...}。"""
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

    # -------------------------------------------------------------------------
    # 损失函数类成员
    # -------------------------------------------------------------------------

    def detector_loss(
        self,
        heatmap: torch.Tensor,
        points_gt: Any,
    ) -> torch.Tensor:
        """【类成员损失项 1】端点检测加权二值交叉熵损失 (Detector BCE Loss)."""
        if points_gt is None:
            return torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype)

        batch_size, _, height, width = heatmap.shape
        target = torch.zeros_like(heatmap)

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

        has_any_gt = False
        for b, pts in enumerate(pts_list):
            if b >= batch_size or pts is None or len(pts) == 0:
                continue
            has_any_gt = True
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

        if not has_any_gt:
            return torch.tensor(0.0, device=heatmap.device, dtype=heatmap.dtype)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """提取 Top-K 候选端点，保留对 heatmap 的可求导梯度置信度 scores。"""
        batch_size, _, height, width = heatmap.shape
        dtype = heatmap.dtype

        k = min(self.num_candidates, height * width)
        flat_heat = heatmap.view(batch_size, height * width)
        _, topk_idx = torch.topk(flat_heat.detach(), k=k, dim=-1)

        scores = torch.gather(flat_heat, 1, topk_idx)  # (B, K)
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
        l_det = self.detector_loss(heatmap, points_gt)

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

            geom0 = self.line_geometric_loss(pred0["heatmap"], df0, af0, points0)
            geom1 = self.line_geometric_loss(pred1["heatmap"], df1, af1, points1)

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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.desc_loss_module = SuperPointDescriptorLoss(
            positive_weight=positive_weight,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            weight_desc=weight_desc,
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
        self, heatmap: torch.Tensor, df_map: torch.Tensor, af_map: torch.Tensor, points_gt: Any = None
    ) -> dict[str, torch.Tensor]:
        return self.det_loss_module.line_geometric_loss(heatmap, df_map, af_map, points_gt)

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
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """联合损失前向计算。同时返回各分项数值与总和 loss。"""
        desc_res = self.desc_loss_module(pred0, pred1, homography, **kwargs)
        det_res = self.det_loss_module(
            pred0, pred1, homography, image0=image0, image1=image1, points0=points0, points1=points1, **kwargs
        )

        total_loss = desc_res["loss"] + det_res["loss"]

        return {
            "loss": total_loss,
            "ld": desc_res["ld"],
            "l_det": det_res["l_det"],
            "l_orphan": det_res["l_orphan"],
            "l_len": det_res["l_len"],
            "l_prem": det_res["l_prem"],
        }


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
) -> tuple[torch.Tensor, torch.Tensor]:
    module = SuperPointDetectorLoss(num_candidates=num_candidates)
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
    "descriptor_hinge_loss",
    "detector_bce_loss",
    "orphan_endpoint_loss",
    "max_length_line_loss",
    "premature_ray_probing_loss",
    "extract_candidate_endpoints",
    "interpolate_and_sample_lines",
    "compute_line_validity_matrix",
]
