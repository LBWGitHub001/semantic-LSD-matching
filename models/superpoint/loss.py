"""SuperPoint 联合损失函数体系 (Unified Loss System for SuperPoint).

涵盖 5 大核心损失组件:
1. L_desc (描述子度量合页损失 / Metric Hinge Loss):
   同名点在特征超球流形上内积逼近 1.0，异名点推离到 0.2 以下 (正样本 250 倍加权)。
2. L_det (端点检测正向引导损失 / Detector BCE Loss):
   若提供角点/端点真值，提供温和的正向监督锚点，防止网络陷入全 0 预测捷径。
3. L_orphan (孤立端点几何合法性惩罚 / Orphan Endpoint Penalty):
   两个端点连线必须在对数距离场上平均值 >= tau_df，且角度场极差 <= tau_ang，满足者为合法线段。
   未能参与任何合法线段的孤立端点，将其置信度强制惩罚归零。
4. L_max_length (超线性长线奖励 / Super-linear Max Length Reward):
   合法线段长度越长收益越高，采用 (d / D_max)^1.5 超线性加权，驱使端点跨越整条连续线段，
   彻底打破短线局部最优陷阱。
5. L_premature (端点外推射线探针惩罚 / Premature Ray Probing Penalty):
   沿线段两端外推 4 像素放置探针，若探针处依然处于线脊上 (未走完提前终止)，施加惩罚，
   强力推动端点直至真正的物理转角或边界。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointLoss(nn.Module):
    """SuperPoint 描述子、端点检测与直线几何场联合损失函数。

    参数说明:
        positive_weight: 描述子正样本对加权 (默认 250.0)。
        positive_margin: 描述子正样本间隔 (默认 1.0)。
        negative_margin: 描述子负样本间隔 (默认 0.2)。
        weight_desc: 描述子损失总权重 (默认 1.0)。
        weight_det: 端点检测正向引导损失权重 (默认 1.0)。
        weight_orphan: 孤立端点几何惩罚权重 (默认 1.0)。
        weight_len: 超线性长线奖励权重 (默认 0.5)。
        weight_premature: 射线外推过早终止惩罚权重 (默认 0.5)。
        tau_df: 距离场合法性阈值 (默认 0.55，在对数归一化场下 1px 以内通常 > 0.61)。
        tau_ang: 角度场一致性阈值 (默认 0.10，相当于约 18 度角内)。
        min_length: 构成合法线段的最小像素长度 (默认 8.0 像素)。
        num_candidates: 几何场约束中采样的候选端点数 K (默认 16)。
        num_line_samples: 沿线段均匀采样的插值点数 M (默认 10)。
        probe_dist: 端点外推探针距离 (默认 4.0 像素)。
        det_pos_weight: 端点检测 BCE 损失正样本加权 (默认 50.0)。
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.positive_weight = float(positive_weight)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)

        self.weight_desc = float(weight_desc)
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
        """前向联合损失计算。

        支持兼容纯描述子模式与全套端点-长线几何损失模式:
        - 若未提供 image0/image1，仅计算经典描述子合页损失 L_desc；
        - 若提供 image0/image1 (5通道: RGB+DF+AF)，则联合计算 L_desc, L_det, L_orphan, L_len, L_prem。
        """
        # 支持 kwargs 兼容传参
        if image0 is None:
            image0 = kwargs.get("image0")
        if image1 is None:
            image1 = kwargs.get("image1")
        if points0 is None:
            points0 = kwargs.get("points0")
        if points1 is None:
            points1 = kwargs.get("points1")

        device = homography.device

        # 1. 描述子合页损失 L_desc
        l_desc = self._descriptor_loss(pred0["desc"], pred1["desc"], homography)

        # 2. 端点与直线几何损失
        has_heatmap = "heatmap" in pred0 and "heatmap" in pred1
        has_images = image0 is not None and image1 is not None

        if has_heatmap and has_images:
            df0, af0 = image0[:, 3:4], image0[:, 4:5]
            df1, af1 = image1[:, 3:4], image1[:, 4:5]

            geom0 = self._line_geometric_loss(pred0["heatmap"], df0, af0, points0)
            geom1 = self._line_geometric_loss(pred1["heatmap"], df1, af1, points1)

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
            self.weight_desc * l_desc
            + self.weight_det * l_det
            + self.weight_orphan * l_orphan
            + self.weight_len * l_len
            + self.weight_premature * l_prem
        )

        return {
            "loss": total_loss,
            "ld": l_desc.detach(),
            "l_det": l_det.detach(),
            "l_orphan": l_orphan.detach(),
            "l_len": l_len.detach(),
            "l_prem": l_prem.detach(),
        }

    def _descriptor_loss(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        homography: torch.Tensor,
    ) -> torch.Tensor:
        """向量化计算全图两两网格单元间的 Hinge 损失。"""
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

    def _line_geometric_loss(
        self,
        heatmap: torch.Tensor,
        df_map: torch.Tensor,
        af_map: torch.Tensor,
        points_gt: Any = None,
    ) -> dict[str, torch.Tensor]:
        """计算单视角下的端点检测、孤立惩罚、长线奖励与射线外推探针损失。

        输入:
            heatmap: (B, 1, H, W) 端点预测概率图，值域在 [0, 1]
            df_map: (B, 1, H, W) 归一化对数距离场，线脊峰值为 1.0，背景为 0.0
            af_map: (B, 1, H, W) 归一化角度场，值域在 [0, 1]，对应 [0, π)
            points_gt: 角点/端点真值坐标 (N, 2) 或 list
        """
        batch_size, _, height, width = heatmap.shape
        device = heatmap.device
        dtype = heatmap.dtype

        # -------------------------------------------------------------
        # 1. 提取 top-K 候选端点，并保持可求导的分数 scores
        # -------------------------------------------------------------
        k = min(self.num_candidates, height * width)
        flat_heat = heatmap.view(batch_size, height * width)
        topk_val, topk_idx = torch.topk(flat_heat.detach(), k=k, dim=-1)
        # 通过 gather 保留可导置信度 scores
        scores = torch.gather(flat_heat, 1, topk_idx)  # (B, K)

        coord_y = (topk_idx // width).to(dtype)
        coord_x = (topk_idx % width).to(dtype)
        points = torch.stack([coord_x, coord_y], dim=-1)  # (B, K, 2)

        # -------------------------------------------------------------
        # 2. 全连接端点对构建与沿线插值采样 (Line Interpolation)
        # -------------------------------------------------------------
        p1 = points.unsqueeze(2)  # (B, K, 1, 2)
        p2 = points.unsqueeze(1)  # (B, 1, K, 2)
        diff = p2 - p1            # (B, K, K, 2)
        dist = torch.norm(diff, p=2, dim=-1)  # (B, K, K)

        # 沿线段均匀插值 M 个采样点
        m = self.num_line_samples
        t = torch.linspace(0.0, 1.0, m, device=device, dtype=dtype).view(1, 1, 1, m, 1)
        samples = p1.unsqueeze(3) + t * diff.unsqueeze(3)  # (B, K, K, M, 2)

        # 归一化到 grid_sample [-1, 1] 坐标系
        grid_x = 2.0 * samples[..., 0] / max(width - 1, 1) - 1.0
        grid_y = 2.0 * samples[..., 1] / max(height - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(batch_size, k * k, m, 2)

        sampled_df = F.grid_sample(
            df_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).view(batch_size, k, k, m)
        sampled_af = F.grid_sample(
            af_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).view(batch_size, k, k, m)

        # -------------------------------------------------------------
        # 3. 直线合法度矩阵 V_ij 计算 (Validity Matrix)
        # -------------------------------------------------------------
        # 条件 1: 距离场均值 >= tau_df
        mean_df = sampled_df.mean(dim=-1)  # (B, K, K)
        v_df = torch.sigmoid((mean_df - self.tau_df) / 0.05)

        # 条件 2: 角度场极差 <= tau_ang (处理 0/π 周期性)
        af_spread = sampled_af.max(dim=-1)[0] - sampled_af.min(dim=-1)[0]
        af_spread_wrap = torch.minimum(af_spread, 1.0 - af_spread)
        v_ang = torch.sigmoid((self.tau_ang - af_spread_wrap) / 0.02)

        # 排除自连 (i == j) 以及过短线段 (< min_length)
        min_len_mask = torch.sigmoid((dist - self.min_length) / 2.0)
        eye_mask = (1.0 - torch.eye(k, device=device, dtype=dtype)).unsqueeze(0)

        # 综合合法度 V_ij in [0, 1]
        v_matrix = v_df * v_ang * min_len_mask * eye_mask

        # -------------------------------------------------------------
        # 4. L_orphan: 孤立端点惩罚
        # -------------------------------------------------------------
        # 每个端点在所有线段中的最大合法度支持
        s_support = v_matrix.max(dim=-1)[0]  # (B, K)
        l_orphan = (scores * (1.0 - s_support)).mean()

        # -------------------------------------------------------------
        # 5. L_max_length: 超线性长线收益奖励 (负奖励作为损失)
        # -------------------------------------------------------------
        diag = math.sqrt(float(height**2 + width**2))
        reward_len = (dist / max(diag, 1.0)) ** 1.5
        pair_scores = scores.unsqueeze(2) * scores.unsqueeze(1)  # (B, K, K)

        denom = v_matrix.sum(dim=(-1, -2)) + 1e-4
        l_len = - ((v_matrix * pair_scores * reward_len).sum(dim=(-1, -2)) / denom).mean()

        # -------------------------------------------------------------
        # 6. L_premature: 端点外推射线探针惩罚 (推动端点直到真正转角)
        # -------------------------------------------------------------
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
        l_prem = ((v_matrix * pair_scores * (ext_j + ext_i)).sum(dim=(-1, -2)) / denom).mean()

        # -------------------------------------------------------------
        # 7. L_det: 端点检测正向引导损失 (锚定真值，防止零塌陷)
        # -------------------------------------------------------------
        l_det = self._detector_loss(heatmap, points_gt)

        return {
            "l_orphan": l_orphan,
            "l_len": l_len,
            "l_prem": l_prem,
            "l_det": l_det,
        }

    def _detector_loss(self, heatmap: torch.Tensor, points_gt: Any) -> torch.Tensor:
        """端点检测加权二值交叉熵损失。"""
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


# 导出别名保持兼容
SuperPointLineLoss = SuperPointLoss


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
