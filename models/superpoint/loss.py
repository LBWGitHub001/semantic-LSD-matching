"""SuperPoint 逐步损失训练体系 (Direct Tensor Math, Progressive Stage Decoupled Loss).

数学损失项:
1. 端点检测加权 BCE 损失 (l_det)
2. 孤立端点惩罚 (l_orphan)
3. 超线性长线收益奖励 (l_len)
4. 射线探针早退惩罚 (l_prem)
5. 线内像素一致性损失 (l_intra)
6. 跨视角同名直线原型对齐损失 (l_cross)
7. 异线原型推离对比损失 (l_inter)
8. 背景像素对比推离损失 (l_bg)
9. 经典关键点网格合页度量损失 (l_desc / descriptor_loss)
"""

from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 0. 逐步训练阶段权重定义 (Curriculum / Progressive Stages)
# =============================================================================

STAGE_LOSS_WEIGHTS: dict[str, dict[str, float]] = {
    # 阶段 0: 纯直线场检测器训练阶段 (检测层仅保留: 1. 端点连线上每一个像素和直线场的距离比较; 2. 角度场最大最小极差比较)
    "stage_line_detector": {
        "weight_dist": 1.0, "weight_ang": 1.0,
        "weight_det": 0.0, "weight_orphan": 0.0, "weight_len": 0.0, "weight_premature": 0.0,
        "weight_intra": 0.0, "weight_cross": 0.0, "weight_inter": 0.0, "weight_bg": 0.0, "weight_desc": 0.0,
    },
    # 阶段 1: 端点检测预热 (只激活真值加权 BCE)
    "stage1_det_warmup": {
        "weight_dist": 0.0, "weight_ang": 0.0,
        "weight_det": 1.0, "weight_orphan": 0.0, "weight_len": 0.0, "weight_premature": 0.0,
        "weight_intra": 0.0, "weight_cross": 0.0, "weight_inter": 0.0, "weight_bg": 0.0, "weight_desc": 0.0,
    },
    # 阶段 2: 几何拓扑注入 (端点 + 孤立惩罚 + 超线性长线奖励 + 探针惩罚)
    "stage2_det_geom": {
        "weight_dist": 0.0, "weight_ang": 0.0,
        "weight_det": 1.0, "weight_orphan": 1.0, "weight_len": 0.5, "weight_premature": 0.5,
        "weight_intra": 0.0, "weight_cross": 0.0, "weight_inter": 0.0, "weight_bg": 0.0, "weight_desc": 0.0,
    },
    # 阶段 3: 描述子线内聚类与背景分离 (同线像素相似 + 背景推离)
    "stage3_desc_intra": {
        "weight_dist": 0.0, "weight_ang": 0.0,
        "weight_det": 0.0, "weight_orphan": 0.0, "weight_len": 0.0, "weight_premature": 0.0,
        "weight_intra": 1.0, "weight_cross": 0.0, "weight_inter": 0.0, "weight_bg": 0.3, "weight_desc": 1.0,
    },
    # 阶段 4: 跨视角直线对齐与异线推离 (跨图同名线一致 + 异线正交推离)
    "stage4_desc_cross": {
        "weight_dist": 0.0, "weight_ang": 0.0,
        "weight_det": 0.0, "weight_orphan": 0.0, "weight_len": 0.0, "weight_premature": 0.0,
        "weight_intra": 1.0, "weight_cross": 1.5, "weight_inter": 0.5, "weight_bg": 0.3, "weight_desc": 1.0,
    },
    # 阶段 5: 全任务联合微调 (全量检测 + 全量描述子)
    "stage5_joint": {
        "weight_dist": 1.0, "weight_ang": 1.0,
        "weight_det": 1.0, "weight_orphan": 1.0, "weight_len": 0.5, "weight_premature": 0.5,
        "weight_intra": 1.0, "weight_cross": 1.5, "weight_inter": 0.5, "weight_bg": 0.3, "weight_desc": 1.0,
    },
}

# 默认检测阶段映射为纯找线检测器
STAGE_LOSS_WEIGHTS["detector"] = STAGE_LOSS_WEIGHTS["stage_line_detector"]
STAGE_LOSS_WEIGHTS["descriptor"] = STAGE_LOSS_WEIGHTS["stage4_desc_cross"]
STAGE_LOSS_WEIGHTS["joint"] = STAGE_LOSS_WEIGHTS["stage5_joint"]


# =============================================================================
# 1. 纯粹张量数学算子 (Pure Tensor Math Operations - 零防御性编程)
# =============================================================================

def sample_descriptors_along_lines(
    descriptors: torch.Tensor,
    lines: torch.Tensor,
    num_samples: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿线均匀插值采样单位归一化描述子，并计算每条线的单位特征原型。"""
    lines = torch.as_tensor(lines, dtype=torch.float32, device=descriptors.device)
    if lines.dim() == 2:
        lines = lines.unsqueeze(0).expand(descriptors.shape[0], -1, -1)
    batch_size, num_lines, _ = lines.shape
    if num_lines == 0:
        c = descriptors.shape[1]
        zero_samp = torch.zeros((batch_size, 0, num_samples, c), device=descriptors.device, dtype=descriptors.dtype)
        zero_proto = torch.zeros((batch_size, 0, c), device=descriptors.device, dtype=descriptors.dtype)
        return zero_samp, zero_proto

    _, _, height, width = descriptors.shape
    p1 = lines[..., :2].unsqueeze(2)  # (B, K, 1, 2)
    p2 = lines[..., 2:].unsqueeze(2)  # (B, K, 1, 2)
    t = torch.linspace(0.0, 1.0, num_samples, device=descriptors.device, dtype=descriptors.dtype).view(1, 1, num_samples, 1)
    pts = p1 + t * (p2 - p1)          # (B, K, S, 2)

    gx = 2.0 * pts[..., 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * pts[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)

    sampled = F.grid_sample(descriptors, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = F.normalize(sampled.permute(0, 2, 3, 1), p=2, dim=-1)  # (B, K, S, C)
    proto = F.normalize(sampled.mean(dim=2), p=2, dim=-1)           # (B, K, C)
    return sampled, proto


def sample_background_descriptors(
    descriptors: torch.Tensor,
    df_or_mask: torch.Tensor | None = None,
    num_samples: int = 64,
) -> torch.Tensor:
    """在背景非直线区域采样像素描述子。"""
    batch_size, c, height, width = descriptors.shape
    device = descriptors.device
    if df_or_mask is not None and df_or_mask.shape[-2:] == (height, width):
        prob = (df_or_mask < 0.25).float().view(batch_size, -1)
    else:
        prob = torch.ones((batch_size, height * width), device=device, dtype=descriptors.dtype)

    indices = torch.multinomial(prob.clamp(min=1e-5), num_samples=num_samples, replacement=True)
    flat_desc = descriptors.permute(0, 2, 3, 1).reshape(batch_size, height * width, c)
    bg_desc = torch.gather(flat_desc, 1, indices.unsqueeze(-1).expand(-1, -1, c))
    return F.normalize(bg_desc, p=2, dim=-1)


def loss_intra_line(sampled: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """线内像素一致性损失 (同一条直线上像素描述子与原型拉近至 1.0)。"""
    cos = (sampled * proto.unsqueeze(2)).sum(dim=-1)
    return (1.0 - cos).mean()


def loss_cross_line(
    proto0: torch.Tensor, proto1: torch.Tensor,
    sampled0: torch.Tensor, sampled1: torch.Tensor,
) -> torch.Tensor:
    """跨视角同名直线一致性损失 (跨图同名线原型及对应像素对齐)。"""
    cos_proto = (proto0 * proto1).sum(dim=-1)
    cos_pts = (sampled0 * sampled1).sum(dim=-1)
    return 0.5 * ((1.0 - cos_proto).mean() + (1.0 - cos_pts).mean())


def loss_inter_line(proto0: torch.Tensor, proto1: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """异线原型相互推离 (非对角线样本点积推远至 <= margin)。"""
    dot = torch.bmm(proto0, proto1.transpose(1, 2))  # (B, K, K)
    k = dot.shape[1]
    if k <= 1:
        return torch.tensor(0.0, device=dot.device, dtype=dot.dtype)
    eye = torch.eye(k, device=dot.device, dtype=torch.bool).unsqueeze(0)
    off_diag = dot.masked_select(~eye)
    return F.relu(off_diag - margin).mean()


def loss_bg_contrastive(bg_desc: torch.Tensor, proto: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """背景负样本对比推离损失 (背景像素与直线原型相似度推离至 <= margin)。"""
    sim = torch.bmm(bg_desc, proto.transpose(1, 2))  # (B, N_bg, K)
    return F.relu(sim - margin).mean()


def loss_line_distance(
    sampled_df: torch.Tensor,
    pair_scores: torch.Tensor,
    tau_df: float = 0.55,
) -> torch.Tensor:
    """端点连线上每一个像素和直线距离场的比较损失。

    sampled_df: (B, K, K, S), 沿两候选端点连线均匀采样的距离场数值 (1.0 为线脊，0.0 为背景)
    pair_scores: (B, K, K), 两候选端点置信度乘积 s_i * s_j
    tau_df: 判定为直线的距离场阈值 (默认 0.55)
    """
    # 沿线每个像素偏离线脊 (1.0) 的距离误差: (1.0 - sampled_df)
    pixel_dist_err = 1.0 - sampled_df  # (B, K, K, S)
    line_dist_err = pixel_dist_err.mean(dim=-1)  # (B, K, K)

    # 真实直线目标 (沿线均值处于线脊上)
    target_df = (sampled_df.mean(dim=-1) >= tau_df).float()

    # 偏离直线的连线惩罚高置信度；处于直线上的连线鼓励提高置信度并降低像素距离误差
    loss = pair_scores * line_dist_err + (1.0 - pair_scores) * target_df
    return loss.mean()


def loss_line_angle(
    sampled_af: torch.Tensor,
    pair_scores: torch.Tensor,
    tau_ang: float = 0.10,
) -> torch.Tensor:
    """沿端点连线角度场的角度最大值与最小值之差的比较损失。

    sampled_af: (B, K, K, S), 沿两候选端点连线均匀采样的角度场数值 (归一化至 [0, 1])
    pair_scores: (B, K, K), 两候选端点置信度乘积 s_i * s_j
    tau_ang: 角度场切线方向一致性极差阈值 (默认 0.10)
    """
    # 角度场沿线最大值与最小值之差: max - min
    af_max = sampled_af.max(dim=-1)[0]  # (B, K, K)
    af_min = sampled_af.min(dim=-1)[0]  # (B, K, K)
    ang_spread = af_max - af_min        # (B, K, K)

    # 真实直线目标 (沿线角度变化在极差阈值内)
    target_af = (ang_spread <= tau_ang).float()

    # 角度剧烈波动的连线惩罚高置信度；切线一致的直线鼓励提高置信度并压缩角度极差
    loss = pair_scores * ang_spread + (1.0 - pair_scores) * target_af
    return loss.mean()


def loss_detector_bce(heatmap: torch.Tensor, target: torch.Tensor, pos_weight: float = 50.0) -> torch.Tensor:
    """端点概率图加权二值交叉熵。"""
    eps = 1e-6
    return -(pos_weight * target * torch.log(heatmap + eps) + (1.0 - target) * torch.log(1.0 - heatmap + eps)).mean()


def loss_orphan_endpoint(scores: torch.Tensor, v_matrix: torch.Tensor) -> torch.Tensor:
    """孤立端点惩罚损失 (惩罚未连成任何合法直线的高置信端点)。"""
    s_support = v_matrix.max(dim=-1)[0]
    return (scores * (1.0 - s_support)).mean()


def loss_max_length_line(
    scores: torch.Tensor, dist: torch.Tensor, v_matrix: torch.Tensor,
    height: int, width: int, exponent: float = 1.5,
) -> torch.Tensor:
    """超线性长线收益奖励损失。"""
    diag = math.sqrt(float(height**2 + width**2))
    reward = (dist / max(diag, 1.0)) ** exponent
    pair_scores = scores.unsqueeze(2) * scores.unsqueeze(1)
    denom = v_matrix.sum(dim=(-1, -2)) + 1e-4
    return - ((v_matrix * pair_scores * reward).sum(dim=(-1, -2)) / denom).mean()


def loss_premature_probing(
    scores: torch.Tensor, dist: torch.Tensor, diff: torch.Tensor,
    p1: torch.Tensor, p2: torch.Tensor, v_matrix: torch.Tensor,
    df_map: torch.Tensor, probe_dist: float = 4.0, tau_df: float = 0.55,
) -> torch.Tensor:
    """端点外推射线探针早退惩罚。"""
    batch_size, k, _, _ = p1.shape
    unit_dir = diff / dist.unsqueeze(-1).clamp(min=1.0)
    q1 = p1 - probe_dist * unit_dir
    q2 = p2 + probe_dist * unit_dir

    _, _, height, width = df_map.shape
    g1 = torch.stack([2.0 * q1[..., 0] / max(width - 1, 1) - 1.0, 2.0 * q1[..., 1] / max(height - 1, 1) - 1.0], dim=-1)
    g2 = torch.stack([2.0 * q2[..., 0] / max(width - 1, 1) - 1.0, 2.0 * q2[..., 1] / max(height - 1, 1) - 1.0], dim=-1)

    s1 = F.grid_sample(df_map, g1.view(batch_size, k * k, 1, 2), mode="bilinear", padding_mode="zeros", align_corners=True)
    s2 = F.grid_sample(df_map, g2.view(batch_size, k * k, 1, 2), mode="bilinear", padding_mode="zeros", align_corners=True)
    s1 = s1.view(batch_size, k, k)
    s2 = s2.view(batch_size, k, k)

    premature = ((s1 >= tau_df) | (s2 >= tau_df)).float()
    pair_scores = scores.unsqueeze(2) * scores.unsqueeze(1)
    denom = v_matrix.sum(dim=(-1, -2)) + 1e-4
    return ((v_matrix * pair_scores * premature).sum(dim=(-1, -2)) / denom).mean()


# =============================================================================
# 2. 端点与线几何采样算子 (Geometric Sampling & Candidate Operators)
# =============================================================================

def extract_candidate_endpoints(
    heatmap: torch.Tensor,
    num_candidates: int = 16,
    nms_radius: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """通过 MaxPool2D NMS 提取局部极值候选端点及其置信度。"""
    batch_size, _, height, width = heatmap.shape
    pad = nms_radius
    pooled = F.max_pool2d(heatmap, kernel_size=2 * pad + 1, stride=1, padding=pad)
    nms_mask = (heatmap == pooled) & (heatmap > 0.01)
    suppressed = heatmap * nms_mask.float()

    flat = suppressed.view(batch_size, -1)
    k = min(num_candidates, flat.shape[1])
    top_scores, top_idx = torch.topk(flat, k=k, dim=1)

    ys = top_idx // width
    xs = top_idx % width
    points = torch.stack([xs.float(), ys.float()], dim=-1)
    return top_scores, points


def interpolate_and_sample_lines(
    points: torch.Tensor,
    df_map: torch.Tensor,
    af_map: torch.Tensor,
    num_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """两两候选端点组合插值采样距离场与角度场。"""
    batch_size, k, _ = points.shape
    _, _, height, width = df_map.shape
    device = points.device

    p1 = points.unsqueeze(2).expand(-1, -1, k, -1)  # (B, K, K, 2)
    p2 = points.unsqueeze(1).expand(-1, -1, k, -1)  # (B, K, K, 2)
    diff = p2 - p1
    dist = torch.norm(diff, p=2, dim=-1)

    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=points.dtype).view(1, 1, 1, num_samples, 1)
    pts = p1.unsqueeze(3) + t * diff.unsqueeze(3)  # (B, K, K, num_samples, 2)

    gx = 2.0 * pts[..., 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * pts[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(batch_size, k * k, num_samples, 2)

    sdf = F.grid_sample(df_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    saf = F.grid_sample(af_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    sdf = sdf.view(batch_size, k, k, num_samples)
    saf = saf.view(batch_size, k, k, num_samples)
    return sdf, saf, dist, diff, p1, p2


def compute_line_validity_matrix(
    sampled_df: torch.Tensor,
    sampled_af: torch.Tensor,
    dist: torch.Tensor,
    tau_df: float = 0.55,
    tau_ang: float = 0.10,
    min_length: float = 8.0,
) -> torch.Tensor:
    """根据距离场、角度场与几何最小长度判定两点连线的软合法度矩阵。"""
    mean_df = sampled_df.mean(dim=-1)
    df_valid = torch.sigmoid((mean_df - tau_df) * 10.0)

    af_diff = torch.abs(sampled_af - sampled_af.mean(dim=-1, keepdim=True))
    af_diff = torch.minimum(af_diff, 1.0 - af_diff)
    af_valid = torch.sigmoid((tau_ang - af_diff.mean(dim=-1)) * 20.0)

    len_valid = torch.sigmoid((dist - min_length) * 1.0)
    v_matrix = df_valid * af_valid * len_valid

    k = dist.shape[1]
    eye = torch.eye(k, device=dist.device, dtype=torch.bool).unsqueeze(0)
    v_matrix = v_matrix.masked_fill(eye, 0.0)
    return v_matrix


def _coords_to_target_heatmap(points_gt: Any, height: int, width: int, device: torch.device) -> torch.Tensor:
    """将坐标列表/张量转为单通道目标端点热力图 (3x3 邻域软目标)。"""
    target = torch.zeros((1, 1, height, width), device=device)
    if points_gt is None:
        return target
    pts = points_gt if torch.is_tensor(points_gt) else torch.as_tensor(points_gt, device=device)
    if pts.dim() == 3:
        pts = pts[0]
    if pts.numel() == 0:
        return target
    coords = pts.round().long()
    for pt in coords:
        px = int(pt[0].clamp(0, width - 1).item())
        py = int(pt[1].clamp(0, height - 1).item())
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < height and 0 <= nx < width:
                    val = 1.0 if (dy == 0 and dx == 0) else 0.5
                    if val > target[0, 0, ny, nx]:
                        target[0, 0, ny, nx] = val
    return target


def _cell_centers(cells_h: int, cells_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = (torch.arange(cells_h, device=device, dtype=dtype) + 0.5) * 8.0 - 0.5
    cols = (torch.arange(cells_w, device=device, dtype=dtype) + 0.5) * 8.0 - 0.5
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def _apply_homography(homography: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=-1)
    mapped = torch.einsum("bij,bmj->bmi", homography, homo)
    w = mapped[..., 2:3].clamp(min=1e-6)
    return mapped[..., :2] / w


# =============================================================================
# 3. 统一解耦损失类 (SuperPointLoss / SuperPointJointLoss)
# =============================================================================

class SuperPointLoss(nn.Module):
    """SuperPoint 统一多阶段递进损失模块。

    支持阶段式动态权重配置与单张量数学计算。
    """

    def __init__(
        self,
        stage: str = "stage5_joint",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # 几何拓扑与度量超参数
        self.positive_weight = float(kwargs.get("positive_weight", 250.0))
        self.positive_margin = float(kwargs.get("positive_margin", 1.0))
        self.negative_margin = float(kwargs.get("negative_margin", 0.2))
        self.det_pos_weight = float(kwargs.get("det_pos_weight", 50.0))
        self.length_exponent = float(kwargs.get("length_exponent", 1.5))
        self.probe_dist = float(kwargs.get("probe_dist", 4.0))
        self.tau_df = float(kwargs.get("tau_df", 0.55))
        self.tau_ang = float(kwargs.get("tau_ang", 0.10))
        self.min_length = float(kwargs.get("min_length", 8.0))
        self.num_candidates = int(kwargs.get("num_candidates", 16))
        self.num_line_samples = int(kwargs.get("num_line_samples", 10))
        self.nms_radius = int(kwargs.get("nms_radius", 4))
        self.line_neg_margin = float(kwargs.get("line_neg_margin", 0.2))
        self.bg_neg_margin = float(kwargs.get("bg_neg_margin", 0.1))
        self.num_line_points = int(kwargs.get("num_line_points", 12))
        self.num_bg_samples = int(kwargs.get("num_bg_samples", 64))

        # 阶段权重字典
        self.stage = stage
        self.weights: dict[str, float] = {}
        self.set_stage(stage, **kwargs)

    def set_stage(self, stage: str | dict[str, float], **override_weights: Any) -> None:
        """设置当前训练阶段，动态更新各损失项系数。"""
        if isinstance(stage, str):
            self.stage = stage
            base = dict(STAGE_LOSS_WEIGHTS.get(stage, STAGE_LOSS_WEIGHTS["stage5_joint"]))
        else:
            self.stage = "custom"
            base = dict(stage)

        # 允许构造函数与命令行直接覆盖特定权重
        for k, v in override_weights.items():
            if k in base and v is not None:
                base[k] = float(v)
        self.weights = base

    # 快捷属性访问 (兼容现有测试代码)
    @property
    def weight_dist(self) -> float:
        return self.weights.get("weight_dist", 0.0)

    @property
    def weight_ang(self) -> float:
        return self.weights.get("weight_ang", 0.0)

    @property
    def weight_det(self) -> float:
        return self.weights.get("weight_det", 1.0)

    @property
    def weight_orphan(self) -> float:
        return self.weights.get("weight_orphan", 1.0)

    @property
    def weight_len(self) -> float:
        return self.weights.get("weight_len", 0.5)

    @property
    def weight_premature(self) -> float:
        return self.weights.get("weight_premature", 0.5)

    @property
    def weight_intra(self) -> float:
        return self.weights.get("weight_intra", 1.0)

    @property
    def weight_cross(self) -> float:
        return self.weights.get("weight_cross", 1.5)

    @property
    def weight_inter(self) -> float:
        return self.weights.get("weight_inter", 0.5)

    @property
    def weight_bg(self) -> float:
        return self.weights.get("weight_bg", 0.3)

    @property
    def weight_desc(self) -> float:
        return self.weights.get("weight_desc", 1.0)

    # -------------------------------------------------------------------------
    # 各分项损失对外接口 (Direct Method Calls)
    # -------------------------------------------------------------------------

    def detector_loss(self, heatmap: torch.Tensor, points_gt: Any) -> torch.Tensor:
        if heatmap.shape[1] == 65:
            # 65 通道网格交叉熵
            batch, _, hc, wc = heatmap.shape
            labels = torch.full((batch, hc, wc), 64, dtype=torch.long, device=heatmap.device)
            if points_gt is not None:
                pts = points_gt[0] if (torch.is_tensor(points_gt) and points_gt.dim() == 3) else points_gt
                for pt in pts:
                    cx, cy = int(pt[0].item()) // 8, int(pt[1].item()) // 8
                    dx, dy = int(pt[0].item()) % 8, int(pt[1].item()) % 8
                    if 0 <= cx < wc and 0 <= cy < hc:
                        labels[0, cy, cx] = dy * 8 + dx
            weights = torch.ones(65, device=heatmap.device, dtype=heatmap.dtype)
            weights[:-1] = self.det_pos_weight
            return F.cross_entropy(heatmap, labels, weight=weights)

        _, _, h, w = heatmap.shape
        target = _coords_to_target_heatmap(points_gt, h, w, heatmap.device)
        return loss_detector_bce(heatmap, target, self.det_pos_weight)

    def orphan_endpoint_loss(self, scores: torch.Tensor, v_matrix: torch.Tensor) -> torch.Tensor:
        return loss_orphan_endpoint(scores, v_matrix)

    def max_length_line_loss(
        self, scores: torch.Tensor, dist: torch.Tensor, v_matrix: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        return loss_max_length_line(scores, dist, v_matrix, height, width, self.length_exponent)

    def premature_ray_probing_loss(
        self, scores: torch.Tensor, dist: torch.Tensor, diff: torch.Tensor,
        p1: torch.Tensor, p2: torch.Tensor, v_matrix: torch.Tensor, df_map: torch.Tensor
    ) -> torch.Tensor:
        return loss_premature_probing(scores, dist, diff, p1, p2, v_matrix, df_map, self.probe_dist, self.tau_df)

    def descriptor_loss(self, desc0: torch.Tensor, desc1: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
        batch, channels, cells_h, cells_w = desc0.shape
        num_cells = cells_h * cells_w
        flat0 = desc0.permute(0, 2, 3, 1).reshape(batch, num_cells, channels)
        flat1 = desc1.permute(0, 2, 3, 1).reshape(batch, num_cells, channels)
        dot = torch.matmul(flat0, flat1.transpose(1, 2))
        centers = _cell_centers(cells_h, cells_w, desc0.device, desc0.dtype).reshape(1, num_cells, 2).expand(batch, -1, -1)
        warped = _apply_homography(homography, centers)
        dist = torch.cdist(warped, centers)
        pos = (dist <= 8.0).to(dot.dtype)
        hinge_pos = self.positive_weight * pos * F.relu(self.positive_margin - dot)
        hinge_neg = (1.0 - pos) * F.relu(dot - self.negative_margin)
        return (hinge_pos + hinge_neg).mean()

    def extract_candidate_endpoints(self, heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_candidate_endpoints(heatmap, self.num_candidates, self.nms_radius)

    def interpolate_and_sample_lines(
        self, points: torch.Tensor, df_map: torch.Tensor, af_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return interpolate_and_sample_lines(points, df_map, af_map, self.num_line_samples)

    def compute_line_validity_matrix(
        self, sampled_df: torch.Tensor, sampled_af: torch.Tensor, dist: torch.Tensor
    ) -> torch.Tensor:
        return compute_line_validity_matrix(sampled_df, sampled_af, dist, self.tau_df, self.tau_ang, self.min_length)

    # 别名映射
    loss_descriptor = descriptor_loss
    loss_detector = detector_loss
    loss_orphan = orphan_endpoint_loss
    loss_max_length = max_length_line_loss
    loss_premature = premature_ray_probing_loss

    # -------------------------------------------------------------------------
    # 前向计算 (Forward): 根据激活权重计算，不激活项跳过计算省算力
    # -------------------------------------------------------------------------

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
        device = homography.device
        zero = torch.tensor(0.0, device=device)

        out = {
            "loss": zero,
            "l_dist": zero,
            "l_ang": zero,
            "ld": zero,
            "l_det": zero,
            "l_orphan": zero,
            "l_len": zero,
            "l_prem": zero,
            "l_intra": zero,
            "l_cross": zero,
            "l_inter": zero,
            "l_bg": zero,
        }

        # ----------------- 0. 纯直线场找线检测损失 (Line Field Finding: 距离比较与角度极差) -----------------
        w_dist = self.weights.get("weight_dist", 0.0)
        w_ang = self.weights.get("weight_ang", 0.0)

        if (w_dist > 0 or w_ang > 0) and image0 is not None and image0.shape[1] >= 5:
            hm0 = pred0.get("endpoint_mask", pred0.get("heatmap"))
            if hm0 is not None:
                df0, af0 = image0[:, 3:4], image0[:, 4:5]
                sc0, pts0 = self.extract_candidate_endpoints(hm0)
                sdf0, saf0, dist0, diff0, p1_0, p2_0 = self.interpolate_and_sample_lines(pts0, df0, af0)

                k0 = sc0.shape[1]
                pair_scores0 = sc0.unsqueeze(2) * sc0.unsqueeze(1)
                eye0 = torch.eye(k0, device=device, dtype=torch.bool).unsqueeze(0)
                pair_scores0 = pair_scores0.masked_fill(eye0, 0.0)

                l_dist0 = loss_line_distance(sdf0, pair_scores0, self.tau_df) if w_dist > 0 else zero
                l_ang0 = loss_line_angle(saf0, pair_scores0, self.tau_ang) if w_ang > 0 else zero

                if image1 is not None and image1.shape[1] >= 5:
                    hm1 = pred1.get("endpoint_mask", pred1.get("heatmap"))
                    df1, af1 = image1[:, 3:4], image1[:, 4:5]
                    sc1, pts1 = self.extract_candidate_endpoints(hm1)
                    sdf1, saf1, dist1, diff1, p1_1, p2_1 = self.interpolate_and_sample_lines(pts1, df1, af1)
                    k1 = sc1.shape[1]
                    eye1 = torch.eye(k1, device=device, dtype=torch.bool).unsqueeze(0)
                    pair_scores1 = (sc1.unsqueeze(2) * sc1.unsqueeze(1)).masked_fill(eye1, 0.0)
                    l_dist1 = loss_line_distance(sdf1, pair_scores1, self.tau_df) if w_dist > 0 else zero
                    l_ang1 = loss_line_angle(saf1, pair_scores1, self.tau_ang) if w_ang > 0 else zero
                    out["l_dist"] = 0.5 * (l_dist0 + l_dist1)
                    out["l_ang"] = 0.5 * (l_ang0 + l_ang1)
                else:
                    out["l_dist"] = l_dist0
                    out["l_ang"] = l_ang0

                out["loss"] = out["loss"] + w_dist * out["l_dist"] + w_ang * out["l_ang"]

        # ----------------- 1. 检测分支计算 (Detector Losses) -----------------
        w_det = self.weights.get("weight_det", 0.0)
        w_orph = self.weights.get("weight_orphan", 0.0)
        w_len = self.weights.get("weight_len", 0.0)
        w_prem = self.weights.get("weight_premature", 0.0)

        if any(w > 0 for w in (w_det, w_orph, w_len, w_prem)):
            hm0 = pred0.get("endpoint_mask", pred0.get("heatmap"))
            hm1 = pred1.get("endpoint_mask", pred1.get("heatmap"))
            if hm0 is not None:
                _, _, h, w = hm0.shape
                # (1) 端点 BCE
                if w_det > 0:
                    l_det0 = self.detector_loss(pred0.get("semi", hm0), points0)
                    l_det1 = self.detector_loss(pred1.get("semi", hm1), points1) if hm1 is not None else l_det0
                    out["l_det"] = 0.5 * (l_det0 + l_det1)
                    out["loss"] = out["loss"] + w_det * out["l_det"]

                # (2) 几何拓扑场 (若提供了图像的 df 和 af 场)
                if any(w > 0 for w in (w_orph, w_len, w_prem)) and image0 is not None and image0.shape[1] >= 5:
                    df0, af0 = image0[:, 3:4], image0[:, 4:5]
                    sc0, pts0 = self.extract_candidate_endpoints(hm0)
                    sdf0, _, dist0, diff0, p1_0, p2_0 = self.interpolate_and_sample_lines(pts0, df0, af0)
                    v0 = self.compute_line_validity_matrix(sdf0, _, dist0)

                    if w_orph > 0:
                        out["l_orphan"] = self.orphan_endpoint_loss(sc0, v0)
                        out["loss"] = out["loss"] + w_orph * out["l_orphan"]
                    if w_len > 0:
                        out["l_len"] = self.max_length_line_loss(sc0, dist0, v0, h, w)
                        out["loss"] = out["loss"] + w_len * out["l_len"]
                    if w_prem > 0:
                        out["l_prem"] = self.premature_ray_probing_loss(sc0, dist0, diff0, p1_0, p2_0, v0, df0)
                        out["loss"] = out["loss"] + w_prem * out["l_prem"]

        # ----------------- 2. 描述子分支计算 (Descriptor Losses) -----------------
        w_intra = self.weights.get("weight_intra", 0.0)
        w_cross = self.weights.get("weight_cross", 0.0)
        w_inter = self.weights.get("weight_inter", 0.0)
        w_bg = self.weights.get("weight_bg", 0.0)
        w_desc = self.weights.get("weight_desc", 0.0)

        desc0 = pred0.get("descriptor")
        desc1 = pred1.get("descriptor")

        # 方案 A: 线级一致性与对比推离
        if any(w > 0 for w in (w_intra, w_cross, w_inter, w_bg)) and desc0 is not None and desc1 is not None and lines0 is not None and lines1 is not None:
            s0, proto0 = sample_descriptors_along_lines(desc0, lines0, self.num_line_points)
            s1, proto1 = sample_descriptors_along_lines(desc1, lines1, self.num_line_points)

            if s0.shape[1] > 0:
                if w_intra > 0:
                    out["l_intra"] = 0.5 * (loss_intra_line(s0, proto0) + loss_intra_line(s1, proto1))
                    out["loss"] = out["loss"] + w_intra * out["l_intra"]
                if w_cross > 0:
                    out["l_cross"] = loss_cross_line(proto0, proto1, s0, s1)
                    out["loss"] = out["loss"] + w_cross * out["l_cross"]
                if w_inter > 0:
                    out["l_inter"] = loss_inter_line(proto0, proto1, self.line_neg_margin)
                    out["loss"] = out["loss"] + w_inter * out["l_inter"]
                if w_bg > 0:
                    df0 = image0[:, 3:4] if (image0 is not None and image0.shape[1] >= 4) else None
                    bg_desc0 = sample_background_descriptors(desc0, df0, self.num_bg_samples)
                    out["l_bg"] = loss_bg_contrastive(bg_desc0, proto0, self.bg_neg_margin)
                    out["loss"] = out["loss"] + w_bg * out["l_bg"]

        # 经典点级 Hinge Loss (当未提供 lines 或显式指定 weight_desc > 0 时激活)
        if (w_desc > 0 and lines0 is None) or (w_desc > 0 and not any(w > 0 for w in (w_intra, w_cross))):
            if "desc" in pred0 and "desc" in pred1:
                out["ld"] = self.descriptor_loss(pred0["desc"], pred1["desc"], homography)
                out["loss"] = out["loss"] + w_desc * out["ld"]
        elif any(w > 0 for w in (w_intra, w_cross)):
            out["ld"] = out["l_intra"] + out["l_cross"]
        elif w_desc > 0 and "desc" in pred0 and "desc" in pred1:
            out["ld"] = self.descriptor_loss(pred0["desc"], pred1["desc"], homography)
            out["loss"] = out["loss"] + w_desc * out["ld"]

        return out


# =============================================================================
# 4. 独立解耦类与别名导出 (Decoupled Classes & Backward Compatibility)
# =============================================================================

class SuperPointDescriptorLoss(SuperPointLoss):
    """独立描述子损失类。"""
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(stage="stage4_desc_cross", **kwargs)


class SuperPointDetectorLoss(SuperPointLoss):
    """独立端点检测与几何场损失类。"""
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(stage="stage2_det_geom", **kwargs)


SuperPointJointLoss = SuperPointLoss
SuperPointLineLoss = SuperPointLoss


# 兼容函数
def descriptor_hinge_loss(desc0: torch.Tensor, desc1: torch.Tensor, homography: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return SuperPointDescriptorLoss(**kwargs).descriptor_loss(desc0, desc1, homography)


def detector_bce_loss(heatmap: torch.Tensor, points_gt: Any, pos_weight: float = 50.0) -> torch.Tensor:
    return SuperPointDetectorLoss(det_pos_weight=pos_weight).detector_loss(heatmap, points_gt)


def orphan_endpoint_loss(scores: torch.Tensor, v_matrix: torch.Tensor) -> torch.Tensor:
    return loss_orphan_endpoint(scores, v_matrix)


def max_length_line_loss(scores: torch.Tensor, dist: torch.Tensor, v_matrix: torch.Tensor, height: int, width: int, exponent: float = 1.5) -> torch.Tensor:
    return loss_max_length_line(scores, dist, v_matrix, height, width, exponent)


def premature_ray_probing_loss(
    scores: torch.Tensor, dist: torch.Tensor, diff: torch.Tensor,
    p1: torch.Tensor, p2: torch.Tensor, v_matrix: torch.Tensor, df_map: torch.Tensor,
    probe_dist: float = 4.0, tau_df: float = 0.55,
) -> torch.Tensor:
    return loss_premature_probing(scores, dist, diff, p1, p2, v_matrix, df_map, probe_dist, tau_df)


__all__ = [
    "STAGE_LOSS_WEIGHTS",
    "SuperPointLoss",
    "SuperPointDescriptorLoss",
    "SuperPointDetectorLoss",
    "SuperPointJointLoss",
    "SuperPointLineLoss",
    "sample_descriptors_along_lines",
    "sample_background_descriptors",
    "loss_line_distance",
    "loss_line_angle",
    "loss_intra_line",
    "loss_cross_line",
    "loss_inter_line",
    "loss_bg_contrastive",
    "loss_detector_bce",
    "loss_orphan_endpoint",
    "loss_max_length_line",
    "loss_premature_probing",
    "extract_candidate_endpoints",
    "interpolate_and_sample_lines",
    "compute_line_validity_matrix",
    "descriptor_hinge_loss",
    "detector_bce_loss",
    "orphan_endpoint_loss",
    "max_length_line_loss",
    "premature_ray_probing_loss",
]
