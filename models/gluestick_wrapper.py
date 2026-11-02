"""GlueStick 损失函数、评测指标与语义增强网络封装 (GlueStick Loss, Metrics & Semantic Module).

核心设计:
1. GlueStickLoss:
   - 严格遵循 SuperGlue / GlueStick 最优传输 (Optimal Transport) 负对数似然 (NLL) 损失;
   - 考虑关键点匹配对、真值防尘桶 (Dustbin) 以及无匹配惩罚;
   - 线段匹配 NLL 损失与多层中间监督损失 (inter-supervision);
   - 支持 nll_balancing 正负样本平衡因子。
2. GlueStickMetrics:
   - 计算跨视角特征点匹配的 Precision (精确率)、Recall (召回率)、F1-Score;
   - 计算几何线段匹配的 Line Precision、Line Recall、Line F1-Score。
3. SemanticGlueStick:
   - 原生兼容官方 GlueStick 模型与预训练权重 (checkpoint_GlueStick_MD.tar);
   - 提供语义增强开关 (use_semantics=True):
     利用 SAM ViT-H 提取的 256 维语义特征与实例 Mask 轮廓属性，通过轻量级门控适配器 (Semantic Adapter)
     将语义先验残差融合至线段几何编码层，在网络推理与图注意力中注入深层语义上下文。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
gluestick_cands = [ROOT / "third_party" / "GlueStick", ROOT / "GlueStick"]
GLUESTICK_DIR = next((p for p in gluestick_cands if p.is_dir()), gluestick_cands[0])
if str(GLUESTICK_DIR) not in sys.path:
    sys.path.insert(0, str(GLUESTICK_DIR))

from gluestick.models.gluestick import GlueStick, normalize_keypoints, log_double_softmax


def compute_nll_loss(
    log_assignment: torch.Tensor,
    gt_matches0: torch.Tensor,
    nll_balancing: float = 0.5,
    valid_mask0: torch.Tensor | None = None,
    valid_mask1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """针对最优传输分配矩阵 (M+1, N+1) 计算平衡负对数似然损失。

    参数:
        log_assignment: (B, N0 + 1, N1 + 1) log_double_softmax 分配矩阵
        gt_matches0: (B, N0) 视图 0 的真值匹配索引 (>=0 表示匹配至视图 1 对应点，-1 表示未匹配)
        nll_balancing: 匹配点与未匹配点损失平衡系数 (默认 0.5)
        valid_mask0: (B, N0) 视图 0 特征点有效性掩码 (排除 Padding)
        valid_mask1: (B, N1) 视图 1 特征点有效性掩码 (排除 Padding)

    返回:
        loss: 标量损失张量
        stats: 统计字典 {loss_matched, loss_unmatched, num_matched, num_unmatched}
    """
    b_size, n0_plus_1, n1_plus_1 = log_assignment.shape
    n0, n1 = n0_plus_1 - 1, n1_plus_1 - 1
    device = log_assignment.device

    total_loss = log_assignment.new_tensor(0.0)
    stats_list = []

    for b in range(b_size):
        gt0 = gt_matches0[b]  # (N0,)
        mask0 = valid_mask0[b] if valid_mask0 is not None else torch.ones(n0, dtype=torch.bool, device=device)
        mask1 = valid_mask1[b] if valid_mask1 is not None else torch.ones(n1, dtype=torch.bool, device=device)

        # 1. 匹配对损失
        matched_mask0 = (gt0 >= 0) & (gt0 < n1) & mask0
        matched_i = torch.where(matched_mask0)[0]
        matched_j = gt0[matched_mask0]

        if len(matched_i) > 0:
            log_p_matched = log_assignment[b, matched_i, matched_j]
            loss_matched = -log_p_matched.mean()
        else:
            loss_matched = log_assignment.new_tensor(0.0)

        # 2. 视角 0 未匹配点 -> 投影至视角 0 防尘桶 (列 n1)
        unmatched_mask0 = (gt0 < 0) & mask0
        unmatched_i0 = torch.where(unmatched_mask0)[0]
        log_p_unm0 = log_assignment[b, unmatched_i0, n1] if len(unmatched_i0) > 0 else log_assignment.new_tensor(0.0)

        # 3. 视角 1 未匹配点 -> 投影至视角 1 防尘桶 (行 n0)
        matched_j_set = set(matched_j.tolist())
        valid_j1 = torch.where(mask1)[0]
        unmatched_j1 = torch.tensor([j.item() for j in valid_j1 if j.item() not in matched_j_set], device=device, dtype=torch.long)
        log_p_unm1 = log_assignment[b, n0, unmatched_j1] if len(unmatched_j1) > 0 else log_assignment.new_tensor(0.0)

        num_matched = len(matched_i)
        num_unmatched = len(unmatched_i0) + len(unmatched_j1)

        if num_unmatched > 0:
            loss_unmatched = -(log_p_unm0.sum() + log_p_unm1.sum()) / num_unmatched
        else:
            loss_unmatched = log_assignment.new_tensor(0.0)

        if num_matched > 0 and num_unmatched > 0:
            sample_loss = (1.0 - nll_balancing) * loss_matched + nll_balancing * loss_unmatched
        elif num_matched > 0:
            sample_loss = loss_matched
        else:
            sample_loss = loss_unmatched

        total_loss = total_loss + sample_loss
        stats_list.append({
            "loss_matched": float(loss_matched.item()),
            "loss_unmatched": float(loss_unmatched.item()),
            "num_matched": num_matched,
            "num_unmatched": num_unmatched,
        })

    avg_loss = total_loss / max(b_size, 1)
    avg_stats = {
        "loss_matched": float(np.mean([s["loss_matched"] for s in stats_list])),
        "loss_unmatched": float(np.mean([s["loss_unmatched"] for s in stats_list])),
        "num_matched": int(np.mean([s["num_matched"] for s in stats_list])),
        "num_unmatched": int(np.mean([s["num_unmatched"] for s in stats_list])),
    }
    return avg_loss, avg_stats


class GlueStickLoss(nn.Module):
    """GlueStick 联合特征点与几何线匹配损失模块。"""

    def __init__(
        self,
        point_weight: float = 1.0,
        line_weight: float = 1.0,
        nll_balancing: float = 0.5,
        inter_sup_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.point_weight = point_weight
        self.line_weight = line_weight
        self.nll_balancing = nll_balancing
        self.inter_sup_weight = inter_sup_weight

    def forward(self, pred: dict[str, Any], data: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. 特征点分配 NLL 损失
        point_loss, pt_stats = compute_nll_loss(
            log_assignment=pred["log_assignment"],
            gt_matches0=data["gt_matches0"],
            nll_balancing=self.nll_balancing,
        )

        # 2. 线段分配 NLL 损失
        line_loss, ln_stats = compute_nll_loss(
            log_assignment=pred["line_log_assignment"],
            gt_matches0=data["gt_line_matches0"],
            nll_balancing=self.nll_balancing,
        )

        # 3. 中间层监督 (若启用)
        inter_loss = point_loss.new_tensor(0.0)
        num_inter = 0
        for k, v in pred.items():
            if k.startswith("line_") and k.endswith("_log_assignment") and k != "line_log_assignment":
                sub_loss, _ = compute_nll_loss(v, data["gt_line_matches0"], nll_balancing=self.nll_balancing)
                inter_loss = inter_loss + sub_loss
                num_inter += 1

        if num_inter > 0:
            inter_loss = inter_loss / num_inter

        total_loss = self.point_weight * point_loss + self.line_weight * line_loss + self.inter_sup_weight * inter_loss

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "point_loss": float(point_loss.item()),
            "line_loss": float(line_loss.item()),
            "inter_loss": float(inter_loss.item()) if num_inter > 0 else 0.0,
            "pt_matched": pt_stats["num_matched"],
            "ln_matched": ln_stats["num_matched"],
        }
        return total_loss, loss_dict


def compute_matching_metrics(pred: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
    """计算单批次样本的关键点匹配与线段匹配评估指标。"""
    b_size = len(pred["matches0"])

    p_prec_list, p_rec_list = [], []
    l_prec_list, l_rec_list = [], []

    for b in range(b_size):
        # 1. 关键点指标
        m0 = pred["matches0"][b]  # (N0,)
        gt0 = data["gt_matches0"][b]  # (N0,)

        pred_mask = m0 >= 0
        gt_mask = gt0 >= 0
        correct_mask = pred_mask & (m0 == gt0)

        num_pred = int(pred_mask.sum().item())
        num_gt = int(gt_mask.sum().item())
        num_correct = int(correct_mask.sum().item())

        prec = num_correct / max(num_pred, 1)
        rec = num_correct / max(num_gt, 1)
        p_prec_list.append(prec)
        p_rec_list.append(rec)

        # 2. 线段指标
        lm0 = pred["line_matches0"][b]  # (M0,)
        gt_lm0 = data["gt_line_matches0"][b]  # (M0,)

        pred_lmask = lm0 >= 0
        gt_lmask = gt_lm0 >= 0
        correct_lmask = pred_lmask & (lm0 == gt_lm0)

        num_lpred = int(pred_lmask.sum().item())
        num_lgt = int(gt_lmask.sum().item())
        num_lcorrect = int(correct_lmask.sum().item())

        lprec = num_lcorrect / max(num_lpred, 1)
        lrec = num_lcorrect / max(num_lgt, 1)
        l_prec_list.append(lprec)
        l_rec_list.append(lrec)

    mean_pprec = float(np.mean(p_prec_list))
    mean_prec = float(np.mean(p_rec_list))
    mean_lprec = float(np.mean(l_prec_list))
    mean_lrec = float(np.mean(l_rec_list))

    pf1 = 2 * mean_pprec * mean_prec / max(mean_pprec + mean_prec, 1e-6)
    lf1 = 2 * mean_lprec * mean_lrec / max(mean_lprec + mean_lrec, 1e-6)

    return {
        "point_precision": mean_pprec,
        "point_recall": mean_prec,
        "point_f1": pf1,
        "line_precision": mean_lprec,
        "line_recall": mean_lrec,
        "line_f1": lf1,
    }


class SemanticGlueStick(nn.Module):
    """语义融合版 GlueStick 匹配模型。

    既可原生运行官方 GlueStick，亦可启用 SAM ViT-H 语义适配器进行跨模态特征融合。
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        use_semantics: bool = False,
        semantic_dim: int = 256,
        attr_dim: int = 3,
        gnn_layers: int = 9,
        checkpointed: bool = False,
    ) -> None:
        super().__init__()
        self.use_semantics = use_semantics

        resolved_weights = None
        if weights and Path(weights).is_file():
            resolved_weights = str(Path(weights).resolve())
        else:
            cand_weights = [
                ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar",
                ROOT / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar",
                ROOT / "weights" / "gluestick" / "checkpoint_GlueStick_MD.tar",
            ]
            for p in cand_weights:
                if p.is_file():
                    resolved_weights = str(p.resolve())
                    break

        conf = {
            "weights": resolved_weights,
            "GNN_layers": ["self", "cross"] * gnn_layers,
            "checkpointed": checkpointed,
            "num_line_iterations": 1,
            "line_attention": False,
            "filter_threshold": 0.2,
        }
        self.gluestick = GlueStick(conf)

        if self.use_semantics:
            # 语义适配器: 将 SAM ViT-H 256 维线语义描述子 + 3 维 Mask 属性映射至 256 维空间
            in_dim = semantic_dim + attr_dim
            self.semantic_adapter = nn.Sequential(
                nn.Conv1d(in_dim, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=1),
            )
            # 可学习残差门控标量，初始设为 0.1 保证稳定微调
            self.semantic_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.use_semantics:
            return self.gluestick(data)

        # 启用语义分支时，注入 SAM 语义残差
        data_mod = dict(data)
        device = data["keypoints0"].device
        b_size = len(data["keypoints0"])
        image_size0 = data.get("image_size0", data["image0"].shape)
        image_size1 = data.get("image_size1", data["image1"].shape)

        pred = {}
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        n_lines0, n_lines1 = data["lines0"].shape[1], data["lines1"].shape[1]

        kpts0_norm = normalize_keypoints(kpts0, image_size0)
        kpts1_norm = normalize_keypoints(kpts1, image_size1)
        desc0 = desc0 + self.gluestick.kenc(kpts0_norm, data["keypoint_scores0"])
        desc1 = desc1 + self.gluestick.kenc(kpts1_norm, data["keypoint_scores1"])

        lines0_flat = data["lines0"].flatten(1, 2)
        lines1_flat = data["lines1"].flatten(1, 2)
        lines_junc_idx0 = data["lines_junc_idx0"].flatten(1, 2)
        lines_junc_idx1 = data["lines_junc_idx1"].flatten(1, 2)

        if n_lines0 > 0 and n_lines1 > 0:
            lines0_norm = normalize_keypoints(lines0_flat, image_size0).reshape(b_size, n_lines0, 2, 2)
            lines1_norm = normalize_keypoints(lines1_flat, image_size1).reshape(b_size, n_lines1, 2, 2)
            line_enc0 = self.gluestick.lenc(lines0_norm, data["line_scores0"])
            line_enc1 = self.gluestick.lenc(lines1_norm, data["line_scores1"])

            # 融合 SAM 语义线特征 (B, 256, M) 与 Mask 属性 (B, M, 3) -> (B, 259, M)
            sem0 = data["line_semantic_descs0"]
            attrs0 = data["line_mask_attrs0"].transpose(1, 2)
            sem_input0 = torch.cat([sem0, attrs0], dim=1)
            sem_feat0 = self.semantic_adapter(sem_input0)  # (B, 256, M0)
            # 每个线段包含 2 个端点，沿最后一个维度扩展 2 倍对齐 (B, 256, 2 * M0)
            sem_feat0_rep = sem_feat0.repeat_interleave(2, dim=-1)
            line_enc0 = line_enc0 + self.semantic_gate * sem_feat0_rep

            sem1 = data["line_semantic_descs1"]
            attrs1 = data["line_mask_attrs1"].transpose(1, 2)
            sem_input1 = torch.cat([sem1, attrs1], dim=1)
            sem_feat1 = self.semantic_adapter(sem_input1)  # (B, 256, M1)
            sem_feat1_rep = sem_feat1.repeat_interleave(2, dim=-1)
            line_enc1 = line_enc1 + self.semantic_gate * sem_feat1_rep
        else:
            line_enc0 = torch.zeros(b_size, self.gluestick.conf.descriptor_dim, n_lines0 * 2, dtype=torch.float, device=device)
            line_enc1 = torch.zeros(b_size, self.gluestick.conf.descriptor_dim, n_lines1 * 2, dtype=torch.float, device=device)

        # 通过原版 GNN 与双视角注意力传播
        desc0, desc1 = self.gluestick.gnn(desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)

        # 匹配点
        mdesc0, mdesc1 = self.gluestick.final_proj(desc0), self.gluestick.final_proj(desc1)
        raw_kp_scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1) / (self.gluestick.conf.descriptor_dim ** 0.5)
        kp_scores = log_double_softmax(raw_kp_scores, self.gluestick.bin_score)

        m0, m1, mscores0, mscores1 = self.gluestick._get_matches(kp_scores)
        pred["log_assignment"] = kp_scores
        pred["matches0"] = m0
        pred["matches1"] = m1
        pred["match_scores0"] = mscores0
        pred["match_scores1"] = mscores1

        # 匹配线段
        if n_lines0 > 0 and n_lines1 > 0:
            line_scores, m0_lines, m1_lines, mscores0_lines, mscores1_lines, raw_line_scores = self.gluestick._get_line_matches(
                desc0[:, :, :2 * n_lines0], desc1[:, :, :2 * n_lines1], lines_junc_idx0, lines_junc_idx1, self.gluestick.final_line_proj
            )
        else:
            line_scores = torch.zeros(b_size, n_lines0, n_lines1, dtype=torch.float, device=device)
            m0_lines = torch.full((b_size, n_lines0), -1, device=device, dtype=torch.int64)
            m1_lines = torch.full((b_size, n_lines1), -1, device=device, dtype=torch.int64)
            mscores0_lines = torch.zeros((b_size, n_lines0), device=device, dtype=torch.float32)
            mscores1_lines = torch.zeros((b_size, n_lines1), device=device, dtype=torch.float32)
            raw_line_scores = torch.zeros(b_size, n_lines0, n_lines1, dtype=torch.float, device=device)

        pred["line_log_assignment"] = line_scores
        pred["line_matches0"] = m0_lines
        pred["line_matches1"] = m1_lines
        pred["line_match_scores0"] = mscores0_lines
        pred["line_match_scores1"] = mscores1_lines
        pred["raw_line_scores"] = raw_line_scores

        return pred
