"""GlueStick 损失函数、评测指标与官方匹配网络封装 (GlueStick Loss, Metrics & Wrapper).

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
   - 封装官方 GlueStick 与预训练权重 (checkpoint_GlueStick_MD.tar);
   - 前向与论文一致，可选冻结 GNN 并保持其 BatchNorm 使用预训练统计。
"""

# 启用 Python 3.7+ 的延迟类型注解评估特性，支持使用更简洁的类型提示语法 (例如 list[int] | None)
from __future__ import annotations

# 导入跨平台路径处理模块 Path
from pathlib import Path
# 导入系统模块 sys，用于动态修改 Python 模块搜索路径
import sys
# 导入通用类型 Any，用于字典或动态变量的类型注解
from typing import Any

# 导入数值计算库 NumPy
import numpy as np
# 导入 PyTorch 深度学习框架核心库
import torch
# 导入 PyTorch 神经网络模块基类与网络层
import torch.nn as nn
# 导入 PyTorch 函数式接口 (如激活函数、卷积操作等)
import torch.nn.functional as F

# 计算当前项目根目录 (当前文件 models/gluestick_wrapper.py 向上两级为 new-LSD-matching 根目录)
ROOT = Path(__file__).resolve().parent.parent
# 定义候选的 GlueStick 第三方库所在路径列表 (支持在 third_party 下或根目录下查找)
gluestick_cands = [ROOT / "third_party" / "GlueStick", ROOT / "GlueStick"]
# 查找首个实际存在的 GlueStick 目录；若均不存在则默认选用第一个路径
GLUESTICK_DIR = next((p for p in gluestick_cands if p.is_dir()), gluestick_cands[0])
# 若选中的 GlueStick 目录尚未包含在 sys.path 中，则将其插入到首位以便能够顺利 import gluestick
if str(GLUESTICK_DIR) not in sys.path:
    sys.path.insert(0, str(GLUESTICK_DIR))

# GlueStick: 网络模块；normalize_keypoints / log_double_softmax: 编码与匹配头所用的官方函数
from gluestick.models.gluestick import GlueStick, log_double_softmax, normalize_keypoints


def compute_nll_loss(
    log_assignment: torch.Tensor,
    gt_matches0: torch.Tensor,
    nll_balancing: float = 0.5,
    valid_mask0: torch.Tensor | None = None,
    valid_mask1: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """针对最优传输分配矩阵 (M+1, N+1) 计算平衡负对数似然损失 (Balanced NLL Loss)。

    参数:
        log_assignment: (B, N0 + 1, N1 + 1) 由 log_double_softmax 计算的最优传输分配对数矩阵
        gt_matches0: (B, N0) 视图 0 特征点的真值匹配索引 (>=0 表示匹配至视图 1 对应索引，-1 表示无匹配)
        nll_balancing: 匹配点与未匹配点损失的平衡加权系数 (取值范围 0~1，默认 0.5)
        valid_mask0: (B, N0) 视图 0 特征点有效性掩码 (True 为有效特征点，False 为 Padding 点)
        valid_mask1: (B, N1) 视图 1 特征点有效性掩码 (True 为有效特征点，False 为 Padding 点)

    返回:
        avg_loss: 当前批次的平均标量损失张量
        avg_stats: 统计指标字典，包含 {loss_matched, loss_unmatched, num_matched, num_unmatched}
    """
    # 解包获取分配矩阵的维度: 批次大小 b_size, 视角0节点数+1 (n0_plus_1), 视角1节点数+1 (n1_plus_1)
    b_size, n0_plus_1, n1_plus_1 = log_assignment.shape
    # 分别计算两视角实际特征点/线数量 (减去末尾 1 行/列的防尘桶 Dustbin)
    n0, n1 = n0_plus_1 - 1, n1_plus_1 - 1
    # 获取分配矩阵张量所在的计算设备 (CPU 或 CUDA)
    device = log_assignment.device

    # 初始化批次累计总损失为当前设备上的标量 0.0
    total_loss = log_assignment.new_tensor(0.0)
    # 初始化用于存储批次内每个样本统计指标的列表
    stats_list = []

    # 遍历当前批次中的每一个独立样本
    for b in range(b_size):
        # 提取当前样本视角 0 的真值匹配索引向量，形状为 (N0,)
        gt0 = gt_matches0[b]
        # 若传入了 valid_mask0 则采用，否则默认当前所有点均为有效 (形状为 (N0,))
        mask0 = valid_mask0[b] if valid_mask0 is not None else torch.ones(n0, dtype=torch.bool, device=device)
        # 若传入了 valid_mask1 则采用，否则默认当前所有点均为有效 (形状为 (N1,))
        mask1 = valid_mask1[b] if valid_mask1 is not None else torch.ones(n1, dtype=torch.bool, device=device)

        # -------------------------------------------------------------
        # 1. 计算匹配对损失 (Matched Pairs Loss)
        # -------------------------------------------------------------
        # 筛选真值中成功匹配且未越界、并在 mask0 有效区内的点掩码
        matched_mask0 = (gt0 >= 0) & (gt0 < n1) & mask0
        # 获取符合匹配条件的视角 0 特征点行索引
        matched_i = torch.where(matched_mask0)[0]
        # 获取对应的视角 1 特征点列索引 (由真值直接给出)
        matched_j = gt0[matched_mask0]

        # 若当前样本存在匹配成功的点对
        if len(matched_i) > 0:
            # 从分配矩阵中取出对应 (matched_i, matched_j) 位置的对数似然值
            log_p_matched = log_assignment[b, matched_i, matched_j]
            # 对数似然取负并求均值，作为匹配项损失
            loss_matched = -log_p_matched.mean()
        else:
            # 若无匹配点，匹配损失为 0
            loss_matched = log_assignment.new_tensor(0.0)

        # -------------------------------------------------------------
        # 2. 计算视角 0 未匹配点损失 (Unmatched in View 0 -> Dustbin n1)
        # -------------------------------------------------------------
        # 筛选视角 0 中有效但未匹配 (gt0 < 0) 的点掩码
        unmatched_mask0 = (gt0 < 0) & mask0
        # 提取未匹配点的索引
        unmatched_i0 = torch.where(unmatched_mask0)[0]
        # 最优传输中，视角 0 未匹配点对应最后一列防尘桶 (列索引 n1)，提取其对数概率
        log_p_unm0 = log_assignment[b, unmatched_i0, n1] if len(unmatched_i0) > 0 else log_assignment.new_tensor(0.0)

        # -------------------------------------------------------------
        # 3. 计算视角 1 未匹配点损失 (Unmatched in View 1 -> Dustbin n0)
        # -------------------------------------------------------------
        # 将视角 1 中已被匹配的特征点转为 set 集合，以供快速差集检索
        matched_j_set = set(matched_j.tolist())
        # 获取视角 1 中所有有效的特征点索引
        valid_j1 = torch.where(mask1)[0]
        # 筛选有效但未出现在已匹配集合中的索引，即为视角 1 的未匹配点
        unmatched_j1 = torch.tensor([j.item() for j in valid_j1 if j.item() not in matched_j_set], device=device, dtype=torch.long)
        # 视角 1 未匹配点对应最后一行防尘桶 (行索引 n0)，提取其对数概率
        log_p_unm1 = log_assignment[b, n0, unmatched_j1] if len(unmatched_j1) > 0 else log_assignment.new_tensor(0.0)

        # 统计当前样本中匹配对的总数量
        num_matched = len(matched_i)
        # 统计两视角中所有未匹配点的总数量
        num_unmatched = len(unmatched_i0) + len(unmatched_j1)

        # 若存在未匹配点，计算所有未匹配点投影至防尘桶的对数似然平均负值
        if num_unmatched > 0:
            loss_unmatched = -(log_p_unm0.sum() + log_p_unm1.sum()) / num_unmatched
        else:
            # 若无未匹配点，损失为 0
            loss_unmatched = log_assignment.new_tensor(0.0)

        # -------------------------------------------------------------
        # 4. 正负样本平衡融合 (NLL Balancing)
        # -------------------------------------------------------------
        # 若同时存在匹配点与未匹配点，按 nll_balancing 比例进行线性凸组合加权
        if num_matched > 0 and num_unmatched > 0:
            sample_loss = (1.0 - nll_balancing) * loss_matched + nll_balancing * loss_unmatched
        # 若仅有匹配点，样本损失即为匹配损失
        elif num_matched > 0:
            sample_loss = loss_matched
        # 若仅有未匹配点，样本损失即为未匹配损失
        else:
            sample_loss = loss_unmatched

        # 将当前样本的最终损失累加到总损失中
        total_loss = total_loss + sample_loss
        # 记录当前样本的损失数值与数量统计
        stats_list.append({
            "loss_matched": float(loss_matched.item()),
            "loss_unmatched": float(loss_unmatched.item()),
            "num_matched": num_matched,
            "num_unmatched": num_unmatched,
        })

    # 计算批次内所有样本的平均损失
    avg_loss = total_loss / max(b_size, 1)
    # 计算统计字典中各项指标在批次上的平均值
    avg_stats = {
        "loss_matched": float(np.mean([s["loss_matched"] for s in stats_list])),
        "loss_unmatched": float(np.mean([s["loss_unmatched"] for s in stats_list])),
        "num_matched": int(np.mean([s["num_matched"] for s in stats_list])),
        "num_unmatched": int(np.mean([s["num_unmatched"] for s in stats_list])),
    }
    # 返回损失张量与统计字典
    return avg_loss, avg_stats


class GlueStickLoss(nn.Module):
    """GlueStick 联合特征点与几何线匹配损失模块。"""

    def __init__(
        self,
        point_weight: float = 1.0,      # 关键点匹配损失加权权重
        line_weight: float = 1.0,       # 几何线段匹配损失加权权重
        nll_balancing: float = 0.5,     # 匹配点与未匹配点的正负样本平衡因子
        inter_sup_weight: float = 0.5,  # GNN 中间迭代层辅助监督损失权重
    ) -> None:
        super().__init__()
        # 保存传入的超参数配置
        self.point_weight = point_weight
        self.line_weight = line_weight
        self.nll_balancing = nll_balancing
        self.inter_sup_weight = inter_sup_weight

    def forward(self, pred: dict[str, Any], data: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. 计算特征点分配矩阵的 NLL 损失 (基于预测点对数分配矩阵与真实点匹配标签)
        point_loss, pt_stats = compute_nll_loss(
            log_assignment=pred["log_assignment"],
            gt_matches0=data["gt_matches0"],
            nll_balancing=self.nll_balancing,
        )

        # 2. 计算几何线段分配矩阵的 NLL 损失 (基于预测线对数分配矩阵与真实线匹配标签)
        line_loss, ln_stats = compute_nll_loss(
            log_assignment=pred["line_log_assignment"],
            gt_matches0=data["gt_line_matches0"],
            nll_balancing=self.nll_balancing,
        )

        # 3. 计算中间迭代层的监督损失 (若存在中间层分配矩阵)
        inter_loss = point_loss.new_tensor(0.0)
        num_inter = 0
        # 遍历预测字典中的所有键，查找如 line_0_log_assignment 等中间层输出
        for k, v in pred.items():
            if k.startswith("line_") and k.endswith("_log_assignment") and k != "line_log_assignment":
                sub_loss, _ = compute_nll_loss(v, data["gt_line_matches0"], nll_balancing=self.nll_balancing)
                inter_loss = inter_loss + sub_loss
                num_inter += 1

        # 若存在中间层输出，取所有中间层损失的平均值
        if num_inter > 0:
            inter_loss = inter_loss / num_inter

        # 对特征点损失、线段损失以及中间监督损失进行加权求和，得到最终联合损失
        total_loss = self.point_weight * point_loss + self.line_weight * line_loss + self.inter_sup_weight * inter_loss

        # 构建损失与匹配数目的统计字典，方便训练记录与监控
        loss_dict = {
            "total_loss": float(total_loss.item()),
            "point_loss": float(point_loss.item()),
            "line_loss": float(line_loss.item()),
            "inter_loss": float(inter_loss.item()) if num_inter > 0 else 0.0,
            "pt_matched": pt_stats["num_matched"],
            "ln_matched": ln_stats["num_matched"],
        }
        # 返回总损失张量和详细指标字典
        return total_loss, loss_dict


def compute_matching_metrics(pred: dict[str, Any], data: dict[str, Any]) -> dict[str, float]:
    """计算单批次样本的关键点匹配与线段匹配评估指标 (Precision, Recall, F1)。"""
    # 获取批次内样本数
    b_size = len(pred["matches0"])

    # 初始化分别存储点、线精确率与召回率的列表
    p_prec_list, p_rec_list = [], []
    l_prec_list, l_rec_list = [], []

    # 逐样本遍历计算匹配精度
    for b in range(b_size):
        # -------------------------------------------------------------
        # 1. 关键点指标统计 (Point Metrics)
        # -------------------------------------------------------------
        m0 = pred["matches0"][b]       # 预测的视角 0 特征点匹配结果 (形状: (N0,), >=0 表示匹配目标序号，-1 表示未匹配)
        gt0 = data["gt_matches0"][b]   # 真实的视角 0 特征点匹配真值 (形状: (N0,))

        pred_mask = m0 >= 0            # 模型预测为有效匹配的布尔掩码
        gt_mask = gt0 >= 0             # 真值中存在真实匹配的布尔掩码
        correct_mask = pred_mask & (m0 == gt0)  # 预测匹配且目标索引与真值完全一致的正确匹配掩码

        num_pred = int(pred_mask.sum().item())       # 预测匹配总对数
        num_gt = int(gt_mask.sum().item())           # 真实匹配总对数
        num_correct = int(correct_mask.sum().item()) # 正确匹配对数

        # 计算特征点精确率: 预测正确的比例
        prec = num_correct / max(num_pred, 1)
        # 计算特征点召回率: 真实匹配中被正确预测的比例
        rec = num_correct / max(num_gt, 1)
        p_prec_list.append(prec)
        p_rec_list.append(rec)

        # -------------------------------------------------------------
        # 2. 几何线段指标统计 (Line Metrics)
        # -------------------------------------------------------------
        lm0 = pred["line_matches0"][b]         # 预测的视角 0 线段匹配结果 (形状: (M0,), >=0 为匹配线索引)
        gt_lm0 = data["gt_line_matches0"][b]   # 真实的视角 0 线段匹配真值 (形状: (M0,))

        pred_lmask = lm0 >= 0                  # 预测线匹配有效的掩码
        gt_lmask = gt_lm0 >= 0                 # 真实线匹配有效的掩码
        correct_lmask = pred_lmask & (lm0 == gt_lm0)  # 预测线匹配且目标与真值一致的正确掩码

        num_lpred = int(pred_lmask.sum().item())        # 预测线匹配对数
        num_lgt = int(gt_lmask.sum().item())            # 真实线匹配对数
        num_lcorrect = int(correct_lmask.sum().item())  # 正确线匹配对数

        # 计算线段精确率
        lprec = num_lcorrect / max(num_lpred, 1)
        # 计算线段召回率
        lrec = num_lcorrect / max(num_lgt, 1)
        l_prec_list.append(lprec)
        l_rec_list.append(lrec)

    # 计算批次样本的平均特征点精确率和召回率
    mean_pprec = float(np.mean(p_prec_list))
    mean_prec = float(np.mean(p_rec_list))
    # 计算批次样本的平均线段精确率和召回率
    mean_lprec = float(np.mean(l_prec_list))
    mean_lrec = float(np.mean(l_rec_list))

    # 计算特征点 F1-Score 调和平均数: 2 * P * R / (P + R)
    pf1 = 2 * mean_pprec * mean_prec / max(mean_pprec + mean_prec, 1e-6)
    # 计算线段 F1-Score 调和平均数: 2 * P * R / (P + R)
    lf1 = 2 * mean_lprec * mean_lrec / max(mean_lprec + mean_lrec, 1e-6)

    # 返回指标汇总字典
    return {
        "point_precision": mean_pprec,
        "point_recall": mean_prec,
        "point_f1": pf1,
        "line_precision": mean_lprec,
        "line_recall": mean_lrec,
        "line_f1": lf1,
    }


class SemanticGlueStick(nn.Module):
    """官方 GlueStick 匹配模型封装。类名保留，便于现有训练与评测脚本继续导入。"""

    def __init__(
        self,
        weights: str | Path | None = None,  # 预训练权重文件路径 (若为空则自动查找默认目录)
        gnn_layers: int = 9,                 # GNN 图注意力网络层数 (默认 9 层交替注意力)
        checkpointed: bool = False,          # 是否启用激活重计算 (Gradient Checkpointing) 节省显存
        filter_threshold: float = 0.2,       # 匹配置信度阈值过滤 (大于该值的进入最终匹配)
        freeze_gnn: bool = False,            # 冻结 GNN 时保持其 BatchNorm 使用预训练统计
    ) -> None:
        super().__init__()
        self.freeze_gnn = freeze_gnn

        # 构造官方 GlueStick 初始化所需的配置字典
        conf = {
            "weights": weights,                                 # 预训练权重路径 (None 时随机初始化)
            "GNN_layers": ["self", "cross"] * gnn_layers,               # GNN 层设计: 自注意力和交叉注意力交替堆叠
            "checkpointed": checkpointed,                                # 显存节省开关
            "num_line_iterations": 1,                                    # 线匹配内部迭代求解轮次
            "line_attention": False,                                     # 是否使用显式线注意力
            "filter_threshold": filter_threshold,                       # 匹配对置信度阈值过滤 (大于该值的进入最终匹配)
        }
        # 实例化官方底座 GlueStick 模型
        self.gluestick = GlueStick(conf)
        if self.freeze_gnn:
            for param in self.gluestick.gnn.parameters():
                param.requires_grad = False
            self.gluestick.gnn.eval()

    def train(self, mode: bool = True):
        """进入训练模式；冻结 GNN 时把它留在 eval，避免更新预训练 BatchNorm 统计。"""
        super().train(mode)
        if self.freeze_gnn:
            self.gluestick.gnn.eval()
        return self

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """按论文顺序展开：输入特征、位置编码器、GNN、匹配头。"""
        net = self.gluestick
        device = data["keypoints0"].device
        b_size = len(data["keypoints0"])
        image_size0 = data["image_size0"] if "image_size0" in data else data["image0"].shape
        image_size1 = data["image_size1"] if "image_size1" in data else data["image1"].shape

        # 1. 特征。描述子、关键点和线段由 SuperPoint / DeepLSD 在网络外提取，这里只取出。
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        n_kpts0, n_kpts1 = kpts0.shape[1], kpts1.shape[1]
        n_lines0, n_lines1 = data["lines0"].shape[1], data["lines1"].shape[1]

        pred: dict[str, Any] = {}
        if n_kpts0 == 0 or n_kpts1 == 0:
            pred["log_assignment"] = torch.zeros(b_size, n_kpts0, n_kpts1, dtype=torch.float, device=device)
            pred["matches0"] = torch.full((b_size, n_kpts0), -1, device=device, dtype=torch.int64)
            pred["matches1"] = torch.full((b_size, n_kpts1), -1, device=device, dtype=torch.int64)
            pred["match_scores0"] = torch.zeros((b_size, n_kpts0), device=device, dtype=torch.float32)
            pred["match_scores1"] = torch.zeros((b_size, n_kpts1), device=device, dtype=torch.float32)
            pred["line_log_assignment"] = torch.zeros(b_size, n_lines0, n_lines1, dtype=torch.float, device=device)
            pred["line_matches0"] = torch.full((b_size, n_lines0), -1, device=device, dtype=torch.int64)
            pred["line_matches1"] = torch.full((b_size, n_lines1), -1, device=device, dtype=torch.int64)
            pred["line_match_scores0"] = torch.zeros((b_size, n_lines0), device=device, dtype=torch.float32)
            pred["line_match_scores1"] = torch.zeros((b_size, n_kpts1), device=device, dtype=torch.float32)
            return pred

        # 2. 编码器。坐标归一化后，kenc / lenc 把位置和分数编进描述子。
        if net.conf.bottleneck_dim is not None:
            desc0 = net.bottleneck_up(net.bottleneck_down(desc0))
            desc1 = net.bottleneck_up(net.bottleneck_down(desc1))
            desc0 = F.normalize(desc0, p=2, dim=1)
            desc1 = F.normalize(desc1, p=2, dim=1)
        if net.conf.input_dim != net.conf.descriptor_dim:
            desc0 = net.input_proj(desc0)
            desc1 = net.input_proj(desc1)

        kpts0 = normalize_keypoints(kpts0, image_size0)
        kpts1 = normalize_keypoints(kpts1, image_size1)
        desc0 = desc0 + net.kenc(kpts0, data["keypoint_scores0"])
        desc1 = desc1 + net.kenc(kpts1, data["keypoint_scores1"])

        lines_junc_idx0 = data["lines_junc_idx0"].flatten(1, 2)
        lines_junc_idx1 = data["lines_junc_idx1"].flatten(1, 2)
        if n_lines0 != 0 and n_lines1 != 0:
            lines0 = normalize_keypoints(data["lines0"].flatten(1, 2), image_size0).reshape(b_size, n_lines0, 2, 2)
            lines1 = normalize_keypoints(data["lines1"].flatten(1, 2), image_size1).reshape(b_size, n_lines1, 2, 2)
            line_enc0 = net.lenc(lines0, data["line_scores0"])
            line_enc1 = net.lenc(lines1, data["line_scores1"])
        else:
            line_enc0 = torch.zeros(b_size, net.conf.descriptor_dim, n_lines0 * 2, dtype=torch.float, device=device)
            line_enc1 = torch.zeros(b_size, net.conf.descriptor_dim, n_lines1 * 2, dtype=torch.float, device=device)

        # 3. GNN。权重冻结时仍做前向，梯度可以穿过它回到编码器；BatchNorm 保持 eval。
        if self.freeze_gnn:
            net.gnn.eval()
        desc0, desc1 = net.gnn(desc0, desc1, line_enc0, line_enc1, lines_junc_idx0, lines_junc_idx1)

        # 4. 匹配头。投影、双重 softmax，再做互最近邻筛选。
        mdesc0, mdesc1 = net.final_proj(desc0), net.final_proj(desc1)
        kp_scores = torch.einsum("bdn,bdm->bnm", mdesc0, mdesc1)
        kp_scores = kp_scores / net.conf.descriptor_dim ** 0.5
        kp_scores = log_double_softmax(kp_scores, net.bin_score)
        m0, m1, mscores0, mscores1 = net._get_matches(kp_scores)
        pred["log_assignment"] = kp_scores
        pred["matches0"] = m0
        pred["matches1"] = m1
        pred["match_scores0"] = mscores0
        pred["match_scores1"] = mscores1

        if n_lines0 > 0 and n_lines1 > 0:
            line_scores, m0_lines, m1_lines, mscores0_lines, mscores1_lines, raw_line_scores = net._get_line_matches(
                desc0[:, :, :2 * n_lines0],
                desc1[:, :, :2 * n_lines1],
                lines_junc_idx0,
                lines_junc_idx1,
                net.final_line_proj,
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
