# XFeatObjectMatcher：物体级/语义区域双向图匹配网络 ("连连看")

本目录包含 **`XFeatObjectMatcher`** 模型的完整实现。该网络用于跨视角室内场景中的宏观实体区域（家具、墙面、门窗等）进行一对一互斥连连看配对与视野外出画目标剔除。

---

## 目录
- [一、任务定义与数据输入机制](#一任务定义与数据输入机制)
- [二、网络分层架构](#二网络分层架构)
- [三、损失函数详解 (Negative Log-Likelihood on Sinkhorn)](#三损失函数详解-negative-log-likelihood-on-sinkhorn)
  - [1. 为什么用 NLL 损失函数？](#1-为什么用-nll-损失函数)
  - [2. 损失函数的数学公式](#2-损失函数的数学公式)
  - [3. 损失函数的代码逐行解析](#3-损失函数的代码逐行解析)
  - [4. 为什么该损失函数不需要负采样？](#4-为什么该损失函数不需要负采样)
- [四、使用方法与训练命令](#四使用方法与训练命令)

---

## 一、任务定义与数据输入机制

### 1. 任务目标
给定来自两个视角（View 0 和 View 1）拍摄的同一房间场景：
- View 0 包含 $M_0$ 个语义区域；
- View 1 包含 $M_1$ 个语义区域；
- 网络需要输出严格的 **一对一双向互斥配对** $(i, j)$，并将由于视角旋转、遮挡而只在单侧可见的出画目标精确分配至**“垃圾桶”（Dustbin）**。

```mermaid
flowchart LR
    subgraph View0["View 0 (M0 块小图)"]
        A0["沙发 #0"]
        B0["茶几 #1"]
        C0["门 #2 (出画)"]
    end
    subgraph View1["View 1 (M1 块小图)"]
        A1["沙发 #0"]
        B1["茶几 #1"]
        D1["窗 #2 (出画)"]
    end
    subgraph Dustbin["Dustbin (垃圾桶沉底区)"]
        Trash["未匹配目标分类区"]
    end

    A0 <==>|1-to-1 匹配| A1
    B0 <==>|1-to-1 匹配| B1
    C0 -->|自动归入| Trash
    D1 -->|自动归入| Trash
```

### 2. 纯小区域图卡片输入（掩膜不入网络）
- **预处理阶段（在网络外完成）**：原图与语义掩膜直接相乘（背景置黑），沿外接矩形紧密裁剪，并按原始宽高比通过 Letterbox 填充缩放到固定 **$112 \times 112$** 统一正方形；
- **网络输入**：
  - `patches0`: `(B, M0, 3, 112, 112)`
  - `patches1`: `(B, M1, 3, 112, 112)`
- **掩膜彻底剥离**：网络不再接收任何 $(B, M, H, W)$ 的巨大掩膜张量，大幅缩减显存开销，并消除物体边缘的背景卷积污染。

---

## 二、网络分层架构

1. **XFeat 轻量卷积骨干 (`XFeatBackbone`)**：
   - 包含 4 个 Stage，步长累计下采样 8 倍；
   - 对 $B \times M$ 张小图进行全并行批处理卷积，输出特征图 $(B \times M, 64, 14, 14)$；
2. **全局自适应池化与投影 (`AdaptiveAvgPool2d + Linear`)**：
   - 将每张小区域图池化为 64 维全局视觉表示向量，随后投影至 $d_{\text{model}} = 128$ 维物体 Token；
3. **物体级双向图注意力网络 (`ObjectGraphTransformer`)**：
   - **自注意力（Self-Attention）**：学习同一视角内不同实体间的上下文拓扑关系；
   - **跨视角注意力（Cross-Attention）**：在两个视角实体间建立交互；
4. **带垃圾桶的 Log-Sinkhorn 最优传输 (`ObjectSinkhornMatcher`)**：
   - 计算两组物体 Token 的相似度矩阵 $S \in \mathbb{R}^{M_0 \times M_1}$；
   - 扩充一行一列可学习的垃圾桶分数 $\alpha$：
     $$Z = \begin{bmatrix} S_{M_0 \times M_1} & \alpha_{M_0 \times 1} \\ \alpha_{1 \times M_1} & \alpha_{1 \times 1} \end{bmatrix} \in \mathbb{R}^{(M_0+1) \times (M_1+1)}$$
   - 经过 20 次 Log-Sinkhorn 迭代，收敛为严格的双随机双向互斥概率矩阵 $P$。

---

## 三、损失函数详解 (Negative Log-Likelihood on Sinkhorn)

模型使用的损失函数为 [`ObjectMatchingLoss`](file:///home/lbw/Project/semantic-LSD-matching/models/object_matcher/loss.py)。

### 1. 为什么用 NLL 损失函数？
在经过 Sinkhorn 算法处理后，网络输出的分配矩阵 $P$ 具有极好的数学性质：
- 对于 View 0 中的任意物体 $i$：
  $$\sum_{j=0}^{M_1 - 1} P_{i, j} + P_{i, M_1} = 1.0$$
  即物体 $i$ 匹配到 View 1 某个物体 $j$ 的概率，加上它落入垃圾桶（未匹配）的概率 $P_{i, M_1}$，**总和严格为 1**。
- 这本质上构成了一个离散多项式概率分布。因此，最优的监督方法就是分类任务中最经典的**负对数似然（Negative Log-Likelihood, NLL）**。

### 2. 损失函数的数学公式 (平衡正负样本 + 变换一致性位移场共识损失)
设样本的真实匹配对集合为 $\mathcal{M} = \{(i_1, j_1), (i_2, j_2), \dots\}$，View 0 中真实未匹配的落单集合为 $\mathcal{U}_0$，View 1 中真实未匹配的落单集合为 $\mathcal{U}_1$。

总损失由两大部分构成：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{nll}} + \lambda_{\text{motion}} \cdot \mathcal{L}_{\text{motion}}$$

#### (1) 匹配与出画负对数似然损失 $\mathcal{L}_{\text{nll}}$
$$\mathcal{L}_{\text{nll}} = \frac{\mathcal{L}_{\text{pos}}}{|\mathcal{M}|} + \lambda_{\text{dustbin}} \cdot \frac{\mathcal{L}_{\text{neg}}}{|\mathcal{U}_0| + |\mathcal{U}_1|}$$
其中：
$$\mathcal{L}_{\text{pos}} = -\sum_{(i, j) \in \mathcal{M}} \log P_{i, j}, \quad \mathcal{L}_{\text{neg}} = -\sum_{i \in \mathcal{U}_0} \log P_{i, M_1} - \sum_{j \in \mathcal{U}_1} \log P_{M_0, j}$$

#### (2) 变换一致性位移场共识损失 $\mathcal{L}_{\text{motion}}$ (方案一)
刚体室内场景中，相机位姿变换会诱发整体一致的 2D 视差位移场。对于 View 0 物体中心 $\mathbf{c}_i^{(0)}$ 与 View 1 物体中心 $\mathbf{c}_j^{(1)}$，其位移向量为 $\mathbf{u}_{ij} = \mathbf{c}_j^{(1)} - \mathbf{c}_i^{(0)}$。
真实匹配对的全局共识运动趋势为：
$$\bar{\mathbf{u}} = \frac{1}{|\mathcal{M}|} \sum_{(i, j) \in \mathcal{M}} (\mathbf{c}_j^{(1)} - \mathbf{c}_i^{(0)})$$
若网络预测出的候选配对 $(i, j)$ 的位移严重偏离共识向量（残差超过视差容忍裕度 $\tau = 0.15$），对其分配的预测概率施加惩罚：
$$\mathcal{L}_{\text{motion}} = \frac{1}{M_0} \sum_{i=0}^{M_0-1} \sum_{j=0}^{M_1-1} P_{ij} \cdot \max(0, \|\mathbf{u}_{ij} - \bar{\mathbf{u}}\|_2 - \tau)^2$$
该损失能极其精准地打击同类别/跨类别的李鬼误匹配（如误把左侧白墙连到右侧白墙），迫使概率被 Sinkhorn 重新导向垃圾桶，大幅拉升 Precision。

### 3. 损失函数的代码逐行解析

```python
class ObjectMatchingLoss(nn.Module):
    def __init__(self, dustbin_weight: float = 0.3, motion_weight: float = 0.2, motion_margin: float = 0.15) -> None:
        super().__init__()
        self.dustbin_weight = float(dustbin_weight)
        self.motion_weight = float(motion_weight)
        self.motion_margin = float(motion_margin)

    def forward(self, log_P, gt_matches, unmatched_0, unmatched_1, centroids0=None, centroids1=None):
        B, M0_plus_1, M1_plus_1 = log_P.shape
        M0 = M0_plus_1 - 1
        M1 = M1_plus_1 - 1
        log_P = torch.clamp(log_P, min=-100.0, max=0.0)

        total_nll = 0.0
        total_motion = 0.0

        for b in range(B):
            matches = gt_matches[b]
            un0 = unmatched_0[b]
            un1 = unmatched_1[b]

            # 1. NLL 匹配损失与垃圾桶平衡损失
            loss_pos = sum(-log_P[b, i, j] for i, j in matches)
            loss_neg = sum(-log_P[b, i, M1] for i in un0) + sum(-log_P[b, M0, j] for j in un1)
            mean_pos = (loss_pos / len(matches)) if matches else 0.0
            mean_neg = (loss_neg / (len(un0) + len(un1))) if (un0 or un1) else 0.0
            total_nll += mean_pos + self.dustbin_weight * mean_neg

            # 2. 变换一致性位移场共识损失
            if self.motion_weight > 0 and centroids0 is not None and matches:
                c0 = centroids0[b]
                c1 = centroids1[b]
                disp_gt = torch.stack([c1[j] - c0[i] for i, j in matches], dim=0)
                mean_disp = disp_gt.mean(dim=0, keepdim=True)
                disp_all = c1.unsqueeze(0) - c0.unsqueeze(1)
                residuals = torch.norm(disp_all - mean_disp, dim=-1)
                penalty = (torch.clamp(residuals - self.motion_margin, min=0.0) ** 2).detach()
                P_real = torch.exp(log_P[b, :M0, :M1])
                total_motion += (P_real * penalty).sum() / max(float(M0), 1.0)

        return (total_nll + self.motion_weight * total_motion) / float(B)
```

### 4. 为什么该损失函数不需要负采样？
在通常对比学习（如 InfoNCE）中，必须费力构造负样本（Negative Samples）。但在 `ObjectMatchingLoss` 中**完全不需要显式负样本**：
* 归功于 Sinkhorn 的**双随机行归一化与列归一化约束**：每一行所有项加和恒为 1。
* 当损失函数拉大真实正确匹配 $P_{i, j} \to 1$ 时，**该行中所有与其它物体的错误匹配 $P_{i, k} (k \neq j)$ 以及垃圾桶概率都会被数学上自动压缩向 0**！
* 同样，当损失函数把落单物体推入垃圾桶 $P_{i, M_1} \to 1$ 时，该物体与对面所有物体的误连概率也会被强制归零。

---

## 四、使用方法与训练命令

### 训练模型与历史检查点保存：
```bash
# 从头开始训练（含位移场共识一致性损失、每 5 轮存历史快照）
python train.py \
  --model xfeat_object \
  --data-dir data/structured3d_semantic_regions \
  --save-dir checkpoints/semantic_patch_matcher \
  --epochs 20 \
  --save-interval 5 \
  --dustbin-weight 0.3 \
  --motion-weight 0.2 \
  --lr 5e-4 \
  --amp \
  --num-workers 4

# 断点恢复训练（Resume from milestone or latest checkpoint）
python train.py \
  --model xfeat_object \
  --data-dir data/structured3d_semantic_regions \
  --save-dir checkpoints/semantic_patch_matcher \
  --epochs 20 \
  --resume checkpoints/semantic_patch_matcher/epoch_010.pth \
  --motion-weight 0.2 \
  --lr 2e-4 \
  --amp
```

### 交互式验证与可视化：
```bash
# 验证指定场景对
python tools/verify_semantic_matches.py --sample-idx 0

# 开启类别约束验证
python tools/verify_semantic_matches.py --sample-idx 1 --enforce-category
```
