# LineMatcherNet：基线线段 Transformer 匹配网络

本目录包含 **`LineMatcherNet`** 模型的完整实现。该模型采用传统计算机视觉中经典的 1D 离散沿线插值采样（Discrete Equidistant Sampling）与几何特征融合策略，作为线特征匹配的基线（Baseline）参考模型。

---

## 目录
- [一、模型原理与采样机制](#一模型原理与采样机制)
- [二、网络架构流程](#二网络架构流程)
- [三、损失函数详解](#三损失函数详解)
- [四、模型优缺点与适用场景对比](#四模型优缺点与适用场景对比)
- [五、快速调用与流水线](#五快速调用与流水线)

---

## 一、模型原理与采样机制

与连续几何场模型不同，`LineMatcherNet` 将线段视作离散几何图元：
1. **1D 离散沿线采样 (`LineVisualSampler`)**：
   - 对任意检测到的直线段 $l_i = (x_1, y_1, x_2, y_2)$，沿线段均匀插值采样 $K$ 个关键点（默认 $K=5$ 或 $K=7$）；
   - 在 CNN 外观特征图上通过双线性插值（Bilinear Grid Sampling）采集这 $K$ 个点的特征向量，并通过平均池化聚合为线段外观向量：
     $$f_i^{\text{sample}} = \frac{1}{K} \sum_{k=1}^K F(p_k) \in \mathbb{R}^{D}$$
2. **5 维几何位置编码 (`LineGeometryEncoder`)**：
   - 提取线段的归一化中点坐标、长度以及方向正余弦：
     $$g_i = \left[ \frac{x_1+x_2}{2W}, \frac{y_1+y_2}{2H}, \frac{\|l_i\|}{\sqrt{W^2+H^2}}, \cos \theta, \sin \theta \right] \in \mathbb{R}^5$$
   - 经过 2 层带 LayerNorm 的 MLP 映射为几何 Token，并与视觉特征相加。

---

## 二、网络架构流程

```text
两幅输入图像 (I0, I1) + 检测到的线段坐标集 (L0, L1)
  │
  ├── 1. CNN 骨干提取局部外观特征图
  ├── 2. 沿线均匀插值采样 K 个关键点并做平均池化 -> 得到线段视觉特征
  ├── 3. 几何 MLP 编码中点、长度与朝向 -> 得到线段几何特征
  ├── 4. 融合生成线段初始 Token (N0, D) 与 (N1, D)
  ├── 5. 堆叠多层 Self-Attention (图内拓扑) 与 Cross-Attention (跨图交互)
  └── 6. Log-Sinkhorn 最优传输输出 (N0+1, N1+1) 匹配概率矩阵
```

---

## 三、损失函数详解

模型使用 [`LineMatchingLoss`](file:///home/lbw/Project/semantic-LSD-matching/models/line_transformer/loss.py) 进行监督：

### 1. 损失计算方式
在经过最优传输得到对数分配矩阵 $\log P$ 后，对真实同名线段匹配对计算负对数似然（NLL）：
$$\mathcal{L} = -\frac{1}{M} \sum_{k=0}^{M-1} \log P_{k, k}$$
- 强制模型提高真值线段间的相似度分数，同时 Sinkhorn 矩阵的行/列和为 1 约束会自动抑制与其它不相关线段的误匹配概率。

---

## 四、模型优缺点与适用场景对比

| 特性 | `LineMatcherNet` (本基线模型) | `MaskLineFeatMatchNet` (6通道几何场) |
| :--- | :---: | :---: |
| **特征提取方式** | 离散 1D 关键点沿线采样 ($K$ 点) | 连续 2D 欧氏距离场 (EDT) + 朝向场 (AFM) |
| **视锥截断鲁棒性** | 弱（端点被视野截断时采样点发生物理错位） | 强（像素距离场天然抵抗端点截断） |
| **无纹理白墙鲁棒性** | 较弱（易受平行线与对称线混淆） | 强（距离场自带空间拓扑距离先验） |
| **计算复杂度** | 较低 (~1.40 M 参数量) | 中等 (~1.85 M 参数量) |
| **推荐适用场景** | 线段纹理丰富、视差较小的室外/建筑匹配 | 室内弱纹理白墙、大视差、长线段截断匹配 |

---

## 五、快速调用与流水线

在 Python 中一键启动推理流水线：
```python
from models.line_transformer import LineMatchPipeline

pipeline = LineMatchPipeline.from_pretrained("checkpoints/linefeat/best_model.pth")
result = pipeline.match_images("image_0.png", "image_1.png")

print(f"匹配成功线段对数: {result.num_matches}")
result.visualize("output_line_matches.jpg")
```
