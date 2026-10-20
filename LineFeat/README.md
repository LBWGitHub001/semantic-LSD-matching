# LineFeat: 跨视角 2D 直线特征匹配算法包

`LineFeat` 是本项目的核心算法库，完全自包含、零检测器依赖，专注于跨视角预检测 2D 线段的端到端高精度几何匹配。

系统默认主打模型为 **`MaskLineFeatMatchNet`**（6 通道连续几何场 + 掩膜引导池化 + 规范 $255 \times 255$ 匹配概率矩阵直出），同时保留了经典的轻量基线模型 **`LineMatcherNet`**。

---

## 模块结构概览

```text
LineFeat/
├── __init__.py           # 算法包门面，导出全部核心模型、损失与工具函数
├── mask_line_matcher.py  # 【默认主打模型】MaskLineFeatMatchNet (6通道几何场 + 255x255规范矩阵)
├── model.py              # 【基准模型】LineMatcherNet (ResNet50 + Line-Transformer + Sinkhorn)
├── geometric_field.py    # 高性能连续 EDT 与 AFM 几何场生成器 (OpenCV C++ 底层加速)
├── pipeline.py           # 高层推断接口 (LineMatchPipeline, LineMatchResult)
├── loss.py               # 监督损失函数 (MaskedMatrixLoss, LineMatchingLoss)
├── dataset.py            # PyTorch 标准数据集加载器 (LineFeatPairDataset)
├── matcher.py            # 连线可视化与互为最近邻 (MNN) 过滤工具
├── ground_truth.py       # 3D线框投影与深度 Z-Buffer 遮挡剔除真值引擎
└── read_structured3d.py  # Structured3D 场景位姿与相机参数解析器
```

---

## 核心类与 API 详细使用指南

### 1. 默认主打模型：`MaskLineFeatMatchNet`
> 详细设计原理与数学推导请参阅 [📘 docs/MaskLineFeatMatchNet.md](../docs/MaskLineFeatMatchNet.md)。

* **特点**：
  - 输入 6 通道多模态张量：`RGB(3) + 归一化欧氏距离场 EDT(1) + 连续朝向场 AFM(2)`；
  - 彻底规避 0~255 标量数字掩膜引发的序数偏置（Ordinal Bias）；
  - 使用 $(\sin 2\theta, \cos 2\theta)$ 双倍角连续投影，消除线段端点颠倒翻转不确定性；
  - 输出恒定形状为 $(B, 255, 255)$ 的置换等变匹配概率矩阵。

```python
import torch
from LineFeat import MaskLineFeatMatchNet

# 初始化模型
model = MaskLineFeatMatchNet(d_model=128, num_layers=4).cuda().eval()

# 准备输入 (B=1, RGB 3通道张量由网络自动生成 EDT/AFM 扩展为 6 通道)
img0 = torch.randn(1, 3, 480, 640).cuda()
img1 = torch.randn(1, 3, 480, 640).cuda()
lines0 = torch.rand(1, 10, 4).cuda() * 400.0  # (B, N0, 4)
lines1 = torch.rand(1, 12, 4).cuda() * 400.0  # (B, N1, 4)

with torch.no_grad():
    out = model(img0, img1, lines0, lines1)

# 获取 (B, 255, 255) 规范矩阵与有效子矩阵
matrix_255 = out["matrix_255"]        # shape: (1, 255, 255)
p_sub = out["P_sub"]                  # shape: (1, 10, 12)
```

#### 模型参数训练与冻结划分表
| 模块 | 参数量 | 训练状态 | 职责 |
| :--- | :---: | :---: | :--- |
| `backbone` (ResNet-50) | 1.44 M | **❄️ 冻结** | ImageNet 边缘与纹理提取，防止白墙表征坍塌，支撑 50 倍缓存加速 |
| `sampler.proj` | 0.08 M | **🔥 训练** | 512 维到 128 维自适应视觉特征降维 |
| `geometry_encoder` | 0.01 M | **🔥 训练** | 坐标、长度与 $(\sin 2\theta, \cos 2\theta)$ 几何位置编码 |
| `layers` (Transformer) | 0.80 M | **🔥 训练** | 图内拓扑结构理解 + 跨视角候选线对比匹配 |
| `proj0`, `proj1` | 0.03 M | **🔥 训练** | 最终 L2 归一化内积流形映射 |
| `log_optimal_transport` | 0 (无参数) | **⚙️ 算法** | 对数域 Sinkhorn 求解最优双射分配（带 Dustbin） |

---

### 2. 高层推理流水线：`LineMatchPipeline`

为工程落地与算法调用提供最简洁的 Python 统一接口。**默认即自动加载并运行 `MaskLineFeatMatchNet`**。

```python
from LineFeat import LineMatchPipeline
import numpy as np

# 1. 初始化纯匹配流水线 (自动优先使用 GPU，默认 model_type='mask')
pipeline = LineMatchPipeline("checkpoints/linefeat/best_model.pth")

# 2. 外部传入已知线段坐标 (N, 4) [x1, y1, x2, y2]
lines0 = np.array([[50, 100, 200, 100], [80, 50, 80, 300]], dtype=np.float32)
lines1 = np.array([[52, 98, 202, 99], [79, 45, 81, 290]], dtype=np.float32)

# 3. 运行匹配 (零在线检测器，直接匹配给定坐标)
result = pipeline.match(
    image0="path/to/image0.png",
    image1="path/to/image1.png",
    lines0=lines0,
    lines1=lines1,
    threshold=0.70,
)

# 4. 获取直出的 255x255 规范概率矩阵
print("Canonical Matrix Shape:", result.matrix_255.shape)  # (255, 255)

# 5. 遍历匹配结果
for rank, (i, j, score) in enumerate(result.matches):
    print(f"Match #{rank+1}: Line0[{i}] <--> Line1[{j}] | 置信度得分: {score:.3f}")

# 6. 保存高质量彩色连线可视化图
result.save_visualization("outputs/pure_matching/match_vis.jpg")
```

---

### 3. 高性能几何场生成器：`geometric_field.py`

利用 OpenCV C++ 底层优化，将任意 2D 线段坐标转换为高质量的连续距离与朝向物理场：

```python
from LineFeat import compute_geometry_field_np, create_6ch_input_tensor
import cv2, torch

# 计算 numpy 版本的几何场 (H, W, 3): 通道0为 EDT 距离场，通道1-2为 AFM 朝向场
lines_np = np.array([[100, 150, 300, 150]], dtype=np.float32)
geom_field = compute_geometry_field_np(lines_np, height=480, width=640)

# 在 PyTorch 张量层面将 (B, 3, H, W) RGB 自动拼接为 (B, 6, H, W)
rgb_tensor = torch.randn(1, 3, 480, 640)
lines_tensor = torch.from_numpy(lines_np).unsqueeze(0)
input_6ch = create_6ch_input_tensor(rgb_tensor, lines_tensor)  # (1, 6, 480, 640)
```

---

### 4. 监督损失函数：`loss.py`

提供两套面向不同任务的监督损失函数：

#### `MaskedMatrixLoss`（专为规范 255x255 矩阵定制）
结合带 Dustbin 的对数 Sinkhorn 负对数似然（NLL）与有效子区域的掩膜二值交叉熵（Masked BCE）：
```python
from LineFeat import MaskedMatrixLoss

criterion = MaskedMatrixLoss(lambda_bce=1.0)
# matrix_255: (B, 255, 255), log_assignment: (B, N0+1, N1+1), gt_pairs: list of (idx0, idx1)
loss, metrics = criterion(out["matrix_255"], out["log_assignment"], gt_pairs, N0=10, N1=12)
```

#### `LineMatchingLoss`（基线 Sinkhorn 对数似然损失）
针对已知排列真值的经典最优传输负对数似然损失，支持自动统计 Precision 与 Recall。

---

### 5. 数据集加载：`dataset.py`

从预计算的 `line_gt_dataset` 快速加载真值线对：

```python
from torch.utils.data import DataLoader
from LineFeat import LineFeatPairDataset, collate_line_pairs

dataset = LineFeatPairDataset(
    index_path="data/line_gt_dataset/index.json",
    split="train",  # 支持 'train', 'val', 'test'
    min_matches=3,
    load_images=True,
    load_cameras=False,
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)
for sample in loader:
    img0 = sample["image0"]      # (1, 3, H, W)
    img1 = sample["image1"]      # (1, 3, H, W)
    lines0 = sample["gt_lines0"]  # (1, M, 4)
    lines1 = sample["gt_lines1"]  # (1, M, 4)
    M = sample["num_matches"]    # 真值对数
```
