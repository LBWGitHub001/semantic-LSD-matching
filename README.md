# Semantic Visual Matching: 目标级与线段级跨视角匹配系统

基于深度学习与最优传输（Optimal Transport）的跨视角多层级视觉匹配系统。支持：
1. **目标级双向图匹配（"连连看"，Object-Level Bipartite Matching）**：基于 XFeat 轻量全卷积骨干与掩膜池化，解决跨视角多物体对齐与出画落单过滤；
2. **线段级几何特征匹配（Line-Level Feature Matching）**：基于 6 通道连续几何场（EDT+AFM）与线段图注意力，解决室内弱纹理无点环境下的直线配对。

---

## 目录

- [一、核心架构与统一目录结构](#一核心架构与统一目录结构)
- [二、四步统一工作流 (CLI 快速开始)](#二四步统一工作流-cli-快速开始)
  - [1. 统一生成数据集 (`generate_dataset.py`)](#1-统一生成数据集-generate_datasetpy)
  - [2. 统一训练入口 (`train.py`)](#2-统一训练入口-trainpy)
  - [3. 统一验证评测 (`eval.py`)](#3-统一验证评测-evalpy)
  - [4. 统一交互演示 (`demo.py`)](#4-统一交互演示-demopy)
- [三、核心模型库 (`models/`)](#三核心模型库-models)
  - [1. 物体级连连看网络 (`XFeatObjectMatcher`)](#1-物体级连连看网络-xfeatobjectmatcher)
  - [2. 6通道几何场线段网络 (`MaskLineFeatMatchNet`)](#2-6通道几何场线段网络-masklinefeatmatchnet)
  - [3. 基线线段 Transformer (`LineMatcherNet`)](#3-基线线段-transformer-linematchernet)
- [四、辅助工具与离线脚本 (`tools/`)](#四辅助工具与离线脚本-tools)
- [五、Python API 调用示例](#五python-api-调用示例)
- [六、基准评测表现 (Benchmark)](#六基准评测表现-benchmark)

---

## 一、核心架构与统一目录结构

代码库已全面解耦为**根目录统一对外接口**、**分层模型库（`models/`）**、**数据集抽象（`datasets/`）**与**辅助工具包（`tools/`）**：

```text
semantic-LSD-matching/
├── generate_dataset.py          # [统一数据生成入口] 支持 object / line / all 任务与维度2随机材质
├── train.py                     # [统一训练入口] 支持 --model [xfeat_object | mask_line | linefeat]
├── eval.py                      # [统一验证评测] 盲测场景基准评分与混淆指标统计
├── demo.py                      # [统一可视化演示] 跨视角匹配连线即时可视化
│
├── models/                      # [核心网络模型库] 按任务与模型物理分包
│   ├── __init__.py              # 统一模型工厂 (build_model, build_loss, MODEL_REGISTRY)
│   ├── object_matcher/          # 物体级匹配网络 ("连连看")
│   │   ├── __init__.py
│   │   ├── xfeat_object_matcher.py # XFeatBackbone + MaskPooling + GraphTransformer + Sinkhorn
│   │   └── loss.py              # ObjectMatchingLoss (带垃圾桶 NLL)
│   └── line_matcher/            # 线段级匹配网络
│       ├── __init__.py
│       ├── mask_line_matcher.py # MaskLineFeatMatchNet (6通道 EDT+AFM 几何场)
│       ├── line_transformer.py  # LineMatcherNet (基线线段 Transformer)
│       ├── geometric_field.py   # 欧氏距离场与朝向场生成器
│       └── loss.py              # MaskedMatrixLoss / LineMatchingLoss
│
├── datasets/                    # [统一数据集与数据管道]
│   ├── __init__.py
│   ├── object_dataset.py        # ObjectMatchingDataset, Bipartite MNN 解析器与可视化
│   └── line_dataset.py          # LineFeatPairDataset 与线段批处理
│
├── tools/                       # [辅助工具脚本]
│   ├── cache_line_features.py   # 离线特征缓存工具 (~50x 训练加速)
│   ├── build_line_gt_dataset.py # 3D 物理线段真值构建脚本
│   ├── download_structured3d.py # 真实场景下载辅助
│   ├── unzip_structured3d.py    # 真实场景解压辅助
│   ├── verify_toy_task.py       # 最小 toy 验证
│   ├── sync_push.sh             # 远程同步脚本
│   └── sync_pull.sh
│
├── LineFeat/                    # [向下兼容层] 保证老代码旧脚本 100% 兼容运行
├── checkpoints/                 # 最佳训练权重保存目录
├── outputs/                     # 可视化连线结果与评测导出
└── README.md                    # 本文档
```

---

## 二、四步统一工作流 (CLI 快速开始)

### 1. 统一生成数据集 (`generate_dataset.py`)
支持一键渲染多视角室内场景，支持“维度 2”材质全随机化（彻底破除颜色作弊）与 3D 法线立体光照：

```bash
# 生成物体级“连连看”数据集 (50个场景，每场景8视点，带实例掩膜与匹配真值)
python generate_dataset.py --task object --num-scenes 50 --random-colors

# 生成线段级多视角数据集
python generate_dataset.py --task line --num-scenes 50

# 一次性生成全套多模态数据 (掩膜 + 线段 + 相机位姿)
python generate_dataset.py --task all --num-scenes 50
```

### 2. 统一训练入口 (`train.py`)
通过 `--model` 参数自由决定训练哪一个模型，自动适配超参数、数据管道与权重路径：

```bash
# 训练物体级“连连看”模型 (XFeatObjectMatcher，推荐 GPU + AMP)
python train.py --model xfeat_object --epochs 12 --lr 5e-4 --amp

# 训练 6 通道几何场线段匹配模型 (MaskLineFeatMatchNet)
python train.py --model mask_line --epochs 8 --lr 2e-4 --amp

# 训练基线线段 Transformer (LineMatcherNet)
python train.py --model line_transformer --epochs 8 --lr 2e-4
```

### 3. 统一验证评测 (`eval.py`)
在未见测试集（Test Set）上进行全量盲测，输出准确率、召回率、F1 分数以及垃圾桶过滤率：

```bash
# 评测物体级模型在测试集上的表现
python eval.py --model xfeat_object --checkpoint checkpoints/xfeat_object_matcher/best_model.pth --split test

# 导出测试集前 5 对视角的可视化连线图
python eval.py --model xfeat_object --num-vis 5 --out-dir outputs/eval_vis
```

### 4. 统一交互演示 (`demo.py`)
任意指定场景目录与两个视点编号，即时查看匹配连线结果：

```bash
python demo.py --model xfeat_object --scene data/synthetic_object_matching_scenes/scene_0045 --v0 0 --v1 3 --out outputs/demo_result.jpg
```

---

## 三、核心模型库 (`models/`)

### 1. 物体级连连看网络 (`XFeatObjectMatcher`)
- **适用场景**：多物体跨视角对齐（桌子、椅子、门、窗、挂画、地毯），抗 180° 对视大旋转；
- **骨干网络**：XFeat 超轻量残差卷积（参数量仅 ~0.25M，单对推理耗时 **~2.2 ms**）；
- **掩膜池化（Mask Pooling）**：直接在全图特征图上提取局部物体的外观特征向量；
- **空间位置编码（Geometry MLP）**：输入物体归一化边界框与中心点坐标，赋予宏观空间布局先验；
- **图注意力与最优传输**：通过自注意力融合“邻里关系”（如门在桌子右侧），通过 Log-Sinkhorn 与垃圾桶（Dustbin）严格实现 1-to-1 独占匹配并过滤出画物体。

### 2. 6通道几何场线段网络 (`MaskLineFeatMatchNet`)
- **适用场景**：白墙走廊、室内弱纹理无角点环境下的规整人造直线配对；
- **特征场设计**：输入不仅包含 RGB，更融合欧氏距离变换场（EDT）与二倍角吸引朝向场（AFM），彻底消除端点起止二义性；
- **输出规范**：直接输出规范化 $(B, 255, 255)$ 概率矩阵，严格保持数学置换等变性。

### 3. 基线线段 Transformer (`LineMatcherNet`)
- **适用场景**：沿线段插值采样视觉特征 + 几何位置编码的端到端基线。

---

## 四、辅助工具与离线脚本 (`tools/`)

所有辅助工具与离线加速脚本均归拢至 `tools/` 目录：
- `tools/cache_line_features.py`：预先提取全图特征保存到内存缓存，将线段模型训练速度提升 **~50 倍**；
- `tools/build_line_gt_dataset.py`：基于 3D CAD/室内真实点云投影生成线段真值对应；
- `tools/download_structured3d.py` & `tools/unzip_structured3d.py`：`Structured3D` 真实室内场景数据集下载与自动解压；
- `tools/sync_push.sh` & `tools/sync_pull.sh`：本地工作站与远程算力 GPU 服务器一键免密同步代码与权重。

---

## 五、Python API 调用示例

### 1. 物体级“连连看”匹配 API (5 行核心调用)

```python
import torch
from models import build_model
from datasets import predict_bipartite_matches

# 1. 构建并加载模型
model = build_model("xfeat_object").cuda()
ckpt = torch.load("checkpoints/xfeat_object_matcher/best_model.pth")
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2. 前向推理 (img0, img1: [1, 3, H, W], masks0, masks1: [1, M, H, W], bboxes0, bboxes1: [1, M, 4])
with torch.no_grad():
    out = model(img0, img1, masks0, bboxes0, masks1, bboxes1)

# 3. 解析双向互斥配对与垃圾桶
matches, unmatched_0, unmatched_1 = predict_bipartite_matches(out["prob_matrix"][0], thresh=0.2)
print("配对结果 [(idx0, idx1)]:", matches)
```

---

## 六、基准评测表现 (Benchmark)

在 **10 个未见独立室内测试场景（140 对盲测大视差视角，覆盖 45°、90°、180° 对视）** 上实测表现：

| 评估指标 | `XFeatObjectMatcher` (物体连连看) | `MaskLineFeatMatchNet` (6通道线段网络) |
| :--- | :---: | :---: |
| **匹配精准率 (Precision)** | **100.00%** | **47.82% (MNN)** |
| **匹配召回率 (Recall)** | **100.00%** | **39.80% (Top-1)** |
| **F1 综合评分** | **100.00%** | **43.37%** |
| **垃圾桶识别率 (Dustbin Acc)**| **100.00%** (精准剔除出画物体) | — |
| **单样本前向延迟 (Latency)** | **~2.2 ms** (极速实时) | **~15.0 ms** |
| **抗对称性混淆能力** | **极强** (结合空间相对拓扑) | 受室内 4 桌腿对称性限制 |
| **最佳适用层级** | **宏观家具与构件级关联** | **微观边缘几何与点云配准** |

---

## 许可证 (License)

本项目采用 MIT 协议开源。
