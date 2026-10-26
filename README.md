# Semantic Visual Matching: 目标级与线段级跨视角匹配系统与可视化管理平台

基于深度学习与最优传输（Optimal Transport）的跨视角多层级视觉匹配系统。支持：
1. **线段与语义区域混合匹配（`XFeatLineMatcher`）**：基于多尺度 FPN 特征金字塔、5D 局部线段几何编码与位移场共识边界损失（Motion Consensus Loss），在真实室内弱纹理环境下实现高精度直线/区域配准（**Structured3D 测试集 F1 达 79.22%**）；
2. **目标级双向图匹配（"连连看"，`XFeatObjectMatcher`）**：基于 XFeat 轻量全卷积骨干与掩膜池化，解决跨视角宏观多物体对齐与出画落单过滤；
3. **线段级几何特征匹配（`MaskLineFeatMatchNet`）**：基于 6 通道连续几何场（EDT+AFM）与线段图注意力，解决端点起止二义性；
4. **一站式可视化管理开发工作台（Web Studio）**：内建硬件资源监视器（CPU/内存/GPU/显存）、训练工作台（实时日志流与 Loss 曲线）、基准评测、双排诊断画廊与**交互式模型手册与显存/算力计算器**。

---

## 目录与专题技术文档

- 📚 **专题技术文档**：
  - [**核心匹配网络模型技术详解** (`docs/MODELS.md`)](docs/MODELS.md)：**【新】**全面详解 `XFeatLineMatcher`（多尺度金字塔 + 5D 几何 + 位移共识损失）、`XFeatObjectMatcher`、`MaskLineFeatMatchNet`、`LineMatcherNet` 的网络结构、数学推导、参数规模、推理延迟与 Benchmark。
  - [**全局通用文件与工作流使用指南** (`docs/GENERAL_USAGE.md`)](docs/GENERAL_USAGE.md)：`app.py`（Web 管理后台）、`review_runs.py`（实验复盘 CLI）、`train.py`、`eval.py`、`generate_dataset.py` 全参数配置与生产级最佳实践。
  - [**工厂方法体系与多模态组件技术手册** (`docs/FACTORIES_AND_COMPONENTS.md`)](docs/FACTORIES_AND_COMPONENTS.md)：四大工厂模式（数据生成工厂、模型工厂、数据管道工厂、评测工厂）、LSD 直线核心组件、5D 几何描述子与上下双排对比诊断规范。
  - [**数据集架构与格式技术规范** (`docs/DATASETS.md`)](docs/DATASETS.md)：`datasets/generators/` 模块化生成管线、合成物体级（维度2材质光照）、合成线段级与真实 Structured3D CAD 数据集的标注格式与目录组织。

- ⚡ **快速导航**：
  - [一、核心架构与统一目录结构](#一核心架构与统一目录结构)
  - [二、五步统一工作流 (CLI 快速开始)](#二五步统一工作流-cli-快速开始)
  - [三、Web 可视化管理工作台 (Web Studio)](#三web-可视化管理工作台-web-studio)
  - [四、核心模型库 (`models/`)](#四核心模型库-models)
  - [五、Python API 调用示例](#五python-api-调用示例)
  - [六、基准评测表现 (Benchmark)](#六基准评测表现-benchmark)

---

## 一、核心架构与统一目录结构

代码库全面采用模块化高内聚设计，包含统一命令行入口、Web 服务端、分层模型库（`models/`）、评测套件（`evaluation/`）与数据集抽象（`datasets/`）：

```text
semantic-LSD-matching/
├── app.py                       # [Web Studio 服务端] 基于 Tornado 的统一可视化管理平台后端
├── launch_ui.sh                 # [一键启动脚本] 自动激活 Conda 环境并拉起 Web 工作台 (Port 7860)
├── review_runs.py               # [实验复盘 CLI] 扫描实验 Checkpoint、横向对比多模型、生成复盘报告
├── generate_dataset.py          # [统一数据生成入口] 支持 object / line / semantic_line / structured3d_lines
├── train.py                     # [统一训练入口] 支持 xfeat_line_matcher / xfeat_object / mask_line 等架构
├── eval.py                      # [统一验证评测] 盲测场景基准评分、混淆指标统计与双排诊断图导出
├── demo.py                      # [统一可视化演示] 跨视角匹配连线即时可视化
│
├── web/                         # [Web Studio 前端] 现代化零依赖深色主题前端界面
│   ├── templates/index.html     # 六大主功能标签页骨架 (看板、训练、评测、画廊、数据生成、模型手册)
│   └── static/
│       ├── app.css              # 现代化 Responsive UI 样式表
│       └── app.js               # 状态管理、图表绘制、实时日志流、交互式计算器与复制逻辑
│
├── models/                      # [核心网络模型库]
│   ├── __init__.py              # 统一模型工厂 (build_model, build_loss, MODEL_REGISTRY)
│   ├── object_matcher/          # 物体/区域及线段多模态匹配网络
│   │   ├── xfeat_line_matcher.py   # XFeatLineMatcher (多尺度 FPN + 5D 几何 + 位移共识)
│   │   ├── xfeat_object_matcher.py # XFeatObjectMatcher (单尺度/金字塔物体连连看)
│   │   └── loss.py                 # ObjectMatchingLoss (带垃圾桶 NLL + 位移场共识边界损失)
│   ├── mask_line_matcher/       # 6通道连续几何场线段匹配网络 (MaskLineFeatMatchNet)
│   └── line_transformer/        # 经典线段 Transformer 网络 (LineMatcherNet)
│
├── evaluation/                  # [模块化基准评测套件]
│   ├── __init__.py              # 评测工厂与统一 evaluate() 接口
│   ├── model_factory.py         # 智能权重拓扑探测与模型工厂 (兼容历史与最新权重)
│   ├── dataset_factory.py       # 数据管道工厂 (自动嗅探 Structured3D 与合成数据集)
│   ├── evaluators.py            # ObjectMatchingEvaluator / LineMatchingEvaluator 与双排图绘制
│   └── checkpoint.py            # 检查点自动解析与路径映射
│
├── datasets/                    # [统一数据集与数据管道]
│   ├── __init__.py
│   ├── generators/              # 模块化数据集生成器 (procedural_room.py 等)
│   ├── object_dataset.py        # ObjectMatchingDataset 与 Bipartite MNN 解析器
│   ├── line_dataset.py          # LineFeatPairDataset 与线段批处理
│   ├── ground_truth.py          # 3D 物理线段真值投影与 Z-Buffer 遮挡剔除
│   └── read_structured3d.py     # Structured3D 相机与场景标注解析器
│
├── tests/                       # [自动化测试套件]
│   ├── test_xfeat_line_matcher.py # 多模态线段网络前向/梯度/小图单元测试
│   ├── test_factories.py        # 工厂类拓扑自检与画布回归测试
│   └── run_unit_tests.py        # 矩阵等变性与梯度流数学验证
│
├── checkpoints/                 # 实验运行输出、权重与历史复盘目录
├── outputs/                     # 双排可视化诊断图与评测导出
└── README.md                    # 本技术手册
```

---

## 二、五步统一工作流 (CLI 快速开始)

### 1. 统一生成数据集工厂 (`generate_dataset.py`)
支持基于可插拔组件的工厂模式，一键渲染多视角室内场景并提取 **LSD 亚像素直线**、实例掩膜与相机位姿，支持“维度 2”材质全随机化与 3D 法线立体光照：

```bash
# 1. 一键生成同时包含语义目标与 LSD 直线的数据集 (推荐)
python generate_dataset.py --task semantic_line --num-scenes 50 --views-per-scene 8 --out-dir data/synthetic_semantic_lines

# 2. 从 Structured3D 真实场景提取 CAD 3D 直线与共视区域切片 (带 Z-Buffer 遮挡剔除)
python generate_dataset.py --task structured3d_lines --s3d-dir Structured3D_perspective_full_00/Structured3D --out-dir data/xfeat_line_matcher --max-scenes 50
```

### 2. 统一训练入口 (`train.py`)
通过 `--model` 参数自由决定训练哪一个模型，自动适配超参数、数据管道与权重路径：

```bash
# 训练主推多模态线段网络 (XFeatLineMatcher, 包含 FPN 金字塔与位移共识损失)
python train.py --model xfeat_line_matcher \
  --data-dir data/xfeat_line_matcher \
  --epochs 15 \
  --lr 5e-4 \
  --dustbin-weight 0.3 \
  --motion-weight 0.2 \
  --min-area-ratio 0.40 \
  --pyramid \
  --amp \
  --save-interval 5

# 训练物体级“连连看”基线模型 (XFeatObjectMatcher)
python train.py --model xfeat_object --epochs 20 --lr 5e-4 --amp
```

### 3. 统一评测工厂 (`eval.py`)
采用智能模型工厂与评测器工厂，**自动嗅探 Checkpoint 结构（100% 兼容历史与最新权重）**，并在导出诊断图时采用**“上下双排布局”（上排模型预测诊断 + 下排真实真值基准）**：

```bash
# 评测检查点，自动识别网络结构 (FPN 金字塔 / 几何MLP / 单尺度)，并导出 50 对双排对比图
python eval.py \
  --checkpoint checkpoints/xfeat_line_matcher/20261023_233349/best_model.pth \
  --data-dir data/xfeat_line_matcher \
  --out-dir outputs/xfeat_line_eval_vis \
  --num-vis 50
```

### 4. 实验复盘与横向对比 (`review_runs.py`)
一键汇总全项目历史训练记录，输出横向对比表格与综合分析报告：

```bash
# 查看所有历史实验概览与最佳指标
python review_runs.py --list

# 对比两次实验的超参数与收敛表现
python review_runs.py --compare \
  checkpoints/xfeat_object_matcher/20261022_225315 \
  checkpoints/xfeat_line_matcher/20261023_233349
```

### 5. 交互演示 (`demo.py`)
任意指定场景目录与两个视点编号，即时查看匹配连线结果：

```bash
python demo.py --model xfeat_line_matcher --scene data/synthetic_multiview_scenes/scene_0000 --v0 0 --v1 1 --out outputs/demo_result.jpg
```

---

## 三、Web 可视化管理工作台 (Web Studio)

工程提供了开箱即用的 Web 管理系统，在单界面内打通**训练监控、实验复盘、基准评测、画廊诊断与模型手册**。

### 1. 一键启动

```bash
# 方式一：一键脚本 (自动检查依赖与 Conda 环境)
./launch_ui.sh

# 方式二：Python 直接启动 (指定监听端口)
python app.py --port 7860
```
启动后在浏览器中访问：**`http://localhost:7860`**

### 2. 六大核心功能看板

```
+-------------------------------------------------------------------------------------------------------------+
|  [⚡ Semantic LSD Matching Studio v2.1]                [GPU: RTX 4060 Laptop (8GB)]  [● 系统就绪 (Ready)]    |
+-------------------------------------------------------------------------------------------------------------+
|  [📊 实验复盘看板]  [🚀 训练工作台]  [🧪 评测基准]  [🖼️ 结果可视化画廊]  [📦 数据集生成]  [📖 模型手册与交互文档]  |
+-------------------------------------------------------------------------------------------------------------+
```

1. **📊 实验复盘看板 (Dashboard)**：汇总实验历史、最佳测试集 F1 评分、超参数清单，支持多实验侧向勾选对比并自动生成复盘 Markdown 报告与收敛对比曲线；
2. **🚀 训练工作台 (Training Studio)**：
   - 表单化配置模型、学习率、垃圾桶权重、位移共识权重、FPN 金字塔开关；
   - 包含**硬件资源监视器（CPU 负载、系统内存、GPU 占用率、显存使用量）**，每秒动态心跳刷新；
   - 实时流式捕获终端日志并同步动态绘制训练 Loss 与验证集 F1 曲线，支持一键优雅中止训练；
3. **🧪 评测基准中心 (Evaluation Benchmark)**：支持对任意历史 Checkpoint 执行盲测评测，即时显示混淆矩阵（TP, FP, FN, 跨类别误配率）；
4. **🖼️ 结果可视化画廊 (Gallery)**：流式瀑布流展示上下双排诊断图，支持快捷键翻页查看与单图全屏高分辨率质检；
5. **📦 数据集生成工厂 (Dataset Studio)**：网页端一键触发合成数据集与 Structured3D 真实室内数据集切片生成；
6. **📖 模型手册与交互文档 (Models Guide & Calculator)**：
   - **交互式架构探索器**：点击流水线各阶段节点（Backbone、Line MLP、Fusion、GNN、Sinkhorn），实时查看张量形状、数学公式与源码定位；
   - **参数量与显存/算力计算器**：拖动 $d_{model}$、层数、头数与匹配候选数滑块，实时计算网络参数量、FLOPs、训练/推理显存占用与预期 FPS；
   - **代码速查 cheatsheet**：一键复制 Python API、训练 CLI、评测 CLI、数据生成 CLI 与复盘 CLI。

---

## 四、核心模型库 (`models/`)

### 1. 多模态区域直线网络 (`XFeatLineMatcher`，当前推荐主力模型)
- **适用场景**：真实室内复杂场景弱纹理白墙、家具边缘与多视角区域对齐；
- **特征骨干**：`XFeatPyramidBackbone`，提取 P1 ($H/2$), P2 ($H/4$), P3 ($H/8$) 三级特征金字塔；
- **几何编码**：5D 空间归一化直线描述子 $[x_1, y_1, x_2, y_2, \text{length}]$；
- **损失设计**：负对数似然（NLL）+ 垃圾桶惩罚 + **位移场刚体共识边界损失（Motion Consensus Loss，Margin=0.15）**；
- **实测表现**：Structured3D 测试集（340 对独立场景视角）**F1 达 79.22%（精准率 84.21%）**。

### 2. 物体级连连看网络 (`XFeatObjectMatcher`)
- **适用场景**：多物体跨视角对齐（桌子、椅子、门、窗、挂画、地毯），抗 180° 对视大旋转；
- **骨干网络**：XFeat 超轻量残差卷积（参数量仅 ~0.25M，单对推理耗时 **~2.2 ms**）；
- **掩膜引导池化**：直接在全图特征图上执行加权池化提取局部物体特征；
- **图注意力与最优传输**：交替自注意力与交叉注意力融合邻里关系，Log-Sinkhorn 严密过滤出画物体。

### 3. 6通道几何场线段网络 (`MaskLineFeatMatchNet`)
- **适用场景**：白墙走廊、室内弱纹理无角点环境下的规整人造直线配对；
- **几何场设计**：融合欧氏距离变换场（EDT）与二倍角吸引朝向场（AFM），彻底消除端点起止二义性；
- **数学性质**：严格保持行/列置换等变性（误差 $< 10^{-7}$）。

### 4. 基线线段 Transformer (`LineMatcherNet`)
- **适用场景**：沿线段等距插值采样视觉特征 + 几何位置编码的端到端基线。

---

## 五、Python API 调用示例

### 1. 多模态直线与区域匹配推理 API (XFeatLineMatcher)

```python
import torch
from models import build_model
from datasets import predict_bipartite_matches

# 1. 实例化模型并加载权重
model = build_model("xfeat_line_matcher", d_model=128, num_layers=2, nhead=4).cuda()
ckpt = torch.load("checkpoints/xfeat_line_matcher/20261023_233349/best_model.pth")
model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
model.eval()

# 2. 准备图像张量与 5D 直线几何特征
# patches0, patches1: (1, N, 3, 112, 112), lines0, lines1: (1, N, 5)
with torch.no_grad():
    out = model(patches0, lines0, patches1, lines1)

# 3. 提取双向互斥配对与垃圾桶过滤
matches, unmatched_0, unmatched_1 = predict_bipartite_matches(out["prob_matrix"][0], thresh=0.2)
print("成功配对索引对 [(idx0, idx1)]:", matches)
```

### 2. 物体级“连连看”匹配 API (XFeatObjectMatcher)

```python
import torch
from models import build_model
from datasets import predict_bipartite_matches

model = build_model("xfeat_object", d_model=128, num_layers=4).cuda()
ckpt = torch.load("checkpoints/xfeat_object_matcher/best_model.pth")
model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
model.eval()

with torch.no_grad():
    out = model(img0, img1, masks0, bboxes0, masks1, bboxes1)

matches, unmatched_0, unmatched_1 = predict_bipartite_matches(out["prob_matrix"][0], thresh=0.2)
print("物体配对结果:", matches)
```

---

## 六、基准评测表现 (Benchmark)

### 1. 真实室内 Structured3D 测试集评测 (340 对盲测大视差视角)

| 评估配置 / 模式 | 匹配精准率 (Precision) | 匹配召回率 (Recall) | F1-Score | 垃圾桶过滤召回 (Dustbin Recall) |
| :--- | :---: | :---: | :---: | :---: |
| **`XFeatLineMatcher` (零类别先验纯外观)** | **77.32%** | **73.40%** | **75.31%** | **76.61%** |
| **`XFeatLineMatcher` (语义类别先验约束)** | **84.21%** | **73.69%** | **78.60%** (最高 79.22%) | **77.10%** |

### 2. 合成多视角室内场景盲测 (10 场景 140 对视角，含 45°、90°、180° 对视)

| 评估指标 | `XFeatObjectMatcher` (物体连连看) | `MaskLineFeatMatchNet` (6通道线段网络) |
| :--- | :---: | :---: |
| **匹配精准率 (Precision)** | **100.00%** | **47.82% (MNN)** |
| **匹配召回率 (Recall)** | **100.00%** | **39.80% (Top-1)** |
| **F1 综合评分** | **100.00%** | **43.37%** |
| **垃圾桶识别率 (Dustbin Acc)**| **100.00%** (精准剔除出画目标) | 软阈值截断 |
| **单样本推理延迟 (Latency)** | **~2.2 ms** (实时高帧率) | **~15.0 ms** |
| **参数量** | **0.25 M** | **1.85 M** |

---

## 许可证 (License)

本项目采用 MIT 协议开源。
