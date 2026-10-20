# LineFeat: 跨视角 2D 直线特征匹配系统

基于深度学习与最优传输（Optimal Transport）的跨视角已知直线特征匹配算法库。  
核心任务：**在直线已由上游完成检测的前提下，直接输入图像与直线坐标，实现高精度的端到端跨视角直线配对。**

---

## 目录

- [一、核心架构与设计思想](#一核心架构与设计思想)
- [二、完整文件树与各模块详析](#二完整文件树与各模块详析)
  - [1. LineFeat 核心算法包 (`LineFeat/`)](#1-linefeat-核心算法包-linefeat)
  - [2. 统一工作台脚本 (`scripts/`)](#2-统一工作台脚本-scripts)
  - [3. 数据与权重目录 (`data/`, `checkpoints/`, `outputs/`)](#3-数据与权重目录)
- [三、核心模型详解](#三核心模型详解)
  - [1. MaskLineFeatMatchNet (默认主打模型：6 通道几何场 + 255x255 规范矩阵)](#1-masklinefeatmatchnet-默认主打模型6-通道几何场--255x255-规范矩阵)
  - [2. LineMatcherNet (基线模型：纯外观插值 + Line-Transformer)](#2-linematchernet-基线模型纯外观插值--line-transformer)
  - [3. 深入技术文档 (`docs/MaskLineFeatMatchNet.md`)](#3-深入技术文档)
- [四、典型使用场景与操作指南](#四典型使用场景与操作指南)
  - [场景 1：Python 极简调用纯匹配 (5 行代码，直出 255x255 矩阵)](#场景-1python-极简调用纯匹配-5-行代码直出-255x255-矩阵)
  - [场景 2：精选测试集定量基准评测 (Mode A 与 Mode B)](#场景-2精选测试集定量基准评测-mode-a-与-mode-b)
  - [场景 3：新型 6 通道几何场网络演示 (255x255 矩阵)](#场景-3新型-6-通道几何场网络演示-255x255-矩阵)
  - [场景 4：数学单元测试 (置换等变性与可微性)](#场景-4数学单元测试-置换等变性与可微性)
  - [场景 5：从零训练模型 (32核加速 + 缓存训练)](#场景-5从零训练模型-32核加速--缓存训练)
  - [场景 6：本地与云端 GPU 双向免密同步](#场景-6本地与云端-gpu-双向免密同步)

---

## 一、核心架构与设计思想

在三维视觉与机器人领域，线段匹配比点匹配更具挑战性：线段易受遮挡导致端点截断（Stick-Breaking）、缺乏显著内部纹理、且存在大量平行重复结构。

本项目坚守以下设计原则：
1. **纯粹匹配，零检测器依赖**：不将算力浪费在不可控的在线检测上，输入直接为预先检测出的线段端点坐标；
2. **多模态表征融合**：沿线段均匀插值采样稠密外观纹理，同时引入端点无序的 $(\sin 2\theta, \cos 2\theta)$ 连续几何编码；
3. **全局拓扑交互**：利用 Transformer 自注意力捕获房间结构，交叉注意力实现跨视角候选线对比；
4. **最优传输排他性**：通过带垃圾桶（Dustbin）的 Sinkhorn 算法求解全局唯一配对，自然过滤未匹配与单侧遮挡线段；
5. **前沿连续场扩展**：探索采用欧氏距离场（EDT）与吸引朝向场（AFM）消除离散标量偏置，直出规范的 $255 \times 255$ 置换等变矩阵。

---

## 二、完整文件树与各模块详析

### 1. LineFeat 核心算法包 (`LineFeat/`)
> 详细 API 文档与类接口解析请阅读 [📘 LineFeat/README.md](LineFeat/README.md)。

所有核心算法与数据解析代码全部聚合在此，自包含、无外部非标准依赖。

#### ① `LineFeat/__init__.py`
- **功能**：算法包的统一对外门面，导出全部关键类与工具函数，定义版本号。
- **导出符号**：`MaskLineFeatMatchNet`, `MaskLineMatcherNet`, `LineMatcherNet`, `LineMatchPipeline`, `LineFeatPairDataset`, `compute_geometry_field_np`, `create_6ch_input_tensor`, `LineMatchingLoss`, `MaskedMatrixLoss` 等。

#### ② `LineFeat/mask_line_matcher.py`（系统默认主打模型）
- **功能**：新型 `MaskLineFeatMatchNet`（别名 `MaskLineMatcherNet`）实现，支持 6 通道输入并直出 $255 \times 255$ 规范矩阵。
- **核心逻辑**：
  - 首层卷积扩充为 6 通道，自适应迁移 ImageNet 权重，接收 `RGB (3) + EDT 距离场 (1) + AFM 方向场 (2)`；
  - 空间掩膜引导池化：以几何场与线段掩膜为索引提取各线段专属 Token，严格确定性绑定 ID；
  - 4 层交替注意力 Transformer 图推理与带 Dustbin 的对数 Sinkhorn 最优传输；
  - 投影至规范形状 $(B, 255, 255)$，严格保持数学置换等变性（Permutation Equivariance）。
- **详尽技术文档**：参见专门编写的 [docs/MaskLineFeatMatchNet.md](docs/MaskLineFeatMatchNet.md)。

#### ③ `LineFeat/model.py`
- **功能**：轻量基线 `LineMatcherNet` 模型的完整实现。
- **核心组件**：
  - `LineVisualSampler`：沿线段两端点均匀采样 $K=7$ 个点，通过 `grid_sample`（双线性插值）从 ResNet-50 特征图提取纹理，平均池化为 128 维视觉向量；
  - `LineGeometryEncoder`：将中心点 $(c_x, c_y)$、长度 $l$ 以及连续方向 $(\sin 2\theta, \cos 2\theta)$ 编码为 128 维几何位置特征（端点翻转不影响编码）；
  - `MultiHeadAttentionBlock`：包含自注意力、交叉注意力与前馈神经网络的 Transformer 层；
  - `log_optimal_transport`：带有对数域数值稳定化与可学习垃圾桶的 Sinkhorn 算法求解器。

#### ④ `LineFeat/geometric_field.py`
- **功能**：高性能连续几何场生成器。
- **核心函数**：
  - `compute_geometry_field_np(lines, height, width, max_dist=60.0)`：基于 OpenCV C++ 底层，极速计算出归一化欧氏距离场（EDT）与二倍角平滑朝向场（AFM）；
  - `create_6ch_input_tensor(rgb_tensor, lines)`：自动将 3 通道 RGB 张量与几何场拼接为 6 通道输入张量。

#### ⑤ `LineFeat/pipeline.py`
- **功能**：高层推理流水线，为用户提供极其简洁的调用接口（默认采用 `MaskLineFeatMatchNet`）。
- **核心类**：
  - `LineMatchPipeline(weights, device, similarity_threshold, model_type="mask")`：一行代码初始化模型，默认使用 `MaskLineFeatMatchNet`；
  - `match(image0, image1, lines0, lines1)`：纯推断匹配，完全不运行任何检测器；
  - `LineMatchResult`：封装匹配对元组、匹配线段坐标、置信度得分，以及 $(255, 255)$ 规范概率矩阵 `result.matrix_255`，并提供 `save_visualization()` 连线导出方法。

#### ⑥ `LineFeat/dataset.py`
- **功能**：PyTorch 标准数据集加载器 `LineFeatPairDataset`。
- **核心功能**：从预计算的真值索引中读取同房间双视点图像、相机内外参（$K, R, t$）、旋转角度、基线距离以及三维物理线段投影产生的 1-to-1 真实线段匹配对。支持 `train` / `val` / `test` 划分。

#### ⑦ `LineFeat/loss.py`
- **功能**：模型训练损失函数库。
- **核心类**：
  - `LineMatchingLoss`：监督真值匹配在 Sinkhorn 分配矩阵中的负对数似然 (NLL)；
  - `MaskedMatrixLoss`：专门针对 $255 \times 255$ 规范矩阵的损失函数，结合 NLL 与只在有效 $N_0 \times N_1$ 子区域计算的带掩膜二值交叉熵（Masked BCE），防止未定义空白区稀释梯度。

#### ⑧ `LineFeat/matcher.py`
- **功能**：匹配后处理与渲染绘制工具。
- **核心函数**：
  - `LineMatcher.draw_matches(...)`：将两张图像横向拼接，给每对匹配线赋予鲜艳独立的颜色，绘制端点并使用彩色连线连接直线中心；
  - `get_distinct_colors(n)`：在 HSV 空间均匀采样生成高对比度易辨认的颜色列表。

#### ⑨ `LineFeat/ground_truth.py` 与 `LineFeat/read_structured3d.py`
- **功能**：数据集解析与三维物理真值生成引擎。
- **核心功能**：解析 Structured3D 数据集的场景层级、三维线框网格（annotation_3d）；利用相机矩阵做近平面视锥裁剪与基于 Z-Buffer 深度图的遮挡剔除，生成 100% 物理真实的双视角 2D 直线对应关系。

---

### 2. 统一工作台脚本 (`scripts/`)
> 完整脚本参数表与用法指南请阅读 [📘 scripts/README.md](scripts/README.md)。

所有的可执行脚本全部统一存放在 `scripts/` 目录下：

| 脚本文件名 | 功能分类 | 核心作用与使用方法 |
| :--- | :---: | :--- |
| [`scripts/example_call.py`](scripts/example_call.py) | **极简调用** | **最适合新手接入**。5 行 Python 代码演示如何传入图像与直线坐标进行纯匹配。 |
| [`scripts/test_matching.py`](scripts/test_matching.py) | **基准测试** | 在精选测试集上做纯匹配评测，对比真值输出 Precision/Recall 表，并渲染 GT vs 预测连线对比大图。 |
| [`scripts/demo_mask_network.py`](scripts/demo_mask_network.py) | **前沿演示** | 演示 6 通道几何场生成、网络推断及 $255 \times 255$ 匹配概率矩阵热力图输出。 |
| [`scripts/run_unit_tests.py`](scripts/run_unit_tests.py) | **单元测试** | 运行数学置换等变性测试（误差验证 $< 3 \times 10^{-8}$）、输出规范形状测试与反向传播可微性测试。 |
| [`scripts/train.py`](scripts/train.py) | **模型训练** | 端到端训练脚本，支持 `--use-cache` 缓存极速训练（~35s/epoch）与 `--amp` FP16 自动混合精度。 |
| [`scripts/cache_line_features.py`](scripts/cache_line_features.py) | **性能加速** | 32 核异步生产者-消费者架构，训练前将全量数据集视觉 Token 预提取并压缩至 RAM，提速 50 倍。 |
| [`scripts/build_line_gt_dataset.py`](scripts/build_line_gt_dataset.py) | **数据构建** | 多进程批处理脚本，扫描原始 Structured3D 场景并预计算全部相机的 3D-2D 线段投影真值库。 |
| [`scripts/sync_push.sh`](scripts/sync_push.sh) | **双机同步** | 一键将本地代码与修改增量推送到远程 GPU 服务器（排除 19G 大数据集，秒级完成）。 |
| [`scripts/sync_pull.sh`](scripts/sync_pull.sh) | **双机同步** | 一键将远程服务器跑出的最新模型权重 (`checkpoints/`) 与实验结果 (`outputs/`) 拉回本地。 |
| [`scripts/download_structured3d.py`](scripts/download_structured3d.py) | **数据工具** | 官方源 Structured3D 数据集多线程断点续传下载工具。 |
| [`scripts/unzip_structured3d.py`](scripts/unzip_structured3d.py) | **数据工具** | 规范化解压脚本，自动剔除冗余路径层级。 |

---

### 3. 数据与权重目录

- [`data/test_samples/`](data/test_samples/)：精选了 4 组具有难度梯度的真实测试集，每组目录均已导出纯文本真值文件：
  - `image0.png`, `image1.png`：原始两视角视图；
  - `mode_a_pure_permuted/`：`lines0.json`, `lines1.json`（顺序已随机打乱）、`gt_matches.json`；
  - `mode_b_with_distractors/`：混入了单视角可见但无对应的干扰线，附带 `unmatched_lines0.json` 与 `unmatched_lines1.json` 用于检验垃圾桶拒识率。
- [`checkpoints/linefeat/best_model.pth`](checkpoints/linefeat/best_model.pth)：训练收敛的最优权重，准确率达 79%~85%。
- [`outputs/pure_matching/`](outputs/pure_matching/)：测试生成的连线对比大图（绿色表示正确配对，红色表示误配）。
- [`outputs/field_network/`](outputs/field_network/)：连续几何场与 $255 \times 255$ 匹配概率矩阵热力图。

---

## 三、核心模型详解

### 1. MaskLineFeatMatchNet (默认主打模型：6 通道几何场 + 255x255 规范矩阵)
> **技术白皮书**：详细数学推导、消融实验与架构细节请阅读 [📘 docs/MaskLineFeatMatchNet.md](docs/MaskLineFeatMatchNet.md)。

- **输入**：`input0`, `input1`（6 通道张量：RGB + 距离场 EDT + 朝向场 AFM；若输入 3 通道 RGB 则自动计算几何场）+ `lines0` $(N_0, 4)$, `lines1` $(N_1, 4)$
- **核心创新与优势**：
  - **连续场消除序数偏置**：规避离散标量数字编号的数值偏置（Ordinal Bias），提供平滑空间法向与朝向梯度；
  - **二倍角朝向编码**：使用 $(\sin 2\theta, \cos 2\theta)$ 消除线段端点颠倒翻转不确定性（Flip Invariance）；
  - **严格数学置换等变性**：端到端运算满足 $\Pi_0 P \Pi_1^T$ 置换等变律，误差 $< 3 \times 10^{-8}$；
  - **规范化矩阵输出**：通过带有 Dustbin 垃圾桶的对数 Sinkhorn 求解最优传输，并映射为固定 $(255, 255)$ 概率矩阵 `result.matrix_255`。
- **输出**：`matrix_255` 张量，形状恒为 $(B, 255, 255)$，同时解析出互为最近邻的高置信度匹配对。

### 2. LineMatcherNet (基线模型：纯外观插值 + Line-Transformer)
- **输入**：`image0`, `image1`（3 通道 RGB 张量）+ `lines0` $(N_0, 4)$, `lines1` $(N_1, 4)$
- **网络流**：
  1. ResNet-50 提取特征图；
  2. 沿各直线采样 7 点做插值池化 + 5 维几何编码，得到两组初始线向量 $(N_0, 128)$ 与 $(N_1, 128)$；
  3. 4 层 Transformer（图内 Self-Attention + 跨图 Cross-Attention）；
  4. 余弦内积计算亲和度，扩充 Dustbin 垃圾桶行列，执行 20 次 Sinkhorn 迭代；
  5. 互为最近邻（MNN）校验与阈值过滤。
- **输出**：匹配对索引列表 `[(idx0, idx1, score), ...]` 与配对线段物理坐标。

### 3. 模型全流程参数划分：哪些训练？哪些冻结？

系统遵循**“冻结底层感知 + 训练高层推理”**的设计哲学，各组件训练状态划分如下：

| 组件名称 | 对应模块 | 参数量 | 训练状态 | 为什么这样设计？ |
| :--- | :--- | :---: | :---: | :--- |
| **视觉骨干网络** | `backbone` (ResNet-50 浅层) | **1.44 M** | **❄️ 100% 冻结** | • 保留 ImageNet 1400万张图像的通用边缘先验；<br>• 防止在白墙上出现表征坍塌；<br>• 支持离线预提取缓存，**训练速度提速 50 倍**。 |
| **线段采样投影器** | `sampler.proj` | **0.08 M** | **🔥 参与训练** | 自适应将 ResNet 的 512 维特征压缩至 128 维。 |
| **几何位置编码器** | `geometry_encoder` | **0.01 M** | **🔥 参与训练** | 将坐标、长度与 $(\sin 2\theta, \cos 2\theta)$ 映射为高维位置嵌入。 |
| **跨视角 Transformer** | `layers` (4~8层) | **0.80 M** | **🔥 核心训练** | **大脑逻辑中枢**。学习单图内结构拓扑与跨图候选线比对。 |
| **双向归一化投影头** | `proj0`, `proj1` | **0.03 M** | **🔥 参与训练** | 调整最终特征流形，使之最适宜 L2 归一化内积计算。 |
| **最优传输分配器** | `log_optimal_transport` | **0** | **⚙️ 固定算法** | 无参数对数 Sinkhorn 算法，严格保证全局互斥排他性。 |

> 💡 **总计**：训练模式下仅需优化 **约 0.92 M** 轻量高阶参数，极其轻量、训练飞快且不易过拟合！

---

## 四、典型使用场景与操作指南

### 场景 1：Python 极简调用纯匹配 (5 行代码，直出 255x255 矩阵)

直接运行 [`scripts/example_call.py`](scripts/example_call.py)：
```bash
python scripts/example_call.py
```

代码核心示例（默认即使用 `MaskLineFeatMatchNet`）：
```python
import json, numpy as np
from LineFeat import LineMatchPipeline

# 1. 读入已检测的直线端点坐标 (N0, 4) 与 (N1, 4)
lines0 = np.array(json.load(open("data/test_samples/pair1_small_rotation/mode_a_pure_permuted/lines0.json")), dtype=np.float32)
lines1 = np.array(json.load(open("data/test_samples/pair1_small_rotation/mode_a_pure_permuted/lines1.json")), dtype=np.float32)

# 2. 初始化纯匹配流水线（默认加载 MaskLineFeatMatchNet 到 GPU）
pipeline = LineMatchPipeline("checkpoints/linefeat/best_model.pth")

# 3. 执行匹配（零检测器运行，纯匹配给定线段，自动构建 6 通道几何场）
result = pipeline.match("image0.png", "image1.png", lines0=lines0, lines1=lines1, threshold=0.70)

# 4. 获取规范 255x255 匹配概率矩阵
print("Canonical Matrix:", result.matrix_255.shape)  # (255, 255)

# 5. 打印互为最近邻匹配结果
for rank, (i, j, score) in enumerate(result.matches):
    print(f"Match #{rank+1}: Line0[{i}] <--> Line1[{j}] | 置信度: {score:.3f}")

# 6. 导出连线可视化图
result.save_visualization("outputs/pure_matching/result.jpg")
```

---

### 场景 2：精选测试集定量基准评测 (Mode A 与 Mode B)

运行测试脚本对比 Ground Truth 并统计 Precision / Recall（默认评估 `MaskLineFeatMatchNet`）：

```bash
# 模式 A: 纯真值线打乱测试（验证无空间顺序先验下的匹配能力）
python scripts/test_matching.py --mode A

# 模式 B: 混入单视角未匹配干扰线（测试 Sinkhorn 垃圾桶的拒识过滤能力）
python scripts/test_matching.py --mode B

# 测试全部模式，并可自由选择评估网络类型 (--model-type mask 或 linefeat)
python scripts/test_matching.py --model-type mask --pair pair2_medium_rotation --mode both
```
> 输出结果保存在 `outputs/pure_matching/` 中，上半幅为真实真值对应连线，下半幅为模型预测（**绿色连线**为正确配对，**红色连线**为误配）。

---

### 场景 3：新型 6 通道几何场网络演示 (255x255 矩阵)

运行演示脚本：
```bash
python scripts/demo_mask_network.py
```
终端会输出 $(255, 255)$ 概率矩阵中激活的子矩阵数值，同时在 `outputs/field_network/` 下生成：
- `view0_geometry_fields.jpg`：RGB 原图、EDT 距离场与 AFM 朝向场的三联全景图；
- `canonical_255x255_probability_matrix.png`：输出的 $255 \times 255$ 匹配概率矩阵热力图。

---

### 场景 4：数学单元测试 (置换等变性与可微性)

运行单元测试确保网络数学特性健全：
```bash
python scripts/run_unit_tests.py
```
> **通过标准**：
> 1. `[Permutation Test]` 互换线段输入顺序，输出矩阵对应行列发生精确置换，误差 $< 10^{-7}$；
> 2. `[Differentiability Test]` 损失反向传播时，梯度可正常反传至投影层与特征骨干；
> 3. `[Shapes Test]` 输出张量形状严格为 $(B, 255, 255)$。

---

### 场景 5：从零训练模型 (32核加速 + 缓存训练)

若需要在全量数据集上重新训练或微调：

#### 步骤 1：32 核异步预提取特征（仅需跑一次，耗时约 60 秒）
```bash
python scripts/cache_line_features.py --num-threads 32 --batch-size 8
```
> 将全量图像的线段视觉特征提取并压缩缓存至 `data/line_feature_cache/` 中。

#### 步骤 2：启动高速缓存训练（每轮仅需 ~35 秒）
```bash
# 默认训练 MaskLineFeatMatchNet (支持 6 通道几何场 + 255x255 矩阵输出)
python scripts/train.py --model-type mask --use-cache --epochs 15 --lr 1e-4 --amp

# 或指定训练经典轻量 LineMatcherNet
python scripts/train.py --model-type linefeat --use-cache --epochs 15 --lr 1e-4 --amp
```
> 自动使用余弦退火学习率调度与 AMP 自动混合精度，训练最优模型会自动存至 `checkpoints/linefeat/best_model.pth`。

---

### 场景 6：本地与云端 GPU 双向免密同步

配合配置好的 SSH 免密环境，在本地终端执行即可实现双向秒级同步：

```bash
# 1. 本地写完代码，一键推送到远程服务器（自动排除 19G 大数据集）：
./scripts/sync_push.sh

# 2. 服务器跑完实验，一键将最新的模型权重与生成图片拉回本地：
./scripts/sync_pull.sh
```
