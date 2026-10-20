# LineFeat: 跨视角 2D 直线特征匹配系统

基于深度学习与最优传输（Optimal Transport）的跨视角已知直线特征匹配算法库。  
核心任务：**在直线已由上游完成检测的前提下，直接输入图像与直线坐标，实现高精度的端到端跨视角直线配对。**

---

## 一、系统架构与核心特性

本项目提供两套互补的前沿直线匹配网络模型：

1. **`LineMatcherNet` (Line-Transformer + Sinkhorn)**:
   - 提取 ResNet 50 稠密特征，沿线段均匀插值采样纹理特征；
   - 结合中心位置、长度与 $(\sin 2\theta, \cos 2\theta)$ 连续角度几何编码；
   - 4 层交替 Self-Attention（图内拓扑）与 Cross-Attention（跨视角互查）；
   - 对数域 Sinkhorn 求解最优传输分配矩阵，并配有 Dustbin 垃圾桶丢弃未配对/遮挡线段。

2. **`MaskLineMatcherNet` (连续几何场 + 掩膜引导 + 255x255 规范矩阵)**:
   - 输入 6 通道张量：`RGB(3) + 归一化欧氏距离场 EDT(1) + 吸引朝向场 AFM(2)`，彻底避免离散数字标量偏置；
   - 掩膜空间池化（Mask-Pooling）：利用线段掩膜在特征图上聚合各线段专属 Token，严格确定性绑定 ID；
   - 规范输出：直接映射至固定 $(255, 255)$ 匹配概率矩阵，具备严格的数学**置换等变性（Permutation Equivariance）**。

---

## 二、精简目录结构

```text
semantic-LSD-matching/
├── LineFeat/                       # 【核心算法包】纯直线匹配模型库（全部自包含）
│   ├── __init__.py                 # 模块统一对外导出
│   ├── model.py                    # LineMatcherNet 网络架构
│   ├── mask_line_matcher.py        # MaskLineMatcherNet (6通道几何场 + 255x255矩阵)
│   ├── geometric_field.py          # 高性能连续 EDT 与 AFM 几何场生成器 (OpenCV C++)
│   ├── pipeline.py                 # 高层推理接口 LineMatchPipeline (纯匹配，零检测)
│   ├── dataset.py                  # LineFeatPairDataset 数据集加载器
│   ├── loss.py                     # LineMatchingLoss 与 MaskedMatrixLoss
│   ├── matcher.py                  # 匹配后处理 (MNN, 阈值过滤) 与彩色连线可视化
│   ├── ground_truth.py             # 3D线框投影与深度遮挡剔除真值引擎
│   ├── read_structured3d.py        # Structured3D 相机位姿与数据解析器
│   └── README.md                   # 算法包细节文档
│
├── scripts/                        # 【统一工作台】所有的训练、测试、调用脚本全在此
│   ├── train.py                    # 模型训练脚本 (支持缓存加速与AMP混合精度)
│   ├── test_matching.py            # 在精选测试集上做纯匹配评估 (Mode A & Mode B)
│   ├── demo_mask_network.py        # 演示 6通道输入与 255x255 规范矩阵输出
│   ├── example_call.py             # 极简 5 行 Python 调用纯匹配示例
│   ├── run_unit_tests.py           # 严谨数学单元测试 (置换等变性与可微性验证)
│   ├── cache_line_features.py      # 32核多线程特征预提取加速器
│   ├── build_line_gt_dataset.py    # 3D Wireframe 真值库预计算工具
│   ├── download_structured3d.py    # Structured3D 多线程断点下载工具
│   └── unzip_structured3d.py       # 自动解压工具
│
├── data/                           # 数据集
│   ├── test_samples/               # 精选 4 组测试集 (包含每组的 lines0.json, lines1.json, gt_matches.json)
│   └── line_gt_dataset/            # 全量真值匹配索引与标注
│
├── checkpoints/                    # 权重目录
│   └── linefeat/best_model.pth     # 训练最优权重
│
├── outputs/                        # 输出可视化成果
│   ├── pure_matching/              # GT vs 预测连线对比图 (绿色正确，红色误配)
│   └── field_network/              # 连续几何场与 255x255 概率热力图
│
├── requirements.txt
└── README.md
```

---

## 三、快速使用指南

### 1. 运行纯匹配测试（零检测器，纯测匹配网络）
```bash
# 在精选测试集上运行纯匹配评估（对比真值输出准确率与召回率）
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python scripts/test_matching.py --mode A

# 测试含干扰线的 Mode B
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python scripts/test_matching.py --mode B
```

### 2. 运行新型网络演示（6 通道输入 -> 255x255 规范矩阵）
```bash
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python scripts/demo_mask_network.py
```

### 3. 运行模型数学单元测试（置换等变性验证）
```bash
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python scripts/run_unit_tests.py
```

### 4. 重新训练模型
```bash
# 极速缓存训练模式（每轮仅需 ~35 秒）
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python scripts/train.py --use-cache --epochs 15 --amp
```

---

## 四、Python 代码中直接调用纯匹配（极简 5 行）

已提供范例 [`scripts/example_call.py`](scripts/example_call.py)：

```python
import json, numpy as np
from LineFeat import LineMatchPipeline

# 1. 外部读入已检测好的直线端点坐标 (N0, 4) 与 (N1, 4)
lines0 = np.array(json.load(open("data/test_samples/pair1_small_rotation/mode_a_pure_permuted/lines0.json")), dtype=np.float32)
lines1 = np.array(json.load(open("data/test_samples/pair1_small_rotation/mode_a_pure_permuted/lines1.json")), dtype=np.float32)

# 2. 初始化纯匹配流水线
pipeline = LineMatchPipeline("checkpoints/linefeat/best_model.pth")

# 3. 执行纯匹配推断（内部零检测器，直接匹配给定坐标）
result = pipeline.match("image0.png", "image1.png", lines0=lines0, lines1=lines1, threshold=0.70)

# 4. 获取配对对索引与得分
for rank, (i, j, score) in enumerate(result.matches):
    print(f"Match #{rank+1}: Line0[{i}] <--> Line1[{j}] | Score: {score:.3f}")

# 5. 保存结果连线图
result.save_visualization("outputs/pure_matching/result.jpg")
```
