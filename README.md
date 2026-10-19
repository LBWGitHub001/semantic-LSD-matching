# semantic-LSD-match

语义线段匹配相关工程。当前仓库已接入基于 **PyTorch / torchvision** 的 **DeepLabv3** 语义分割推理模块，用于从图像得到像素级语义标签图。

远程仓库：<https://github.com/LBWGitHub001/semantic-LSD-matching.git>

## 功能概览

- 使用 torchvision 预训练 DeepLabv3（COCO + PASCAL VOC 21 类）
- 支持 backbone：`resnet50` / `resnet101` / `mobilenet_v3_large`
- 命令行推理：输出彩色 mask 与原图叠加图
- 一键 demo：自动下载样例图并跑通流程
- **特征匹配 demo（ORB）**：两图关键点匹配并可视化连线
- **XFeat demo（CVPR 2024）**：轻量学习型特征匹配（官方代码在 `third_party/`）

## 环境依赖

- Python 3.9+
- PyTorch、torchvision、Pillow、numpy、opencv-python

```bash
pip install -r requirements.txt
```

建议安装与本机 CUDA 匹配的 PyTorch；无 GPU 时加 `--cpu` 即可。  
特征匹配 demo 只需 `opencv-python`（可自行安装，本仓库不改动你的 conda 环境）。

## 快速开始

```bash
# DeepLabv3 语义分割 demo
python demo/demo_deeplab.py

# 特征匹配 demo（ORB，无参数：下载样例图并做旋转缩放，再匹配）
python demo/demo_feature_match.py
python demo/demo_feature_match.py path/to/img1.jpg path/to/img2.jpg

# XFeat（需先 clone 官方仓库到 third_party/accelerated_features）
python demo/demo_xfeat.py
python demo/demo_xfeat.py path/to/img1.jpg path/to/img2.jpg
# 半稠密匹配
python demo/demo_xfeat.py --star

# LineFeat 直线特征与跨图匹配 demo
python demo/demo_linefeat.py --pair-idx 0

# 探索实验：直线邻域仿射对齐与条带匹配实验
python demo/demo_neighborhood_matching.py

# 训练 Line-Transformer 直线匹配网络
python train_linefeat.py --epochs 10 --lr 1e-4 --save-dir checkpoints/linefeat

# 工具脚本：Structured3D 数据集多线程断点下载、解压与读取检查
python tools/download_structured3d.py
python tools/unzip_structured3d.py Structured3D_perspective_full_00.zip
python tools/read_structured3d.py            # 读取视点并导出 5 图拼贴可视化
python tools/read_structured3d.py --pair     # 检查同房间双视点相对位姿与基线
python tools/read_structured3d.py --stats-only
python tools/generate_line_gt.py             # 提取双视点直线匹配真值并导出配对图与 JSON
python tools/build_line_gt_dataset.py        # 多进程对全量数据集预计算直线匹配真值库 (15秒)

# 指定图片做语义分割
python -m deeplab.infer path/to/image.jpg -o outputs
python -m deeplab.infer path/to/image.jpg -b resnet101 --cpu
```

输出默认写到 `outputs/`：

| 文件 | 说明 |
|------|------|
| `*_mask.png` | VOC 调色板着色的语义图 |
| `*_overlay.png` | 原图与语义图半透明叠加 |
| `feature_match/matches.jpg` | ORB 特征匹配连线可视化 |
| `xfeat/matches.jpg` | XFeat 匹配连线可视化 |
| `linefeat/*_ground_truth.jpg` | 3D 线框投影遮挡剔除真值匹配图 |
| `linefeat/*_lsd_semantic_matches.jpg` | LSD 直线语义门控匹配图 |
| `linefeat/neighborhood_alignment_experiment.jpg` | 直线局部邻域对齐与残差分析图 |

语义分割 demo 会打印各类像素数；特征匹配 demo 会打印关键点与匹配数量。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── train_linefeat.py           # Line-Transformer 端到端直线匹配训练脚本
├── LineFeat/                   # 直线特征核心算法包
│   ├── __init__.py             # 统一导出接口
│   ├── detector.py             # 2D LSD 直线检测与过滤
│   ├── descriptor.py           # 深度特征图沿线网格采样与汇聚
│   ├── matcher.py              # 直线匹配器 (MNN, Lowe 比值检验, 语义门控)
│   ├── neighborhood.py         # 局部邻域仿射变换、条带提取与对齐度量
│   ├── model.py                # LineMatcherNet (CNN + 几何编码 + Transformer + Sinkhorn)
│   ├── loss.py                 # LineMatchingLoss (NLL 损失与评价指标)
│   ├── dataset.py              # LineFeatPairDataset (PyTorch 训练/验证数据集)
│   └── ground_truth.py         # 3D 线框投影与视锥裁剪、深度遮挡剔除引擎
├── dataset/                    # 数据集抽象层
│   └── structured3d_pair.py    # Structured3D 基础视图对读取
├── demo/                       # 演示与实验入口
│   ├── demo_deeplab.py         # DeepLabv3 语义分割样例
│   ├── demo_feature_match.py   # ORB 特征匹配样例
│   ├── demo_xfeat.py           # XFeat 匹配样例
│   ├── demo_linefeat.py        # LineFeat 端到端直线匹配验证
│   └── demo_neighborhood_matching.py # 直线邻域变换对齐实验
├── tools/                      # 数据集下载、解压与实用脚本
│   ├── download_structured3d.py# Structured3D 断点续传多线程下载
│   ├── unzip_structured3d.py   # Structured3D 安全解压脚本
│   ├── read_structured3d.py    # Structured3D 数据集读取、相机模型与线段投影
│   ├── generate_line_gt.py     # 单像对 3D 几何投影与真值匹配可视化导出
│   └── build_line_gt_dataset.py# 多进程全量预计算真值库构建工具
├── docs/                       # 参考文献与研读
│   └── papers/                 # XFeat (CVPR 2024), PRS-Net (TVCG 2021)
├── deeplab/                    # 语义分割核心实现
├── feature_match/              # 特征匹配核心实现（ORB）
└── third_party/
    ├── README.md
    └── accelerated_features/   # 官方 XFeat（本地 clone，不入库）
```

## 代码中调用

```python
from deeplab import build_deeplabv3
from deeplab.utils import load_image, predict_mask, colorize_mask

model, weights = build_deeplabv3(backbone="resnet101", pretrained=True)
image = load_image("your.jpg")
mask = predict_mask(model, image, weights)
colorize_mask(mask).save("mask.png")
```

## License 与引用

DeepLabv3 论文：

> Chen et al. *Rethinking Atrous Convolution for Semantic Image Segmentation*. arXiv:1706.05587, 2017.

模型实现与权重来自 [torchvision](https://pytorch.org/vision/stable/models/deeplabv3.html)。
