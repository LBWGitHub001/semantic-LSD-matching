# semantic-LSD-match

语义线段匹配相关工程。当前仓库已接入基于 **PyTorch / torchvision** 的 **DeepLabv3** 语义分割推理模块，用于从图像得到像素级语义标签图。

远程仓库：<https://github.com/LBWGitHub001/semantic-LSD-matching.git>

## 功能概览

- 使用 torchvision 预训练 DeepLabv3（COCO + PASCAL VOC 21 类）
- 支持 backbone：`resnet50` / `resnet101` / `mobilenet_v3_large`
- 命令行推理：输出彩色 mask 与原图叠加图
- 一键 demo：自动下载样例图并跑通流程

## 环境依赖

- Python 3.9+
- PyTorch、torchvision、Pillow、numpy

```bash
pip install -r requirements.txt
```

建议安装与本机 CUDA 匹配的 PyTorch；无 GPU 时加 `--cpu` 即可。

## 快速开始

```bash
# 样例图推理（自动下载 dog.jpg）
python -m deeplab.demo

# 指定图片
python -m deeplab.infer path/to/image.jpg -o outputs

# 选用 backbone / 强制 CPU
python -m deeplab.infer path/to/image.jpg -b resnet101 --cpu
```

输出默认写到 `outputs/`：

| 文件 | 说明 |
|------|------|
| `*_mask.png` | VOC 调色板着色的语义图 |
| `*_overlay.png` | 原图与语义图半透明叠加 |

终端会打印各出现类别的像素数。

## 预训练权重

首次加载时由 torchvision **自动下载**到：

`~/.cache/torch/hub/checkpoints/`

| Backbone | 权重 URL |
|----------|----------|
| ResNet-50 | https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth |
| ResNet-101 | https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth |

也可手动下载上述文件后，用 `torch.load` 自行加载（当前默认走自动下载）。

**类别**：`__background__` + PASCAL VOC 20 类（person、car、dog 等）。  
若任务需要道路/建筑等标签，需换 Cityscapes 等权重或自行微调。

## 原理简述

DeepLabv3 对每个像素做分类：

1. **Backbone**：用空洞卷积（atrous）提取较密的特征图，扩大感受野且不过度下采样。
2. **ASPP**：多空洞率并行卷积 + 图像级池化，融合多尺度上下文。
3. **输出头**：得到 `[C, H, W]` logits，`argmax` 得到类别 id，再最近邻还原到原图尺寸。

本仓库不训练模型，只封装**推理**（加载、预处理、后处理与可视化）。核心流程见 `deeplab/utils.py` 中的 `predict_mask`。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── deeplab/
    ├── __init__.py
    ├── model.py      # 构建 DeepLabv3、加载预训练权重
    ├── voc_labels.py # VOC 21 类名与调色板
    ├── utils.py      # 预处理、预测、上色、叠加
    ├── infer.py      # 命令行入口
    └── demo.py       # 样例图 demo
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
