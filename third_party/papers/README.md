# 核心学术论文库与文献索引 (Core Papers & Literature)

本项目融合了计算机视觉与深度学习领域在**几何特征匹配**、**亚像素几何引导**与**大规模基础语义模型**方向的 5 项核心奠基性工作。

---

## 1. 论文清单与文献脉络

| 论文名称 | 发表会议 | 官方代码 / 项目主页 | arXiv 编号 | 本项目角色与定位 |
| :--- | :--- | :--- | :--- | :--- |
| **GlueStick: A Robust Line and Keypoint Matcher** | **CVPR 2023** | [cvg/GlueStick](https://github.com/cvg/GlueStick) | [2304.02008](https://arxiv.org/abs/2304.02008) | 匹配网络主干架构 (GNN + 最优传输) |
| **DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients** | **CVPR 2023** | [cvg/DeepLSD](https://github.com/cvg/DeepLSD) | [2212.07766](https://arxiv.org/abs/2212.07766) | 亚像素几何线段检测与连续距离场 (DF) 引导 |
| **Segment Anything (SAM)** | **ICCV 2023** | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | [2304.02643](https://arxiv.org/abs/2304.02643) | 旗舰版 ViT-H 语义先验与实例轮廓分析 |
| **SuperPoint: Self-Supervised Interest Point Detection and Description** | **CVPRW 2018** | [SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork) | [1712.07629](https://arxiv.org/abs/1712.07629) | 密集/稀疏关键点提取与 256 维视觉描述子 |
| **ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes** | **CVPR 2017** | [ScanNet](http://www.scan-net.org/) | [1702.04405](https://arxiv.org/abs/1702.04405) | 评测数据集基准 (ScanNet-1500 室内校准配对) |

---

## 2. 核心论文解析与技术关联

### 2.1 GlueStick (CVPR 2023)
* **论文标题**：*GlueStick: A Robust Line and Keypoint Matcher*
* **作者**：Rémi Pautrat, Iago Suárez, Yifan Yu, Marc Pollefeys, Viktor Larsson (ETH Zürich)
* **核心创新**：
  将独立的特征点与几何线段构建为统一的**线框图 (Wireframe Graph)**，通过多层自注意力 (Self-Attention) 与跨视角交叉注意力 (Cross-Attention) 消息传递，并在端点与线段之间交替传播更新，最后通过 Log-Double-Softmax (最优传输) 联合输出点和线匹配。
* **本项目复现与改进**：
  在 `models/gluestick_wrapper.py` 中实现了完整的训练损失计算体系；同时研发了 `SemanticGlueStick`，在几何线段编码处引入 SAM 256 维语义特征残差融合。

### 2.2 DeepLSD (CVPR 2023)
* **论文标题**：*DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients*
* **作者**：Rémi Pautrat, Juan-Ting Lin, Viktor Larsson, Martin R. Oswald, Marc Pollefeys (ETH Zürich)
* **核心创新**：
  引入深度神经网络预测亚像素几何距离场 (Distance Field, DF) 与最近线角度场 (Angle Field, AF)，结合经典的 LSD 算法实现高精度线段检测，并可基于 DF 对线段端点进行最小二乘非线性优化。
* **本项目应用**：
  在 `extractors/deeplsd_extractor.py` 中封装推理模型，输出连续距离场作为图结构强几何先验。

### 2.3 Segment Anything (SAM, ICCV 2023)
* **论文标题**：*Segment Anything*
* **作者**：Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, et al. (Meta AI Research)
* **核心创新**：
  构建了首个具备强大零样本泛化能力的大型图像分割基础模型 (ViT-H 达 6.36 亿参数)，能输出高维稠密语义特征图并精细勾勒出物体物理边界轮廓。
* **本项目应用**：
  在 `extractors/sam_extractor.py` 中基于 PyTorch 2 SDPA 注意力机制优化显存，沿线段高密插值均值池化出 256 维语义描述子，并在掩膜上计算是否为物体轮廓边界。

---

## 3. 一键下载论文原文

在项目根目录下运行如下脚本，即可自动拉取 5 篇原版官方 PDF 文件至本目录：

```bash
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python third_party/papers/download_papers.py
```
