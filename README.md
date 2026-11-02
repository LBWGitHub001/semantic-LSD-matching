# Multi-Modal Semantic GlueStick (语义与几何增强线点跨视角匹配系统)

本项目基于 **Meta SAM ViT-H 旗舰基础大模型** 与 **DeepLSD 亚像素几何距离场**，对经典图神经网络线点匹配模型 **GlueStick (CVPR 2023)** 进行了全流程的数据集构建、网络重现与深层语义特征融合改进。

---

## 1. 核心架构与 5 大模块组织

整个代码库经过系统化重构，解耦为 5 大高内聚核心模块：

```text
new-LSD-matching/
├── datasets/                           # 【1. 数据集以及生成】
│   ├── build_semantic_gluestick_dataset.py  # 数据集批量构建主程序 (融合 DeepLSD + SAM + SP)
│   ├── semantic_gluestick_dataset.py        # PyTorch Dataset 样本加载器与变长动态批对齐
│   └── README.md                            # 数据集规范说明
│
├── models/                             # 【2. GlueStick 复现以及改进 - 模型网络】
│   └── gluestick_wrapper.py            # GlueStick 官方模型封装、SAM 语义适配器、NLL 损失与评价指标
│
├── extractors/                         # 【2. GlueStick 复现以及改进 - 特征引擎】
│   ├── sam_extractor.py                # SAM ViT-H 密集语义特征提取与沿线双线性池化 (SDPA 显存优化)
│   ├── deeplsd_extractor.py            # DeepLSD 几何线段与连续距离场 (DF) 提取器
│   └── wireframe_builder.py            # SuperPoint 提取与线框图 DBSCAN 空间聚类合流
│
├── train_gluestick.py                  # 【2. 训练主入口】支持原版微调与 SAM 语义增强训练
├── eval_gluestick.py                   # 【2. 评测主入口】计算点/线 Precision, Recall, F1 并导出匹配诊断图
│
├── third_party/                        # 【3. 第三方源代码和论文】
│   ├── DeepLSD/                        # DeepLSD 官方开源推理仓库 (CVPR 2023)
│   ├── GlueStick/                      # GlueStick 官方开源模型仓库 (CVPR 2023)
│   └── papers/                         # 5 篇核心学术论文索引 (GlueStick, DeepLSD, SAM, SP, ScanNet)
│       ├── README.md                   # 论文创新要点、BibTeX 与架构关系解读
│       └── download_papers.py          # 自动下载原版 PDF 论文脚本
│
├── checkpoints/                        # 【4. 检查点 Checkout】
│   ├── gluestick_semantic_run/         # 语义增强训练成果权重与指标 (best_model.pth, train_history.json)
│   └── gluestick_baseline_run/         # 基线微调权重
│
├── weights/                            # 【4. 预训练基础模型权重】
│   ├── sam/sam_vit_h_4b8939.pth        # SAM ViT-H 旗舰超大权重 (2.56 GB)
│   ├── deeplsd/deeplsd_wireframe.tar   # DeepLSD 几何权重
│   ├── gluestick/checkpoint_GlueStick_MD.tar
│   ├── superpoint/superpoint_v1.pth
│   └── scripts/download_models.py      # 模型下载脚本
│
├── visualizers/                        # 【5. 可视化】
│   ├── visualize_multimodal_combo.py   # 四合一综合看板渲染器 (原图 + DF + SAM 掩膜 + LSD 融合)
│   └── visualize_semantic_dataset.py   # 多场景对比全景合辑与 3 行综合诊断画廊
│
├── outputs/                            # 【5. 输出结果】
│   ├── multimodal_combo/               # 四合一 2x2 宫格图与 1x4 横条大图
│   ├── gluestick_eval/                 # 跨视角预测匹配 vs 真值匹配对比图
│   └── visualizations/                 # 样本诊断图集
│
└── data/                               # 物理数据集实际存储目录 (ScanNet-1500 及预生成 npz)
```

---

## 2. 运行环境配置

推荐使用专用 Conda 环境：
```bash
# 激活环境
conda activate LSD-matching
# Python 路径: /home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python
```

---

## 3. 常用操作与运行命令一览

### 3.1 数据集构建与加载验证
```bash
# 批量构建 ScanNet 语义增强 GlueStick 数据集
python datasets/build_semantic_gluestick_dataset.py \
    --pairs-file data/scannet/scannet1500/pairs_calibrated.txt \
    --data-dir data/scannet/scannet1500 \
    --out-dir data/scannet_semantic_gluestick

# 测试 Dataset 与 DataLoader 批次装配
python datasets/semantic_gluestick_dataset.py \
    --data-dir data/scannet_semantic_gluestick \
    --batch-size 2
```

### 3.2 GlueStick 模型训练
```bash
# 模式 A: 语义增强训练 (融入 SAM ViT-H 256 维语义特征，推荐)
python train_gluestick.py \
    --data-dir data/scannet_semantic_gluestick \
    --use-semantics \
    --epochs 30 \
    --lr 1e-5 \
    --batch-size 1 \
    --save-dir checkpoints/gluestick_semantic_run

# 模式 B: 原版基线微调 (纯几何线框与特征点)
python train_gluestick.py \
    --data-dir data/scannet_semantic_gluestick \
    --epochs 30 \
    --lr 1e-5 \
    --batch-size 1 \
    --save-dir checkpoints/gluestick_baseline_run
```

### 3.3 模型量化评测与匹配诊断
```bash
# 在验证集上执行全量评测并输出匹配连线对比图
python eval_gluestick.py \
    --checkpoint checkpoints/gluestick_semantic_run/best_model.pth \
    --use-semantics \
    --split val \
    --out-dir outputs/gluestick_eval
```

### 3.4 多模态四合一综合可视化
```bash
# 渲染单张大图包含: 原图 + DeepLSD 几何距离场 + SAM 语义掩膜叠加 + LSD全要素叠加
python visualizers/visualize_multimodal_combo.py \
    --dataset-dir data/scannet_semantic_gluestick \
    --sample-name pair_00000 \
    --out-dir outputs/multimodal_combo
```

### 3.5 核心论文文献下载
```bash
# 一键下载项目引用的 5 篇核心学术论文 PDF
python third_party/papers/download_papers.py
```
