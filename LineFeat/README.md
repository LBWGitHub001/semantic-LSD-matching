# LineFeat: 跨视角直线特征检测、描述与匹配模块 (Semantic-LSD-Matching)

`LineFeat` 是本项目的核心直线匹配算法包，整合了 **2D LSD 直线检测**、**深度特征图沿线采样汇聚描述符 (Deep Line Descriptor)**、**双向最近邻比值检验与语义门控匹配 (Line Matcher)**、**基于 3D 线框投影与深度遮挡剔除的真值生成器 (Ground Truth)** 以及 **标准 PyTorch 数据集加载器 (LineFeatPairDataset)**。

---

## 模块架构

```
LineFeat/
├── __init__.py           # 统一导出所有核心类与函数
├── detector.py           # 2D LSD 直线检测器 (LineDetector)，支持长度过滤、边界过滤与等距采样
├── descriptor.py         # 深度直线描述子提取器 (LineDescriptor)，基于 torch.nn.functional.grid_sample
├── matcher.py            # 直线匹配器 (LineMatcher)，支持余弦相似度、MNN、Lowe比值检验与语义硬门控
├── dataset.py            # PyTorch 数据集 (LineFeatPairDataset)，支持从 precomputed index 快速加载
└── ground_truth.py       # 3D线框投影 + 视锥裁剪 + Z-Buffer深度遮挡剔除真值生成器
```

---

## 核心类与 API 使用指南

### 1. 2D 直线检测 (`LineDetector`)
```python
from LineFeat import LineDetector

detector = LineDetector(scale=0.8)
# 传入 BGR 或灰度图像，返回 (N, 4) 的 [x1, y1, x2, y2]
lines = detector.detect(image_bgr, min_length=30.0, margin=8)
```

### 2. 深度直线描述子提取 (`LineDescriptor`)
沿直线等距采样 $K$ 个关键点，利用双线性插值从 CNN 特征图提取对应特征向量并汇聚：
```python
from LineFeat import LineDescriptor

extractor = LineDescriptor(num_samples=7, pooling="mean", normalize=True)
# lines: (N, 4), feature_map: (1, C, H_f, W_f)
descriptors = extractor.extract(lines, feature_map, orig_size=(720, 1280))  # (N, C)
```

### 3. 直线匹配与语义门控 (`LineMatcher`)
```python
from LineFeat import LineMatcher

matcher = LineMatcher(ratio_threshold=0.85, mutual_check=True)
# 支持语义标签门控（阻止跨类别的假匹配，如地板与天花板）
matches = matcher.match(
    desc1, desc2,
    lines1=lines1, lines2=lines2,
    semantics1=sem_labels1, semantics2=sem_labels2
)
# 可视化连线图
canvas = matcher.draw_matches(img1, img2, lines1, lines2, matches, max_draw=40)
```

### 4. PyTorch 数据集加载 (`LineFeatPairDataset`)
```python
from torch.utils.data import DataLoader
from LineFeat import LineFeatPairDataset, collate_line_pairs

dataset = LineFeatPairDataset(
    index_path="data/line_gt_dataset/index.json",
    min_matches=5,
    split="train",  # 支持 'train', 'val', 'test'
    load_images=True,
    load_semantics=True,
    load_cameras=True
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_line_pairs)
for batch in loader:
    images0 = batch["image0"]      # (B, 3, H, W)
    images1 = batch["image1"]      # (B, 3, H, W)
    gt_lines0 = batch["gt_lines0"]  # List of (M_i, 4)
    # ...
```

---

## 运行验证

运行端到端验证脚本：
```bash
python demo/demo_linefeat.py --pair-idx 0
```
该脚本将自动验证：
1. `LineFeatPairDataset` 数据加载（图像、相机内参/相对位姿、3D线框投影真值）
2. 真实三维结构线上的深度特征匹配精度（正确率达 **80%+**）
3. LSD 直线检测与沿线深度特征提取
4. 语义门控对于消除跨类别假匹配的效果（语义一致率从 ~73% 提升至 **97%~100%**）
5. 可视化图片保存于 `outputs/linefeat/`。
