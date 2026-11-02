# 数据集生成与组织规范 (Datasets & Generation Pipeline)

本模块负责基于 **ScanNet-1500** 室内标准校准序列，通过融合几何分支 (DeepLSD)、点特征分支 (SuperPoint) 与旗舰语义分支 (SAM ViT-H)，自动批量构建供 GlueStick 与语义增强网络直接训练与评测的多模态数据集。

---

## 1. 核心文件组成

- **[`build_semantic_gluestick_dataset.py`](file:///home/lbw/Project/new-LSD-matching/datasets/build_semantic_gluestick_dataset.py)**：
  - 数据集批量构建主生成程序；
  - 读取 `pairs_calibrated.txt` 配对表，自适应将图像规整至 640x480 并同步重算相机内参 $K$；
  - 自动运行 DeepLSD (线段+距离场)、SuperPoint (关键点+描述子) 与 SAM ViT-H (256维线特征池化 + 实例 Mask 轮廓分析)；
  - 基于真实相机相对位姿 $T_{0\to 1}$ 与稠密深度图严格反投影，计算点和线的监督匹配真值 (`gt_matches0`, `gt_line_matches0`)；
  - 支持混合存储调度（如仅前 3 个场景输出为 `image+npz`，其余场景输出为纯 `npz` 格式）。
- **[`semantic_gluestick_dataset.py`](file:///home/lbw/Project/new-LSD-matching/datasets/semantic_gluestick_dataset.py)**：
  - 标准 PyTorch Dataset 样本加载器与批量组装器；
  - 严格支持场景隔离式的 Train / Val 数据集切分（杜绝跨视角数据泄露）；
  - `semantic_gluestick_collate_fn` 动态将变长关键点、几何线段、拓扑邻接矩阵及 SAM 语义特征对齐组装为模型张量。

---

## 2. 单样本存储数据结构字典

单对 `.npz` 文件中包含以下全部字段：

| 键名 | 形状 / 维度 | 字段含义与技术规范 |
| :--- | :--- | :--- |
| `image0`, `image1` | `(480, 640)` uint8 | 原始灰度对齐图像 (纯 npz 模式内嵌；若为 image+npz 模式则存路径) |
| `keypoints0`, `keypoints1` | `(N, 2)` float32 | 线框交点 (Junctions) 坐标 $[x, y]$ |
| `descriptors0`, `descriptors1` | `(256, N)` float32 | SuperPoint 视觉外观特征描述子 |
| `keypoint_scores0`, `1` | `(N,)` float32 | 关键点显著性响应评分 |
| `lines0`, `lines1` | `(M, 2, 2)` float32 | DeepLSD 亚像素几何线段端点对 $[(x_1, y_1), (x_2, y_2)]$ |
| `line_scores0`, `1` | `(M,)` float32 | 线段几何置信度评分 |
| `lines_junc_idx0`, `1` | `(M, 2)` int64 | 线段端点对应到 `keypoints` 数组的索引映射 |
| `connectivity0`, `1` | `(N, N)` bool | 线框图端点拓扑邻接矩阵 |
| `df0`, `df1` | `(480, 640)` float32 | DeepLSD 亚像素几何连续距离场 (0~5.0 px) |
| `line_semantic_descs0`, `1` | `(256, M)` float32 | **SAM ViT-H 沿线 16 点高密采样均值池化与 L2 归一化语义描述子** |
| `line_mask_attrs0`, `1` | `(M, 3)` float32 | **SAM 实例掩膜属性 [主导实例ID, 掩膜稳定性得分, 是否为物理轮廓边界]** |
| `gt_matches0` | `(N0,)` int64 | 基于位姿反投影推导的关键点匹配真值 (>=0 为对应索引，-1 为未匹配) |
| `gt_line_matches0` | `(M0,)` int64 | 基于几何投影与重合率推导的几何线段匹配真值 |
| `K0`, `K1` | `(3, 3)` float32 | 规整至 640x480 分辨率适配后的相机内参矩阵 |
| `T_0to1` | `(4, 4)` float32 | 视图 0 到视图 1 的真实相对相机刚体外参变换 $[R \mid t]$ |

---

## 3. 运行指南

```bash
# 执行全量数据构建
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python datasets/build_semantic_gluestick_dataset.py \
    --pairs-file data/scannet/scannet1500/pairs_calibrated.txt \
    --data-dir data/scannet/scannet1500 \
    --out-dir data/scannet_semantic_gluestick

# 测试 DataLoader 装载与批对齐
/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python datasets/semantic_gluestick_dataset.py \
    --data-dir data/scannet_semantic_gluestick \
    --batch-size 2
```
