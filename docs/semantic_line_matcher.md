# SemanticLineMatcher 设计说明

独立于 `XFeatLineMatcher` 的双输出匹配网络：区域匹配 + 直线匹配。

注册名：`semantic_line_matcher` / `semantic_line`  
包路径：`models/semantic_line_matcher/`

## 目标输出

| 键 | 含义 |
|----|------|
| `log_P` / `P` | 区域 Sinkhorn（含 dustbin） |
| `log_P_line` / `P_line` | 直线 Sinkhorn（含 dustbin） |
| `line_parents0/1` | 展平后每条线的父区域下标 |

## 网络

1. 共享 XFeat（或 pyramid）对每个 region patch 卷积 → 特征图  
2. 区域向量 `f_r`：GAP（多尺度拼接）+ 可选 bbox 几何 → fusion → Graph Transformer → 区域 Sinkhorn  
3. 直线向量 `f_ℓ`：在特征图上沿线采样 `T` 点（默认 8）+ Conv1d 聚合  
4. 直线描述子：`d = Proj([f_ℓ ‖ f_r])` → 展平有效槽 → 直线 Sinkhorn  

相对邻接 **不进入** 打分网络，只在损失中约束。

## 损失

\[
\mathcal{L}=\lambda_R\mathcal{L}_R+\lambda_L\mathcal{L}_L+\lambda_S\mathcal{L}_{\mathrm{sem}}+\lambda_G\mathcal{L}_{\mathrm{geo}}
\]

| 项 | 默认权重 | 含义 |
|----|----------|------|
| 区域 KL | 1.0 | 硬标签 \(\mathrm{KL}(Y_R\|P_R)\) ≡ 连连看 NLL + dustbin |
| 直线 KL | 1.0 | 同上，作用在 `log_P_line` |
| 语义集中 JSD | 0.1 | 双向线→语义边缘化矩阵上，同父区域内广义 JSD |
| 邻接 geo | 0.2 | 软邻接 \(A_0 \approx P A_1 P^\top\)（相邻区域任意视角仍相邻） |

旧全局平移 `motion_weight` 对本模型默认关闭。

### 线→语义矩阵

\(M^{A\to B}_{i,s}=\sum_{j:\mathrm{parent}(j)=s}P_L[i,j]\)，dustbin 维保留；反向由 \(P_L^\top\) 得到。

### 邻接保持

\(A[i,k]=\exp(-\|c_i-c_k\|/\sigma)\)（对角为 0），\(\sigma\) 默认 0.3。

## 数据

`SemanticRegionDataset` / `collate_region_pairs`：

- 加载 `patch_lines.npz` 的 `ids0/ids1`（CAD `line_id`）  
- 透传 `line_matches`  
- 产出 `gt_line_pairs_flat` / `unmatched_lines_flat_*` / `has_line_ids`  
- 无 ids 时跳过线级监督（`has_line_ids=False`）

## 训练示例

```bash
python train.py \
  --model semantic_line_matcher \
  --data-dir data/your_semantic_region_dataset \
  --region-weight 1.0 --line-weight 1.0 \
  --sem-weight 0.1 --geo-weight 0.2 \
  --adj-sigma 0.3 --line-sample-points 8 \
  --dustbin-weight 0.2 --motion-weight 0.0
```

## 测试

```bash
python -m unittest \
  tests.test_along_line_encoder \
  tests.test_semantic_line_region_branch \
  tests.test_semantic_line_matcher \
  tests.test_semantic_line_loss \
  tests.test_semantic_line_integration -v
```

## 文件

- `models/semantic_line_matcher/model.py` — `SemanticLineMatcher`  
- `models/semantic_line_matcher/line_encoder.py` — 沿线纹理编码  
- `models/semantic_line_matcher/loss.py` — 四项损失  
- `datasets/semantic_region_dataset.py` — ids / flat GT  
- `train.py` — CLI 接线  
