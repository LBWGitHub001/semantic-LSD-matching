# SemanticLineMatcher 设计说明（索引）

**主文档（架构、损失、参数表、Mermaid 图）**：  
[`models/semantic_line_matcher/README.md`](../models/semantic_line_matcher/README.md)

本文档仅保留**数据与训练侧**要点，避免与 README 重复。

## 注册与别名

- 注册名：`semantic_line_matcher` / `semantic_line`
- 包路径：`models/semantic_line_matcher/`

## 数据（SemanticRegionDataset）

- `collate_region_pairs` 加载 `patch_lines.npz` 的 `ids0/ids1`（CAD `line_id`）
- 透传 `line_matches`，产出 `gt_line_pairs_flat`、`unmatched_lines_flat_*`、`has_line_ids`
- 无 ids 时 `has_line_ids=False`，训练跳过线级 KL 与 JSD

## 训练示例

```bash
python train.py \
  --model semantic_line_matcher \
  --data-dir data/your_semantic_region_dataset \
  --region-weight 1.0 --line-weight 1.0 \
  --sem-weight 0.1 --geo-weight 0.2 \
  --adj-sigma 0.3 --line-sample-points 8 \
  --dustbin-weight 0.3
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
