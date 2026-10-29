# FCN_field（索引）

架构、损失和配图在 [`models/FCN_field/README.md`](../models/FCN_field/README.md)。

## 注册

- 模型：`FCN_field`
- 数据组件：`DatasetGeneratorFactory` 的 `FCN_field`
- 验证：`EvaluatorFactory` 的 `homography`
- 已生成的数据目录仍是 `data/homo_matcher`

## 生成数据

```bash
python generate_dataset.py --task FCN_field --max-scenes 1 --max-pairs 4 --out-dir data/homo_matcher
```

全量：

```bash
python generate_dataset.py --task FCN_field --s3d-dir Structured3D_perspective_full_00/Structured3D --out-dir data/homo_matcher
```

## 训练

```bash
python train.py --model FCN_field --data-dir data/homo_matcher --epochs 12 --cycle-weight 1 --depth-weight 1
```

## 验证

```bash
python scripts/eval_homo_matcher.py \
  --checkpoint checkpoints/FCN_field/best_model.pth \
  --data-dir data/homo_matcher \
  --split test
```

![亮脊端点](images/homo_matcher/ridges.jpg)
