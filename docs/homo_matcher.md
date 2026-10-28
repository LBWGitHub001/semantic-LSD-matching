# DeepLSD 单应网络（索引）

架构、损失和配图在 [`models/homo_matcher/README.md`](../models/homo_matcher/README.md)。

## 注册

- 模型：`homo_matcher`
- 数据组件：`DatasetGeneratorFactory` 的 `homo_matcher`
- 验证：`EvaluatorFactory` 的 `homography`

## 生成数据

每个样本保存两张 RGB、两张深度，以及 DeepLSD 的距离场和角度场。不保存 LSD 线段。

```bash
python tools/build_homo_matcher_dataset.py --max-scenes 1 --max-pairs 4 --out-dir data/homo_matcher
```

同一件事也可以写成：

```bash
python generate_dataset.py --task homo_matcher --max-scenes 1 --max-pairs 4 --out-dir data/homo_matcher
```

## 训练

```bash
python train.py --model homo_matcher --data-dir data/homo_matcher --epochs 12 --cycle-weight 1 --depth-weight 1
```

## 验证

```bash
python scripts/eval_homo_matcher.py \
  --checkpoint checkpoints/homo_matcher/best_model.pth \
  --data-dir data/homo_matcher \
  --split test
```

输出是 \(L_1\)、\(L_2\)，以及 \(H_{12}\) 把亮脊投到第二张图上的对比图。

## 只看直线端点

不读 CAD，不调用 LSD。端点完全来自距离场亮脊。

```bash
python scripts/test_deeplsd_lines.py path/to/rgb_rawlight.png --out ridges.jpg
```

![亮脊端点](images/homo_matcher/ridges.jpg)
