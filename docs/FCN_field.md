# FCN_field

架构和配图在 [`models/FCN_field/README.md`](../models/FCN_field/README.md)。

网络只输出稠密速度 \((v_x, v_y)\)。方向是 \(\operatorname{atan2}(v_y, v_x)\)，大小是向量长度。

## 监督

当前只有 `symmetry_loss`。\(T_{12}(x)=x+v_{12}(x)\)，\(T_{21}(x)=x+v_{21}(x)\)。每个像素先走一张速度图，再在落点走另一张，应当回到原来的像素。两个方向都算。

## 注册

- 模型：`FCN_field`
- 数据任务：`generate_dataset.py --task FCN_field`
- 验证家族：`homography`
- 数据目录：`data/homo_matcher`

## 命令

```bash
python generate_dataset.py --task FCN_field --max-scenes 1 --max-pairs 4 --out-dir data/homo_matcher
```

```bash
python train.py \
  --model FCN_field \
  --data-dir data/homo_matcher \
  --epochs 12 \
  --lr 1e-4
```

```bash
python scripts/eval_homo_matcher.py \
  --checkpoint checkpoints/FCN_field/best_model.pth \
  --data-dir data/homo_matcher \
  --split test
```
