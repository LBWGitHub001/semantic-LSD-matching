# FCN_field 稠密速度场

两张图各自带上 DeepLSD 的距离场和角度场。全卷积网络只预测稠密速度 \((v_x, v_y)\)。方向是 \(\operatorname{atan2}(v_y, v_x)\)，大小是向量长度。不再拟合成单应矩阵。

| 项目 | 说明 |
|------|------|
| 注册名 | `FCN_field` |
| 包路径 | `models/FCN_field/` |
| 损失 | `FCNFieldLoss`，\(\lambda_1 L_1+\lambda_2 L_2\) |
| 家族 | `homography` |
| 数据目录 | 仍是 `data/homo_matcher`，不用重新生成 |

数据侧索引见 [`docs/homo_matcher.md`](../../docs/homo_matcher.md)。

## 数据流

![从五通道到损失](../../docs/images/homo_matcher/flow.jpg)

共享 FCN 把特征上采样回原图大小。两侧特征拼在一起，预测两张速度图。

## 五个通道

1. 灰度图
2. 距离场除以 5
3. \(\sin 2\theta\)
4. \(\cos 2\theta\)
5. 距离小于 1.6 像素的亮带

![距离场与角度场](../../docs/images/homo_matcher/fields.jpg)

## 损失

\(L_1\)：两个速度场互为反向。从一点走出 \(v_{12}\)，再走 \(v_{21}\)，应当回到原点。反方向同样要求。

\(L_2\)：距离场亮脊上的深度序列，用 \(v_{12}\) 把采样点移到图 2，各自归一化后做 L2。

![亮脊端点画在原图上](../../docs/images/homo_matcher/ridges.jpg)

![一条亮脊和两条深度序列](../../docs/images/homo_matcher/ridges_l2.jpg)

总损失 \(\lambda_1 L_1+\lambda_2 L_2\)。

## 参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `--cycle-weight` | 1 | \(\lambda_1\) |
| `--depth-weight` | 1 | \(\lambda_2\) |
| `--line-sample-points` | 8 | 每条脊上的深度采样数 |

```bash
python train.py --model FCN_field --data-dir data/homo_matcher --cycle-weight 1 --depth-weight 1
```
