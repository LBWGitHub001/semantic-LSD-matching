# third_party

## XFeat（官方实现）

本目录应包含 [verlab/accelerated_features](https://github.com/verlab/accelerated_features) 克隆：

```bash
cd /path/to/semantic-LSD-match
git clone https://github.com/verlab/accelerated_features.git third_party/accelerated_features
```

确认存在权重文件：

- `third_party/accelerated_features/weights/xfeat.pt`

然后在项目根目录运行：

```bash
python demo/demo_xfeat.py
```
