# YAML Configuration Architecture

本项目为所有支持的网络模型提供标准化的 YAML 配置文件，统一调度**数据集生成**、**模型训练**与**评测可视化**三大核心生命周期。

## 1. 配置文件索引

| 模型名称 | 配置文件 | 任务类型 | 默认数据集 | 默认保存路径 |
|---|---|---|---|---|
| **SuperPoint** | [`superpoint.yaml`](superpoint.yaml) | 平面字母端点与长线几何匹配 | `data/planar_letters` | `checkpoints/SuperPoint` |
| **FCN_field** | [`fcn_field.yaml`](fcn_field.yaml) | DeepLSD 几何距离场单应速度估计 | `data/homo_matcher` | `checkpoints/FCN_field` |
| **xfeat_object** | [`xfeat_object.yaml`](xfeat_object.yaml) | 纯目标级二部图“连连看”匹配 | `data/structured3d_semantic_regions` | `checkpoints/xfeat_object_matcher` |
| **xfeat_line_matcher** | [`xfeat_line_matcher.yaml`](xfeat_line_matcher.yaml) | 目标语义区域 + 直线几何特征混合匹配 | `data/synthetic_semantic_lines` | `checkpoints/xfeat_line_matcher` |
| **semantic_line_matcher** | [`semantic_line_matcher.yaml`](semantic_line_matcher.yaml) | 双路 Sinkhorn 几何软约束直线匹配 | `data/synthetic_semantic_lines` | `checkpoints/semantic_line_matcher` |
| **mask_line** | [`mask_line.yaml`](mask_line.yaml) | 6 通道连续几何场端到端直线匹配 | `data/synthetic_line_scenes` | `checkpoints/mask_line` |
| **line_transformer** | [`line_transformer.yaml`](line_transformer.yaml) | Transformer 结构直线匹配基准网络 | `data/synthetic_line_scenes` | `checkpoints/line_transformer` |

---

## 2. 结构范式与字段含义

每个配置文件均由 4 个顶层结构小节组成：

```yaml
model:              # 网络模型架构超参数 (通道数, 特征维度, 注意力头数等)
  model_type: ...
  family: ...

data_generation:    # 数据集生成工厂参数 (对应 generate_dataset.py)
  task: ...
  out_dir: ...
  num_scenes: ...

train:              # 模型训练参数 (对应 train.py)
  data_dir: ...
  epochs: ...
  lr: ...
  save_dir: ...

eval:               # 模型评测参数 (对应 eval.py)
  checkpoint: ...
  data_dir: ...
  split: ...
  out_dir: ...
```

---

## 3. 使用方法

### 方式 A: 分步独立调用 (各脚本均支持 `--config / -c`)

```bash
# 1. 生成对应模型所需的数据集
python generate_dataset.py --config configs/superpoint.yaml

# 2. 启动模型训练 (显式 CLI 参数可覆盖 yaml 默认值，如修改 epochs 和 lr)
python train.py --config configs/superpoint.yaml --epochs 20 --lr 5e-4

# 3. 执行评测与可视化诊断
python eval.py --config configs/superpoint.yaml --split val
```

### 方式 B: 使用统一流水线调度器 (`run.py`)

```bash
# 一键串联执行完整流水线 (生成数据 -> 训练 -> 评测)
python run.py --config configs/superpoint.yaml --step all

# 或者只执行某个指定阶段
python run.py --config configs/superpoint.yaml --step train
```
