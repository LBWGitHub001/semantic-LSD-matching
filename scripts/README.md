# LineFeat 统一脚本工具箱 (`scripts/`) 使用手册

本目录汇集了项目全部可执行脚本与生产力工具，按功能划分为 **推断与评估**、**网络测试与演示**、**模型训练与缓存**、**云端同步** 以及 **数据集预处理工具**。

---

## 脚本速查表

| 脚本文件 | 核心功能 | 典型运行命令 | 默认模型 |
| :--- | :--- | :--- | :--- |
| [`example_call.py`](example_call.py) | **极简推断示例** | `python scripts/example_call.py` | `MaskLineFeatMatchNet` |
| [`test_matching.py`](test_matching.py) | **基准定量评测** | `python scripts/test_matching.py --mode both` | `MaskLineFeatMatchNet` |
| [`demo_mask_network.py`](demo_mask_network.py) | **几何场演示** | `python scripts/demo_mask_network.py` | `MaskLineFeatMatchNet` |
| [`run_unit_tests.py`](run_unit_tests.py) | **数学单元测试** | `python scripts/run_unit_tests.py` | `MaskLineFeatMatchNet` |
| [`verify_toy_task.py`](verify_toy_task.py) | **极简可行性验证** | `python scripts/verify_toy_task.py` | `MaskLineFeatMatchNet` |
| [`train.py`](train.py) | **网络训练与微调** | `python scripts/train.py --use-cache --amp` | `MaskLineFeatMatchNet` |
| [`cache_line_features.py`](cache_line_features.py) | **32核特征预提取** | `python scripts/cache_line_features.py` | ResNet-50 骨干 |
| [`sync_push.sh`](sync_push.sh) | **一键推送远程** | `./scripts/sync_push.sh` | - |
| [`sync_pull.sh`](sync_pull.sh) | **一键拉取结果** | `./scripts/sync_pull.sh` | - |
| [`build_line_gt_dataset.py`](build_line_gt_dataset.py) | **真值库重建** | `python scripts/build_line_gt_dataset.py` | 3D 线框投影引擎 |
| [`download_structured3d.py`](download_structured3d.py) | **数据源下载** | `python scripts/download_structured3d.py` | - |
| [`unzip_structured3d.py`](unzip_structured3d.py) | **数据解压扁平化** | `python scripts/unzip_structured3d.py <zip_path>` | - |

---

## 详细脚本使用指南

### 1. [`example_call.py`](example_call.py)：Python 极简调用纯匹配 (零检测器)
* **设计意图**：演示如何在外部已具备预检测线段坐标的情形下，5 行代码直接完成匹配并获取 $255 \times 255$ 规范矩阵。
* **运行命令**：
  ```bash
  python scripts/example_call.py
  ```
* **输出**：
  - 终端输出各线段匹配对的置信度得分以及与真值（Ground Truth）的比对结果；
  - 打印规范矩阵尺寸 `(255, 255)` 及激活子矩阵统计量；
  - 自动在 `outputs/pure_matching/example_call_vis.jpg` 生成高质量双目连线图。

---

### 2. [`test_matching.py`](test_matching.py)：精选测试集定量基准评测
* **设计意图**：针对精选的 4 组真实旋转/位移测试集进行全自动 Precision / Recall 评测。
* **参数说明**：
  - `--mode`：评测模式，可选 `A`（纯真值打乱测试）、`B`（混入单侧未匹配干扰线测试）、`both`（默认两者都测）；
  - `--model-type`：选择评估模型，`mask`（默认，`MaskLineFeatMatchNet`）或 `linefeat`（经典 `LineMatcherNet`）；
  - `--pair`：指定测试样本（如 `pair1_small_rotation`, `pair2_medium_rotation`, `all`）；
  - `--thresh`：相似度过滤阈值，默认 `0.70`。
* **常用范例**：
  ```bash
  # 评测默认 MaskLineFeatMatchNet 模型的所有模式
  python scripts/test_matching.py --mode both

  # 对比经典 LineMatcherNet 在大旋转场景下的性能
  python scripts/test_matching.py --model-type linefeat --pair pair3_large_rotation --mode A
  ```

---

### 3. [`demo_mask_network.py`](demo_mask_network.py)：6 通道连续几何场与矩阵热力图演示
* **设计意图**：直观可视化输入网络的 6 通道几何场以及输出的 $255 \times 255$ 匹配概率矩阵。
* **运行命令**：
  ```bash
  python scripts/demo_mask_network.py
  ```
* **生成成果（保存在 `outputs/field_network/`）**：
  - `view0_geometry_fields.jpg`：RGB 原图、归一化欧氏距离场（EDT）与连续朝向场（AFM）的三联全景图；
  - `canonical_255x255_probability_matrix.png`：输出的 $255 \times 255$ 匹配概率矩阵热力图。

---

### 4. [`run_unit_tests.py`](run_unit_tests.py)：数学单元测试
* **设计意图**：验证 `MaskLineFeatMatchNet` 的核心数学性质是否严格成立。
* **测试项**：
  1. **形状规格测试**：输出张量形状严格为 $(B, 255, 255)$，非激活区域严格为 0.0；
  2. **置换等变性测试（Permutation Equivariance）**：交换输入线段顺序，输出矩阵对应行列发生严格对应置换，最大误差 $< 3 \times 10^{-8}$；
  3. **可微性测试（Differentiability）**：在 $255 \times 255$ 矩阵上施加损失反向传播，梯度可正常回传至骨干网络与所有投影层。
* **运行命令**：
  ```bash
  python scripts/run_unit_tests.py
  ```

---

### 5. [`verify_toy_task.py`](verify_toy_task.py)：极简可控 2D 任务可行性验证 (Sanity Check)
* **设计意图**：构建 100% 干净、无遮挡歧义的合成 2D 几何线框任务，验证模型算法与 Sinkhorn 矩阵映射机制是否可行。
* **运行效果**：仅需 **1.1 秒（15 步迭代）**，匹配准确率从 10% 瞬间收敛至 **100%**，并在 `outputs/toy_verification/` 导出完美连线大图。
* **运行命令**：
  ```bash
  python scripts/verify_toy_task.py
  ```

---

### 6. [`train.py`](train.py)：端到端训练与微调脚本
* **设计意图**：在 Structured3D 投影线段真值库上训练匹配模型。
* **主要参数**：
  - `--model-type`：模型架构，`mask`（默认 `MaskLineFeatMatchNet`）或 `linefeat`；
  - `--use-cache`：启用高速特征缓存，每轮仅需 ~35 秒；
  - `--amp`：开启 FP16 自动混合精度加速与显存节省；
  - `--epochs`：训练轮数，默认 10；
  - `--lr`：初始学习率，默认 1e-4（配备 CosineAnnealingLR 退火调度）；
  - `--save-dir`：权重保存路径，默认 `checkpoints/linefeat/`。
* **运行范例**：
  ```bash
  # 极速缓存训练主打模型
  python scripts/train.py --model-type mask --use-cache --epochs 15 --amp
  ```

---

### 6. [`cache_line_features.py`](cache_line_features.py)：32 核异步视觉特征预提取
* **设计意图**：基于生产者-消费者多线程架构，提前提取所有真值线段的 CNN 视觉向量，并压缩存入 RAM/硬盘，将后续训练提速 50 倍。
* **运行命令**：
  ```bash
  python scripts/cache_line_features.py --num-threads 32 --batch-size 8
  ```
  生成文件将保存在 `data/line_feature_cache/` 中。

---

### 7. [`sync_push.sh`](sync_push.sh) 与 [`sync_pull.sh`](sync_pull.sh)：本地与远程 GPU 双向同步
* **设计意图**：配合 SSH 密钥认证，秒级打通本地与云端 GPU 环境，自动排除大体积 zip 数据包。
* **运行方式（在本地终端执行）**：
  ```bash
  # 1. 本地写完代码或文档，一键推送到远程 GPU 服务器：
  ./scripts/sync_push.sh

  # 2. 远程跑完实验或训练出新权重后，一键拉回本地：
  ./scripts/sync_pull.sh
  ```
