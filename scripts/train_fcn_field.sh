#!/usr/bin/env bash
# 训练 FCN_field。当前损失只有 symmetry：v12 和 v21 互为逆映射。
# 批次大小在 train.py 里固定为 1。不要加 --flow-weight 等，那些参数现在不会执行。
#
#   bash scripts/train_fcn_field.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python
fi

exec "$PYTHON" train.py \
  --model FCN_field \
  --data-dir data/homo_matcher \
  --epochs 12 \
  --lr 1e-4 \
  --save-dir checkpoints/FCN_field \
  --device cuda
