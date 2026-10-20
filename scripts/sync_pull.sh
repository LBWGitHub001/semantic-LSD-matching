#!/usr/bin/env bash
# ==============================================================================
# sync_pull.sh: 一键将远程服务器跑出的最新权重(checkpoints)与结果图(outputs)拉回本地
# ==============================================================================

REMOTE_HOST="root@cpod-1wozvemid8dz-s3.podtcp.compshare.cn"
REMOTE_PORT="22604"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[*] 正在从远程服务器拉取最新的权重 (checkpoints) 与可视化 (outputs)..."

# 1. 拉取训练权重
echo "  [1/2] 同步权重文件..."
rsync -avzP \
  -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
  "${REMOTE_HOST}:~/semantic-LSD-matching/checkpoints/" "${LOCAL_DIR}/checkpoints/"

# 2. 拉取生成的可视化结果
echo "  [2/2] 同步可视化结果..."
rsync -avzP \
  -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
  "${REMOTE_HOST}:~/semantic-LSD-matching/outputs/" "${LOCAL_DIR}/outputs/"

echo "[+] 远程 -> 本地 拉取完成！"
