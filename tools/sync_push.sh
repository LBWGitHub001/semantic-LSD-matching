#!/usr/bin/env bash
# ==============================================================================
# sync_push.sh: 一键将本地代码与修改推送到远程 GPU 服务器
# ==============================================================================

REMOTE_HOST="root@cpod-1wozvemid8dz-s3.podtcp.compshare.cn"
REMOTE_PORT="22604"
REMOTE_DIR="~/semantic-LSD-matching/"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

echo "[*] 正在将本地修改同步到远程服务器..."
echo "    本地路径: ${LOCAL_DIR}"
echo "    远程路径: ${REMOTE_HOST}:${REMOTE_DIR}"

rsync -avzP \
  --exclude "Structured3D*" \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".idea" \
  -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
  "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"

echo "[+] 本地 -> 远程 同步完成！"
