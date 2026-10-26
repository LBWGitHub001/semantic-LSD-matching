#!/usr/bin/env bash
# Semantic LSD Matching - Unified Visual Management System Launcher
# Automatically detects conda environment and launches the Web UI.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CONDA_PYTHON="/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python"

if [ -f "${CONDA_PYTHON}" ]; then
    PYTHON_CMD="${CONDA_PYTHON}"
else
    PYTHON_CMD="python3"
fi

PORT="${1:-7860}"

echo "================================================================="
echo "   Launching Semantic LSD Matching Web Management Tool...        "
echo "================================================================="
echo "  • Python Interpreter : ${PYTHON_CMD}"
echo "  • Web Port           : ${PORT}"
echo "  • Local Access URL   : http://localhost:${PORT}"

# Check if port is already occupied
OCCUPY_PID=$(lsof -ti :${PORT} 2>/dev/null || fuser ${PORT}/tcp 2>/dev/null | awk '{print $1}' || true)
if [ -n "${OCCUPY_PID}" ]; then
    OCCUPY_CMD=$(ps -p "${OCCUPY_PID}" -o cmd= 2>/dev/null || true)
    if echo "${OCCUPY_CMD}" | grep -q "app.py"; then
        echo "  • 发现端口 ${PORT} 正在运行历史 app.py 实例 (PID: ${OCCUPY_PID})，正在平滑重启..."
        kill -15 "${OCCUPY_PID}" 2>/dev/null || true
        for i in {1..15}; do
            if ! kill -0 "${OCCUPY_PID}" 2>/dev/null; then
                break
            fi
            sleep 0.2
        done
        if kill -0 "${OCCUPY_PID}" 2>/dev/null; then
            kill -9 "${OCCUPY_PID}" 2>/dev/null || true
        fi
        echo "  • 历史实例已释放，正在启动新服务..."
    else
        echo "  [!] 提示: 端口 ${PORT} 当前被其他进程占用 (PID: ${OCCUPY_PID})"
        echo "      正在尝试释放端口..."
        fuser -k "${PORT}/tcp" 2>/dev/null || true
        sleep 0.5
    fi
fi

echo "================================================================="

exec "${PYTHON_CMD}" app.py --port "${PORT}"
