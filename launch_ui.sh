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
echo "================================================================="

exec "${PYTHON_CMD}" app.py --port "${PORT}"
