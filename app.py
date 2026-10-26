#!/usr/bin/env python3
"""
Semantic LSD Matching - Web Management System Launcher
Starts the Tornado web server and opens the management interface.

Usage:
    python app.py [--port 7860] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Auto-switch to project conda environment if tornado or torch is missing
CONDA_PYTHON = "/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python"
try:
    import tornado
except ImportError:
    if os.path.isfile(CONDA_PYTHON) and sys.executable != CONDA_PYTHON:
        os.execv(CONDA_PYTHON, [CONDA_PYTHON] + sys.argv)
    else:
        raise

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web.server import start_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic LSD Matching Web Management System")
    parser.add_argument("--port", "-p", type=int, default=7860, help="Web server port (default: 7860)")
    parser.add_argument("--host", "-H", type=str, default="0.0.0.0", help="Web server bind host (default: 0.0.0.0)")
    args = parser.parse_args()

    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
