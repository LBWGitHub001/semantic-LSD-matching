#!/usr/bin/env python3
"""自动化模型权重下载与完整性校验脚本 (Model Downloader & Verifier).

支持下载:
1. SAM 旗舰版 ViT-H (sam_vit_h_4b8939.pth, ~2.56 GB)
2. DeepLSD 官方 Wireframe 权重 (deeplsd_wireframe.tar, ~42 MB)
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

MODELS = {
    "sam_vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "dest": Path("weights/sam/sam_vit_h_4b8939.pth"),
        "expected_size": 2564550879,  # ~2.56 GB
        "description": "SAM ViT-H 旗舰超大模型权重",
    },
    "deeplsd_wireframe": {
        "url": "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar",
        "dest": Path("third_party/DeepLSD/weights/deeplsd_wireframe.tar"),
        "expected_size": 102213120,
        "description": "DeepLSD Wireframe 几何距离场/角度场预训练权重",
    },
}


def download_file(url: str, dest_path: Path, expected_size: int | None = None) -> bool:
    """使用 wget 断点续传下载文件。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.is_file():
        actual_size = dest_path.stat().st_size
        if expected_size and actual_size == expected_size:
            print(f"[✓] 文件已存在且大小完整: {dest_path} ({actual_size / 1e6:.1f} MB)")
            return True
        elif actual_size > 0:
            print(f"[*] 文件已存在部分 ({actual_size / 1e6:.1f} MB)，尝试断点续传...")

    print(f"[*] 正在从 {url} 下载至 {dest_path}...")
    cmd = ["wget", "-c", url, "-O", str(dest_path)]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[!] 下载失败: {url} (退出码: {ret.returncode})", file=sys.stderr)
        return False

    if expected_size and dest_path.is_file():
        actual_size = dest_path.stat().st_size
        print(f"[✓] 下载成功: {dest_path} ({actual_size / 1e6:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="下载 SAM 与 DeepLSD 预训练模型权重")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["sam_vit_h", "deeplsd_wireframe"],
        choices=list(MODELS.keys()),
        help="指定要下载的模型列表",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("模型权重下载管理 (LSD-matching)")
    print("=" * 60)

    for key in args.models:
        info = MODELS[key]
        print(f"\n[>>>] 准备: {info['description']}")
        ok = download_file(info["url"], info["dest"], info.get("expected_size"))
        if not ok:
            print(f"[!] 警告: {key} 下载未完成")

    print("\n[✓] 模型准备流程完成！")


if __name__ == "__main__":
    main()
