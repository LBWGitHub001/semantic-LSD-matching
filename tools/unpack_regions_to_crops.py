#!/usr/bin/env python3
"""
Unpack Full Scene Images into Folders of Segmented Region Small Images.

For each pair in Structured3D Semantic Regions:
  - image_0.png -> image_0/region_00.png, region_01.png, ... (varying sizes [H_i, W_i], BGRA transparent mask)
  - image_1.png -> image_1/region_00.png, region_01.png, ... (varying sizes [H_j, W_j], BGRA transparent mask)

Usage:
  python tools/unpack_regions_to_crops.py --data-dir data/structured3d_semantic_regions --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def process_single_pair(pair_dir_str: str, overwrite: bool = False) -> tuple[str, int, int]:
    pair_dir = Path(pair_dir_str)
    img0_path = pair_dir / "image_0.png"
    img1_path = pair_dir / "image_1.png"
    reg0_path = pair_dir / "regions_0.json"
    reg1_path = pair_dir / "regions_1.json"

    if not (img0_path.is_file() and img1_path.is_file() and reg0_path.is_file() and reg1_path.is_file()):
        return pair_dir.name, 0, 0

    dir0 = pair_dir / "image_0"
    dir1 = pair_dir / "image_1"

    # View 0
    c0 = 0
    if overwrite or not dir0.exists() or len(list(dir0.glob("region_*.png"))) == 0:
        dir0.mkdir(parents=True, exist_ok=True)
        img0 = cv2.imread(str(img0_path))
        with open(reg0_path, "r", encoding="utf-8") as f:
            regs0 = json.load(f)

        for r in regs0:
            idx = r["idx"]
            m_path = pair_dir / "masks_0" / f"mask_{idx:02d}.png"
            if not m_path.is_file():
                continue
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

            x1, y1, x2, y2 = r["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img0.shape[1] - 1, x2), min(img0.shape[0] - 1, y2)
            if x2 < x1 or y2 < y1:
                continue

            crop_img = img0[y1 : y2 + 1, x1 : x2 + 1]
            crop_mask = m[y1 : y2 + 1, x1 : x2 + 1]

            # 4-channel BGRA: RGB is color, A is binary mask
            h_c, w_c = crop_img.shape[:2]
            bgra = np.zeros((h_c, w_c, 4), dtype=np.uint8)
            bin_mask = crop_mask > 0
            bgra[..., :3] = crop_img * bin_mask[..., None]
            bgra[..., 3] = np.where(bin_mask, 255, 0).astype(np.uint8)

            save_file = dir0 / f"region_{idx:02d}.png"
            cv2.imwrite(str(save_file), bgra)
            c0 += 1
    else:
        c0 = len(list(dir0.glob("region_*.png")))

    # View 1
    c1 = 0
    if overwrite or not dir1.exists() or len(list(dir1.glob("region_*.png"))) == 0:
        dir1.mkdir(parents=True, exist_ok=True)
        img1 = cv2.imread(str(img1_path))
        with open(reg1_path, "r", encoding="utf-8") as f:
            regs1 = json.load(f)

        for r in regs1:
            idx = r["idx"]
            m_path = pair_dir / "masks_1" / f"mask_{idx:02d}.png"
            if not m_path.is_file():
                continue
            m = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

            x1, y1, x2, y2 = r["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img1.shape[1] - 1, x2), min(img1.shape[0] - 1, y2)
            if x2 < x1 or y2 < y1:
                continue

            crop_img = img1[y1 : y2 + 1, x1 : x2 + 1]
            crop_mask = m[y1 : y2 + 1, x1 : x2 + 1]

            h_c, w_c = crop_img.shape[:2]
            bgra = np.zeros((h_c, w_c, 4), dtype=np.uint8)
            bin_mask = crop_mask > 0
            bgra[..., :3] = crop_img * bin_mask[..., None]
            bgra[..., 3] = np.where(bin_mask, 255, 0).astype(np.uint8)

            save_file = dir1 / f"region_{idx:02d}.png"
            cv2.imwrite(str(save_file), bgra)
            c1 += 1
    else:
        c1 = len(list(dir1.glob("region_*.png")))

    return pair_dir.name, c0, c1


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack whole images into folders of segmented small images")
    parser.add_argument("--data-dir", type=str, default="data/structured3d_semantic_regions", help="Dataset directory")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing crop folders")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {data_dir} not found.")

    pair_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "correspondence.json").is_file()])
    print("=" * 70)
    print("        Dataset Preprocessor: Unpack Images to Segmented Folders       ")
    print("=" * 70)
    print(f"  Target Directory:  {data_dir}")
    print(f"  Total Scene Pairs: {len(pair_dirs)}")
    print(f"  Worker Processes:  {args.num_workers}")
    print(f"  Overwrite:         {args.overwrite}")
    print("=" * 70 + "\n")

    t0 = time.time()
    total_crops = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_pair, str(p), args.overwrite): p for p in pair_dirs}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Unpacking Crops", unit="pair")
        for fut in pbar:
            pair_name, c0, c1 = fut.result()
            total_crops += (c0 + c1)

    elapsed = time.time() - t0
    print(f"\n[+] Successfully finished! Unpacked {total_crops} small images across {len(pair_dirs)} pairs in {elapsed:.1f}s.")
    print(f"    Structure created in each pair directory:")
    print(f"      - image_0/  -> region_00.png, region_01.png, ...")
    print(f"      - image_1/  -> region_00.png, region_01.png, ...")


if __name__ == "__main__":
    main()
