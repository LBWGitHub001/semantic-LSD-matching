#!/usr/bin/env python3
"""解压 Structured3D 压缩包，已解出且大小一致的文件会跳过。"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile


def _format_bytes(num: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num:.2f} B"


def _safe_target(out_dir: str, name: str) -> str:
    target = os.path.realpath(os.path.join(out_dir, name))
    root = os.path.realpath(out_dir)
    if target != root and not target.startswith(root + os.sep):
        raise RuntimeError(f"压缩包路径越界：{name}")
    return target


def unzip(zip_path: str, out_dir: str) -> None:
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        files = [info for info in infos if not info.is_dir()]
        total_bytes = sum(info.file_size for info in files)
        done_bytes = 0
        skipped = 0
        extracted = 0
        started = time.monotonic()

        for info in infos:
            if info.is_dir():
                os.makedirs(_safe_target(out_dir, info.filename), exist_ok=True)

        for index, info in enumerate(files, start=1):
            target = _safe_target(out_dir, info.filename)
            if os.path.isfile(target) and os.path.getsize(target) == info.file_size:
                skipped += 1
                done_bytes += info.file_size
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(info, "r") as source, open(target, "wb") as handle:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
                        done_bytes += len(block)
                extracted += 1

            elapsed = max(time.monotonic() - started, 1e-6)
            speed = done_bytes / elapsed
            percent = (done_bytes / total_bytes * 100) if total_bytes else 100
            print(
                f"\r{percent:6.2f}%  {_format_bytes(done_bytes)} / {_format_bytes(total_bytes)}"
                f"  {_format_bytes(speed)}/s  {index}/{len(files)}",
                end="",
                flush=True,
            )

    print()
    print(f"解压完成：新写出 {extracted} 个文件，跳过 {skipped} 个已存在文件")
    print(f"输出目录：{os.path.abspath(out_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解压 Structured3D zip，支持中断后跳过已完成文件")
    parser.add_argument("zip_path", help="zip 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="解压目录，默认与 zip 同级、同名去掉 .zip",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output
    if not out_dir:
        stem = os.path.basename(args.zip_path)
        if stem.lower().endswith(".zip"):
            stem = stem[:-4]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.zip_path)), stem)
    try:
        unzip(args.zip_path, out_dir)
    except KeyboardInterrupt:
        print("\n已中断。再次运行同一命令会跳过已经解压且大小一致的文件。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"解压失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
