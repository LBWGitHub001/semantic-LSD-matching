"""一键下载项目核心论文官方原版 PDF 脚本 (Paper PDF Downloader).

包含:
1. GlueStick (CVPR 2023)
2. DeepLSD (CVPR 2023)
3. Segment Anything (ICCV 2023)
4. SuperPoint (CVPRW 2018)
5. ScanNet (CVPR 2017)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PAPERS = [
    {
        "name": "GlueStick_CVPR2023.pdf",
        "title": "GlueStick: A Robust Line and Keypoint Matcher",
        "url": "https://arxiv.org/pdf/2304.02008.pdf",
        "venue": "CVPR 2023",
    },
    {
        "name": "DeepLSD_CVPR2023.pdf",
        "title": "DeepLSD: Line Segment Detection and Refinement with Deep Image Gradients",
        "url": "https://arxiv.org/pdf/2212.07766.pdf",
        "venue": "CVPR 2023",
    },
    {
        "name": "SAM_ICCV2023.pdf",
        "title": "Segment Anything",
        "url": "https://arxiv.org/pdf/2304.02643.pdf",
        "venue": "ICCV 2023",
    },
    {
        "name": "SuperPoint_CVPRW2018.pdf",
        "title": "SuperPoint: Self-Supervised Interest Point Detection and Description",
        "url": "https://arxiv.org/pdf/1712.07629.pdf",
        "venue": "CVPRW 2018",
    },
    {
        "name": "ScanNet_CVPR2017.pdf",
        "title": "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes",
        "url": "https://arxiv.org/pdf/1702.04405.pdf",
        "venue": "CVPR 2017",
    },
]


def main():
    parser = argparse.ArgumentParser(description="下载项目核心参考论文 PDF")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent), help="保存目录")
    parser.add_argument("--list-only", action="store_true", help="仅列出论文清单不下载")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  📚 项目核心文献论文库 (Core Research Papers)")
    print("=" * 70)
    for idx, p in enumerate(PAPERS, start=1):
        print(f"[{idx}] {p['title']} ({p['venue']})")
        print(f"    URL: {p['url']}")
        print(f"    Save: {p['name']}")

    if args.list_only:
        print("\n[✓] 论文清单展示完毕 (list-only 模式，未执行下载)")
        return

    print("\n[*] 开始下载论文 PDF 文件...")
    for p in PAPERS:
        dest = out_dir / p["name"]
        if dest.is_file() and dest.stat().st_size > 100000:
            print(f"  [✓] 论文已存在: {p['name']} ({dest.stat().st_size / 1e6:.2f} MB)")
            continue

        print(f"  [-] 正在下载: {p['title']} -> {p['name']}...")
        cmd = ["wget", "-q", "--show-progress", "-c", p["url"], "-O", str(dest)]
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print(f"  [✓] 下载完成: {p['name']}")
        else:
            print(f"  [!] 下载失败 (可通过浏览器手动访问): {p['url']}")

    print(f"\n[✓] 论文下载流程结束，文件已保存在: {out_dir}")


if __name__ == "__main__":
    main()
