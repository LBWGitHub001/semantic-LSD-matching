"""Render per-view line and semantic overlays for Structured3D perspective data.

Each viewpoint becomes one folder with two images:
  lines.jpg     RGB with visible CAD lines
  semantic.jpg  RGB blended with the semantic mask
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from datasets.read_structured3d import (
    STRUCTURED3D_CLASSES,
    Structured3DView,
    generate_semantic_colormap,
)
from generate_dataset import extract_structured3d_gt_lines


def _semantic_overlay(bgr: np.ndarray, view: Structured3DView) -> np.ndarray:
    mask = view.load_semantic()
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    palette = generate_semantic_colormap(len(STRUCTURED3D_CLASSES))
    color = palette[np.clip(mask, 0, len(STRUCTURED3D_CLASSES) - 1)]
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(bgr, 0.45, color_bgr, 0.55, 0)


def _line_overlay(bgr: np.ndarray, scene_anno: dict, view: Structured3DView) -> np.ndarray:
    h, w = bgr.shape[:2]
    lines = extract_structured3d_gt_lines(
        scene_anno,
        view,
        target_h=h,
        target_w=w,
        min_length=15.0,
        min_vis_ratio=0.65,
    )
    canvas = bgr.copy()
    for rec in lines:
        coords = rec.get("coords") or []
        if len(coords) < 4:
            continue
        p0 = (int(round(coords[0])), int(round(coords[1])))
        p1 = (int(round(coords[2])), int(round(coords[3])))
        cv2.line(canvas, p0, p1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, (0, 80, 255), 3, cv2.LINE_AA)
    return canvas


def process_scene(scene_dir: str, out_root: str) -> tuple[int, int]:
    scene_path = Path(scene_dir)
    out_path = Path(out_root)
    anno_path = scene_path / "annotation_3d.json"
    if not anno_path.is_file():
        return 0, 0
    import json

    with open(anno_path, "r", encoding="utf-8") as handle:
        anno = json.load(handle)

    written = 0
    skipped = 0
    for view_dir in sorted(scene_path.glob("2D_rendering/*/perspective/full/*")):
        if not (view_dir / "rgb_rawlight.png").is_file():
            continue
        room_id = view_dir.parts[-4]
        view_id = view_dir.name
        dest = out_path / f"{scene_path.name}_{room_id}_v{view_id}"
        line_file = dest / "lines.jpg"
        sem_file = dest / "semantic.jpg"
        if line_file.is_file() and sem_file.is_file():
            skipped += 1
            continue
        view = Structured3DView(scene_path.name, room_id, view_id, view_dir)
        bgr = view.load_rgb(rawlight=True, bgr=True)
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(line_file), _line_overlay(bgr, anno, view), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(sem_file), _semantic_overlay(bgr, view), [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Structured3D_perspective_full_00/Structured3D"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Structured3D_perspective_full_00/vis"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    scenes = sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[*] {len(scenes)} scenes -> {args.out}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    total_written = 0
    total_skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_scene, str(scene), str(args.out)): scene.name
            for scene in scenes
        }
        for done, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                written, skipped = future.result()
            except Exception as exc:
                print(f"[!] {name}: {exc}", flush=True)
                continue
            total_written += written
            total_skipped += skipped
            if done % 10 == 0 or done == len(scenes):
                print(
                    f"[{done}/{len(scenes)}] wrote {total_written}, skipped {total_skipped}",
                    flush=True,
                )
    print(f"[+] done. new views={total_written}, already existed={total_skipped}", flush=True)


if __name__ == "__main__":
    main()
