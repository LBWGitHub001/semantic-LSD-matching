"""
Batch Evaluation & Demonstration on Curated Test Samples.

Runs the trained Line-Transformer matcher on all 4 curated test pairs in data/test_samples/,
prints an evaluation summary table, and generates match visualization images in outputs/test_samples/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import LineMatchPipeline


def main() -> None:
    print("=" * 80)
    print("       LineFeat Batch Test Runner on Curated Test Pairs")
    print("=" * 80)

    weights_path = ROOT / "checkpoints" / "linefeat" / "best_model.pth"
    if not weights_path.exists():
        print(f"[!] Error: Model checkpoint not found at {weights_path}")
        return

    pipeline = LineMatchPipeline(
        weights=weights_path,
        similarity_threshold=0.85,
    )
    print(f"[*] Pipeline initialized successfully on device: {pipeline.device}")

    samples_dir = ROOT / "data" / "test_samples"
    out_dir = ROOT / "outputs" / "test_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir() and (d / "image0.png").exists()])
    if not test_dirs:
        print(f"[!] No test folders found in {samples_dir}")
        return

    results_table = []

    for d in test_dirs:
        info_file = d / "info.json"
        meta = {}
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        category = meta.get("category", d.name)
        rot_deg = meta.get("rotation_deg", 0.0)
        baseline_m = meta.get("baseline_m", 0.0)

        img0 = d / "image0.png"
        img1 = d / "image1.png"

        t0 = time.perf_counter()
        res = pipeline.match(img0, img1)
        elapsed = (time.perf_counter() - t0) * 1000

        vis_path = out_dir / f"{d.name}_matches.png"
        res.save_visualization(vis_path, max_draw=40)

        mean_conf = float(res.match_scores.mean()) if len(res) > 0 else 0.0
        results_table.append({
            "folder": d.name,
            "category": category,
            "rotation": f"{rot_deg:.1f}°",
            "baseline": f"{baseline_m:.2f}m",
            "lines0": len(res.lines0),
            "lines1": len(res.lines1),
            "matches": len(res),
            "mean_conf": f"{mean_conf:.3f}",
            "time_ms": f"{elapsed:.1f}ms",
            "vis_file": vis_path.name
        })

    # Print markdown table
    print("\n" + "=" * 80)
    print(f"{'Folder':<24} | {'Rot':<7} | {'Base':<6} | {'Lines(0/1)':<12} | {'Matches':<8} | {'Conf':<6} | {'Time':<8}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['folder']:<24} | {r['rotation']:<7} | {r['baseline']:<6} | {r['lines0']:>4}/{r['lines1']:<6} | {r['matches']:<8} | {r['mean_conf']:<6} | {r['time_ms']:<8}")
    print("=" * 80)
    print(f"\n[*] All visualization images saved to: {out_dir.resolve()}/\n")


if __name__ == "__main__":
    main()
