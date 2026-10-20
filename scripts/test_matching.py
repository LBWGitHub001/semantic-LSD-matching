"""
Pure Line Matching Evaluation Script (Zero Line Detection).

Evaluates the trained Line-Transformer matcher strictly on user-provided/ground-truth 2D lines.
No detector (LSD) is run. The inputs are purely (image0, image1, lines0, lines1).

Supported Modes:
  - Mode A (pure_permuted): True mutual line correspondences, with View 1 randomly shuffled.
  - Mode B (with_distractors): True mutual lines + single-view distractor lines (tests dustbin rejection).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import LineMatchPipeline, get_distinct_colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pure Line Matching on Ground Truth Lines")
    parser.add_argument(
        "--pair",
        type=str,
        default="all",
        help="Test pair folder (e.g. 'pair1_small_rotation', 'pair2_medium_rotation', or 'all')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Evaluation mode: 'A' (pure permuted), 'B' (with distractors), or 'both'",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "checkpoints" / "linefeat" / "best_model.pth",
        help="Trained checkpoint path",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.70,
        help="Similarity score threshold (default: 0.70)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="mask",
        choices=["mask", "linefeat"],
        help="Model architecture: 'mask' (MaskLineFeatMatchNet, default) or 'linefeat' (LineMatcherNet)",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=ROOT / "data" / "test_samples",
        help="Root directory of curated test samples",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "pure_matching",
        help="Output directory for evaluation visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def render_gt_panel(
    img0: np.ndarray,
    img1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    gt_matches: list[dict],
) -> np.ndarray:
    """Renders Ground Truth match panel with distinct colors per pair."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.hstack([img0.copy(), img1.copy()])
    colors = get_distinct_colors(len(gt_matches))

    overlay = canvas.copy()
    for k, m in enumerate(gt_matches):
        i, j = m["idx0"], m["idx1"]
        col = colors[k % len(colors)]

        # Line in img0
        x1, y1, x2, y2 = [int(round(v)) for v in lines0[i]]
        cv2.line(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 4, (255, 255, 255), -1, cv2.LINE_AA)
        mid0 = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Line in img1 (offset by w0)
        x3, y3, x4, y4 = [int(round(v)) for v in lines1[j]]
        cv2.line(canvas, (x3 + w0, y3), (x4 + w0, y4), col, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x3 + w0, y3), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x4 + w0, y4), 4, (255, 255, 255), -1, cv2.LINE_AA)
        mid1 = (int((x3 + x4) / 2) + w0, int((y3 + y4) / 2))

        # Connecting dashed-like line
        cv2.line(overlay, mid0, mid1, col, 1, cv2.LINE_AA)

    canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

    # Header banner
    header = np.zeros((48, canvas.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    cv2.putText(
        header,
        f"[GROUND TRUTH] Mutual True Line Correspondences: {len(gt_matches)} pairs",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, canvas])


def render_pred_panel(
    img0: np.ndarray,
    img1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    pred_matches: list[tuple[int, int, float]],
    gt_pairs_set: set[tuple[int, int]],
    metrics: dict[str, float],
) -> np.ndarray:
    """Renders Model Prediction panel with Green (Correct) / Red (Wrong) connections."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    canvas = np.hstack([img0.copy(), img1.copy()])

    overlay = canvas.copy()
    correct_color = (0, 220, 0)     # Green for True Positives
    wrong_color = (0, 0, 240)       # Red for False Positives

    for idx0, idx1, score in pred_matches:
        is_correct = (idx0, idx1) in gt_pairs_set
        col = correct_color if is_correct else wrong_color

        # Line in img0
        x1, y1, x2, y2 = [int(round(v)) for v in lines0[idx0]]
        cv2.line(canvas, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 4, (255, 255, 255), -1, cv2.LINE_AA)
        mid0 = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Line in img1 (offset by w0)
        x3, y3, x4, y4 = [int(round(v)) for v in lines1[idx1]]
        cv2.line(canvas, (x3 + w0, y3), (x4 + w0, y4), col, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x3 + w0, y3), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x4 + w0, y4), 4, (255, 255, 255), -1, cv2.LINE_AA)
        mid1 = (int((x3 + x4) / 2) + w0, int((y3 + y4) / 2))

        # Connecting line
        cv2.line(overlay, mid0, mid1, col, 2 if not is_correct else 1, cv2.LINE_AA)

    canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

    # Header banner
    header = np.zeros((48, canvas.shape[1], 3), dtype=np.uint8)
    header[:] = (40, 40, 40)
    title = (
        f"[MODEL PREDICTIONS] Matches: {len(pred_matches)} | "
        f"Correct: {metrics['correct']} | "
        f"Precision: {metrics['precision']:.1f}% | "
        f"Recall: {metrics['recall']:.1f}% (GREEN = Correct, RED = Mismatch)"
    )
    cv2.putText(
        header,
        title,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0) if metrics["precision"] >= 50 else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, canvas])


def evaluate_single_mode(
    pipeline: LineMatchPipeline,
    pair_dir: Path,
    mode_name: str,
    mode_dir_name: str,
    thresh: float,
    out_dir: Path,
) -> dict:
    mode_dir = pair_dir / mode_dir_name
    if not mode_dir.exists():
        return {}

    img0_path = pair_dir / "image0.png"
    img1_path = pair_dir / "image1.png"

    with open(mode_dir / "lines0.json", "r", encoding="utf-8") as f:
        lines0 = np.array(json.load(f), dtype=np.float32)
    with open(mode_dir / "lines1.json", "r", encoding="utf-8") as f:
        lines1 = np.array(json.load(f), dtype=np.float32)
    with open(mode_dir / "gt_matches.json", "r", encoding="utf-8") as f:
        gt_matches = json.load(f)

    gt_pairs_set = {(m["idx0"], m["idx1"]) for m in gt_matches}
    M = len(gt_matches)

    t0 = time.perf_counter()
    # PURE MATCHING: lines0 and lines1 passed directly, NO LSD RUN!
    result = pipeline.match(
        img0_path,
        img1_path,
        lines0=lines0,
        lines1=lines1,
        threshold=thresh,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    pred_matches = result.matches
    K = len(pred_matches)

    correct = sum(1 for (i, j, s) in pred_matches if (i, j) in gt_pairs_set)
    precision = (correct / max(K, 1)) * 100.0
    recall = (correct / max(M, 1)) * 100.0

    metrics = {
        "pair": pair_dir.name,
        "mode": mode_name,
        "N0": len(lines0),
        "N1": len(lines1),
        "gt_matches": M,
        "pred_matches": K,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "time_ms": elapsed_ms,
    }

    # Render Visualizations
    bgr0 = cv2.imread(str(img0_path))
    bgr1 = cv2.imread(str(img1_path))

    panel_gt = render_gt_panel(bgr0, bgr1, lines0, lines1, gt_matches)
    panel_pred = render_pred_panel(bgr0, bgr1, lines0, lines1, pred_matches, gt_pairs_set, metrics)

    # Combine Top (GT) and Bottom (Prediction)
    separator = np.zeros((16, panel_gt.shape[1], 3), dtype=np.uint8)
    separator[:] = (20, 20, 20)
    full_vis = np.vstack([panel_gt, separator, panel_pred])

    out_file = out_dir / f"{pair_dir.name}_{mode_dir_name}_eval.png"
    cv2.imwrite(str(out_file), full_vis)
    metrics["vis_path"] = str(out_file)

    return metrics


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("        Pure Line Matching Evaluation (Ground Truth Lines, Zero Detection)")
    print("=" * 85)
    print(f"  Weights:    {args.weights}")
    print(f"  Model Type: {args.model_type} ({'MaskLineFeatMatchNet' if args.model_type == 'mask' else 'LineMatcherNet'})")
    print(f"  Device:     {args.device}")
    print(f"  Threshold:  {args.thresh}")
    print(f"  Output Dir: {args.out_dir}")
    print("=" * 85)

    pipeline = LineMatchPipeline(
        weights=args.weights,
        device=args.device,
        similarity_threshold=args.thresh,
        model_type=args.model_type,
    )

    if args.pair == "all":
        pair_dirs = sorted([d for d in args.samples_dir.iterdir() if d.is_dir() and (d / "image0.png").exists()])
    else:
        target = args.samples_dir / args.pair
        if not target.exists():
            print(f"[!] Specified pair not found: {target}")
            return
        pair_dirs = [target]

    modes_to_run = []
    if args.mode in ("A", "both"):
        modes_to_run.append(("Mode A (Pure Permuted)", "mode_a_pure_permuted"))
    if args.mode in ("B", "both"):
        modes_to_run.append(("Mode B (With Distractors)", "mode_b_with_distractors"))

    all_metrics = []

    for pdir in pair_dirs:
        for mode_title, mode_dirname in modes_to_run:
            m = evaluate_single_mode(pipeline, pdir, mode_title, mode_dirname, args.thresh, args.out_dir)
            if m:
                all_metrics.append(m)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Sample Pair':<24} | {'Mode':<18} | {'Lines(0/1)':<10} | {'GT':<4} | {'Pred':<5} | {'Corr':<5} | {'Prec':<7} | {'Recall':<7}")
    print("-" * 90)
    for r in all_metrics:
        print(
            f"{r['pair']:<24} | {r['mode']:<18} | {r['N0']:>3}/{r['N1']:<6} | {r['gt_matches']:<4} | "
            f"{r['pred_matches']:<5} | {r['correct']:<5} | {r['precision']:>5.1f}% | {r['recall']:>5.1f}%"
        )
    print("=" * 90)
    print(f"\n[*] All side-by-side Ground Truth vs Prediction visual results saved to:\n    {args.out_dir.resolve()}/\n")


if __name__ == "__main__":
    main()
