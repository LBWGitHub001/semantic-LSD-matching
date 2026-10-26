#!/usr/bin/env python3
"""
Experiment Reviewer & Post-Mortem Analytics CLI.

Commands:
  # 1. List all historical training runs with benchmark metrics
  python review_runs.py --list

  # 2. Inspect a specific run in detail (with diagnostics & metrics)
  python review_runs.py --run checkpoints/xfeat_line_matcher/20261023_233349

  # 3. Compare multiple runs side-by-side
  python review_runs.py --compare checkpoints/xfeat_object_matcher/20261022_225315 checkpoints/xfeat_line_matcher/20261023_233349

  # 4. Generate/refresh curves for a run
  python review_runs.py --plot checkpoints/xfeat_line_matcher/20261023_233349
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_all_runs(ckpt_root: Path | str = "checkpoints") -> list[dict[str, Any]]:
    """Recursively scans checkpoints directory for runs with config.json."""
    p_root = Path(ckpt_root).resolve()
    if not p_root.is_dir():
        return []

    runs = []
    config_files = sorted(list(p_root.glob("**/config.json")), key=lambda p: p.stat().st_mtime, reverse=True)

    for cfg_path in config_files:
        run_dir = cfg_path.parent
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue

        exp_info = cfg.get("experiment_info", {})
        model_info = cfg.get("model", {})
        hparams = cfg.get("hyperparameters", {})
        best_val = cfg.get("best_validation") or {}
        test_bench = cfg.get("test_benchmark") or {}
        history = cfg.get("history") or []

        # If history not in config, check metrics.csv
        csv_path = run_dir / "metrics.csv"
        if not history and csv_path.is_file():
            try:
                with open(csv_path, "r", encoding="utf-8") as cf:
                    reader = csv.DictReader(cf)
                    for row in reader:
                        history.append({
                            "epoch": int(row["epoch"]),
                            "train_loss": float(row["train_loss"]) if row.get("train_loss") else None,
                            "val_f1": float(row["val_f1"]) if row.get("val_f1") else None,
                            "val_precision": float(row["val_precision"]) if row.get("val_precision") else None,
                            "val_recall": float(row["val_recall"]) if row.get("val_recall") else None,
                            "dustbin_precision": float(row["dustbin_precision"]) if row.get("dustbin_precision") else None,
                            "dustbin_recall": float(row["dustbin_recall"]) if row.get("dustbin_recall") else None,
                            "dustbin_accuracy": float(row["dustbin_accuracy"]) if row.get("dustbin_accuracy") else None,
                            "cross_cat_fp": int(row["cross_cat_fp"]) if row.get("cross_cat_fp") else None,
                            "lr": float(row["lr"]) if row.get("lr") else None,
                            "epoch_time_sec": float(row["epoch_time_sec"]) if row.get("epoch_time_sec") else None,
                            "is_best": row.get("is_best", "FALSE").upper() == "TRUE",
                        })
            except Exception:
                pass

        raw_test = test_bench.get("raw_visual") if isinstance(test_bench, dict) else None
        sem_test = test_bench.get("semantic_constrained") if isinstance(test_bench, dict) else None

        # Check status
        has_best = (run_dir / "best_model.pth").is_file()
        is_completed = (test_bench is not None) or (exp_info.get("end_time") is not None)

        runs.append({
            "run_name": exp_info.get("run_name", run_dir.name),
            "model_type": model_info.get("model_type", run_dir.parent.name),
            "start_time": exp_info.get("start_time", "N/A"),
            "end_time": exp_info.get("end_time", None),
            "epochs": hparams.get("epochs", len(history) or "N/A"),
            "best_epoch": best_val.get("epoch", "-"),
            "best_val_f1": best_val.get("f1", None),
            "best_val_prec": best_val.get("precision", None),
            "best_val_rec": best_val.get("recall", None),
            "dustbin_prec": best_val.get("dustbin_precision", None),
            "dustbin_rec": best_val.get("dustbin_recall", None),
            "test_raw_f1": raw_test.get("f1") if raw_test else None,
            "test_sem_f1": sem_test.get("f1") if sem_test else None,
            "status": "Done" if is_completed else ("Active" if has_best else "Incomplete"),
            "run_dir": run_dir,
            "config": cfg,
            "history": history,
        })

    return runs


def print_dashboard(runs: list[dict[str, Any]]) -> None:
    """Prints a structured ASCII dashboard table of all training runs."""
    if not runs:
        print("\n[!] No training runs with config.json found in checkpoints directory.")
        return

    print("\n" + "=" * 118)
    print("                              EXPERIMENT RUNS DASHBOARD & HISTORICAL REVIEW                              ")
    print("=" * 118)
    header = (
        f" {'Run Name':<16} | {'Model':<18} | {'Start Time':<16} | {'Epochs':<6} | "
        f"{'Best Ep':<7} | {'Val F1':<7} | {'Test Raw':<8} | {'Test Sem':<8} | {'Status':<6}"
    )
    print(header)
    print("-" * 118)

    for r in runs:
        r_name = r["run_name"][:16]
        m_type = r["model_type"][:18]
        s_time = r["start_time"][:16] if r["start_time"] != "N/A" else "N/A"
        eps = str(r["epochs"])[:6]
        b_ep = str(r["best_epoch"])[:7]
        v_f1 = f"{r['best_val_f1']:.1f}%" if r["best_val_f1"] is not None else "-"
        t_raw = f"{r['test_raw_f1']:.1f}%" if r["test_raw_f1"] is not None else "-"
        t_sem = f"{r['test_sem_f1']:.1f}%" if r["test_sem_f1"] is not None else "-"
        status = r["status"]
        if status == "Done" and r["test_raw_f1"] and r["test_raw_f1"] >= 74.0:
            status = "Done ★"

        print(
            f" {r_name:<16} | {m_type:<18} | {s_time:<16} | {eps:<6} | "
            f"{b_ep:<7} | {v_f1:<7} | {t_raw:<8} | {t_sem:<8} | {status:<6}"
        )

    print("=" * 118)
    print(f"Total Runs Found: {len(runs)} | Use `python review_runs.py --run <run_name>` for deep-dive analysis.\n")


def print_run_details(run: dict[str, Any]) -> None:
    """Prints comprehensive details, diagnostics, and metrics table for a single run."""
    r_dir = run["run_dir"]
    cfg = run["config"]
    hparams = cfg.get("hyperparameters", {})
    exp_info = cfg.get("experiment_info", {})
    best_val = cfg.get("best_validation") or {}
    test_bench = cfg.get("test_benchmark") or {}

    print("\n" + "=" * 80)
    print(f"  RUN REVIEW: {run['run_name']} ({run['model_type']})")
    print("=" * 80)
    print(f"  Directory:    {r_dir}")
    print(f"  Command:      {exp_info.get('command', 'N/A')}")
    print(f"  Start Time:   {exp_info.get('start_time', 'N/A')} ~ {exp_info.get('end_time', 'Active / Interrupted')}")
    print(f"  Device:       {exp_info.get('device_name', 'GPU')} (AMP FP16: {hparams.get('amp', False)})")
    print(f"  Git Commit:   {exp_info.get('git_commit', 'HEAD')}")
    print("-" * 80)
    print("  HYPERPARAMETERS:")
    print(f"    • Learning Rate     : {hparams.get('learning_rate', 'N/A')}")
    print(f"    • Total Epochs      : {hparams.get('epochs', 'N/A')}")
    print(f"    • Dustbin Weight    : {hparams.get('dustbin_weight', 'N/A')}")
    print(f"    • Motion Loss Weight: {hparams.get('motion_loss_weight', 'N/A')} (Margin: {hparams.get('motion_margin', 'N/A')})")
    print(f"    • Min Area Ratio    : {hparams.get('min_area_ratio', 'N/A')}")
    print("-" * 80)
    print("  VALIDATION PERFORMANCE (Best Epoch):")
    if best_val:
        print(f"    • Best Epoch        : Epoch {best_val.get('epoch', 'N/A')}")
        print(f"    • Val F1-Score      : {best_val.get('f1', 'N/A')}%")
        print(f"    • Precision / Recall: {best_val.get('precision', 'N/A')}% / {best_val.get('recall', 'N/A')}%")
        print(f"    • Dustbin P / R / Acc: P={best_val.get('dustbin_precision', 'N/A')}% / R={best_val.get('dustbin_recall', 'N/A')}% / Acc={best_val.get('dustbin_accuracy', 'N/A')}%")
        print(f"    • Cross-Category FP : {best_val.get('cross_cat_fp', 'N/A')}")
    else:
        print("    [!] No validation metrics recorded.")

    print("-" * 80)
    print("  TEST BENCHMARK RESULTS:")
    if test_bench and isinstance(test_bench, dict):
        raw = test_bench.get("raw_visual", {})
        sem = test_bench.get("semantic_constrained", {})
        if raw:
            print(f"    • [Raw Visual]      : Precision={raw.get('precision', 0):.2f}%, Recall={raw.get('recall', 0):.2f}%, F1={raw.get('f1', 0):.2f}%")
        if sem:
            print(f"    • [Semantic Constrained]: Precision={sem.get('precision', 0):.2f}%, Recall={sem.get('recall', 0):.2f}%, F1={sem.get('f1', 0):.2f}%")
    else:
        print("    [!] Test benchmark not yet executed or run was interrupted.")

    # Check for report.md and curves.png
    report_md = r_dir / "report.md"
    curves_png = r_dir / "curves.png"
    print("-" * 80)
    print("  ARTIFACTS ON DISK:")
    print(f"    • Best Model : {'[✓] Found' if (r_dir / 'best_model.pth').is_file() else '[✗] Missing'}")
    print(f"    • Report     : {'[✓] ' + str(report_md) if report_md.is_file() else '[✗] Not yet generated'}")
    print(f"    • Curves Plot: {'[✓] ' + str(curves_png) if curves_png.is_file() else '[✗] Not yet rendered'}")

    # History Table
    history = run.get("history", [])
    if history:
        print("\n  EPOCH METRICS TRAJECTORY:")
        print("  " + "-" * 76)
        print(f"  {'Ep':<4} | {'Loss':<7} | {'Val F1':<8} | {'Prec':<7} | {'Rec':<7} | {'Dustbin P/R':<15} | {'Best':<4}")
        print("  " + "-" * 76)
        for h in history:
            star = "★" if h.get("is_best") else ""
            f1_s = f"{h['val_f1']:.1f}%" if h.get("val_f1") is not None else "-"
            p_s = f"{h['val_precision']:.1f}%" if h.get("val_precision") is not None else "-"
            r_s = f"{h['val_recall']:.1f}%" if h.get("val_recall") is not None else "-"
            db_s = f"{h.get('dustbin_precision', 0):.1f}% / {h.get('dustbin_recall', 0):.1f}%"
            print(f"  {h['epoch']:<4d} | {h['train_loss']:<7.4f} | {f1_s:<8} | {p_s:<7} | {r_s:<7} | {db_s:<15} | {star:<4}")
        print("  " + "-" * 76)

    print("=" * 80 + "\n")


def compare_runs(run_paths: list[str]) -> None:
    """Performs side-by-side comparative analysis between multiple runs."""
    all_runs = find_all_runs("checkpoints")
    selected: list[dict[str, Any]] = []

    for query in run_paths:
        q_path = Path(query).resolve()
        matched = None
        for r in all_runs:
            if r["run_dir"] == q_path or r["run_name"] == query or query in str(r["run_dir"]):
                matched = r
                break
        if matched:
            selected.append(matched)
        else:
            print(f"[!] Warning: Run '{query}' not found.")

    if len(selected) < 2:
        print("[!] Please provide at least 2 valid runs to compare.")
        return

    print("\n" + "=" * 90)
    print("                           SIDE-BY-SIDE EXPERIMENT COMPARISON                           ")
    print("=" * 90)

    col_w = max(24, max(len(r["run_name"]) for r in selected) + 2)
    names_row = f" {'Feature / Metric':<24} | " + " | ".join(f"{r['run_name']:<{col_w}}" for r in selected)
    print(names_row)
    print("-" * 90)

    # 1. Model & Parameters
    fields = [
        ("Model Architecture", lambda r: r["model_type"]),
        ("Total Epochs", lambda r: str(r["config"].get("hyperparameters", {}).get("epochs", "-"))),
        ("Learning Rate", lambda r: str(r["config"].get("hyperparameters", {}).get("learning_rate", "-"))),
        ("Dustbin Weight", lambda r: str(r["config"].get("hyperparameters", {}).get("dustbin_weight", "-"))),
        ("Motion Weight", lambda r: str(r["config"].get("hyperparameters", {}).get("motion_loss_weight", "-"))),
        ("Min Area Ratio", lambda r: str(r["config"].get("hyperparameters", {}).get("min_area_ratio", "-"))),
        ("Dataset", lambda r: Path(r["config"].get("dataset", {}).get("dataset_dir", "N/A")).name),
    ]

    for label, fn in fields:
        row = f" {label:<24} | " + " | ".join(f"{fn(r):<{col_w}}" for r in selected)
        print(row)

    print("-" * 90)
    print(" PERFORMANCE OUTCOMES:")
    perf_fields = [
        ("Best Val Epoch", lambda r: str(r.get("best_epoch", "-"))),
        ("Best Val F1", lambda r: f"{r['best_val_f1']:.2f}%" if r.get("best_val_f1") is not None else "-"),
        ("Best Val Precision", lambda r: f"{r['best_val_prec']:.2f}%" if r.get("best_val_prec") is not None else "-"),
        ("Best Val Recall", lambda r: f"{r['best_val_rec']:.2f}%" if r.get("best_val_rec") is not None else "-"),
        ("Dustbin Recall", lambda r: f"{r['dustbin_rec']:.2f}%" if r.get("dustbin_rec") is not None else "-"),
        ("Test Raw F1", lambda r: f"{r['test_raw_f1']:.2f}%" if r.get("test_raw_f1") is not None else "-"),
        ("Test Semantic F1", lambda r: f"{r['test_sem_f1']:.2f}%" if r.get("test_sem_f1") is not None else "-"),
    ]

    for label, fn in perf_fields:
        vals = [r.get("test_raw_f1") if "Test Raw" in label else r.get("best_val_f1") for r in selected]
        row = f" {label:<24} | " + " | ".join(f"{fn(r):<{col_w}}" for r in selected)
        print(row)

    print("=" * 90 + "\n")

    # Plot multi-run comparison curves if history is available
    histories_available = all(len(r.get("history", [])) >= 2 for r in selected)
    if histories_available:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor("#FAFAFA")

            for idx, r in enumerate(selected):
                hist = r["history"]
                eps = [h["epoch"] for h in hist]
                losses = [h["train_loss"] for h in hist]
                f1s = [h["val_f1"] for h in hist if h["val_f1"] is not None]
                f1_eps = [h["epoch"] for h in hist if h["val_f1"] is not None]

                label = f"{r['run_name']} ({r['model_type']})"
                ax1.plot(eps, losses, marker="o", markersize=3, label=label)
                if f1s:
                    ax2.plot(f1_eps, f1s, marker="s", markersize=3, label=label)

            ax1.set_title("Training Loss Comparison", fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend()

            ax2.set_title("Validation F1-Score Comparison", fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Val F1 (%)")
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()

            out_cmp = Path("outputs") / f"compare_{selected[0]['run_name']}_vs_{selected[1]['run_name']}.png"
            out_cmp.parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(str(out_cmp), dpi=150)
            plt.close(fig)
            print(f"[✓] Multi-run comparison curve saved to: {out_cmp}\n")
        except Exception as e:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Experiment Reviewer & Post-Mortem Diagnostics CLI")
    parser.add_argument("--list", "-l", action="store_true", help="List all historical runs in dashboard table")
    parser.add_argument("--run", "-r", type=str, default=None, help="Inspect a specific run by name or directory")
    parser.add_argument("--compare", "-c", nargs="+", help="Compare two or more runs side-by-side")
    parser.add_argument("--plot", "-p", type=str, default=None, help="Render or refresh curves.png for a specific run")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Checkpoints root directory")
    args = parser.parse_args()

    all_runs = find_all_runs(args.ckpt_dir)

    if args.compare:
        compare_runs(args.compare)
    elif args.run:
        target = args.run
        matched = None
        for r in all_runs:
            if r["run_name"] == target or target in str(r["run_dir"]):
                matched = r
                break
        if matched:
            print_run_details(matched)
        else:
            print(f"[!] Error: Run '{target}' not found in {args.ckpt_dir}.")
    elif args.plot:
        target = args.plot
        matched = None
        for r in all_runs:
            if r["run_name"] == target or target in str(r["run_dir"]):
                matched = r
                break
        if matched:
            from tools.experiment_tracker import ExperimentLogger
            logger = ExperimentLogger(matched["run_dir"], matched["config"], enable_stdout_tee=False)
            logger.history = matched.get("history", [])
            out_p = logger.plot_curves()
            logger.generate_post_mortem_report()
            print(f"[✓] Curves rendered to: {out_p}")
            print(f"[✓] Report generated to: {logger.report_md_path}")
        else:
            print(f"[!] Error: Run '{target}' not found.")
    else:
        # Default behavior: show dashboard
        print_dashboard(all_runs)


if __name__ == "__main__":
    main()
