#!/usr/bin/env python3
"""
Unified Pipeline Orchestrator.

Enables end-to-end multi-modal lifecycle execution from a single YAML configuration:
  1. Data Generation: python run.py --config configs/superpoint.yaml --step generate
  2. Training:        python run.py --config configs/superpoint.yaml --step train
  3. Evaluation:      python run.py --config configs/superpoint.yaml --step eval
  4. Full Pipeline:   python run.py --config configs/superpoint.yaml --step all

Any additional command-line flags (e.g., --epochs 20, --device cuda) are dynamically
forwarded to override YAML defaults for maximum flexibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config, get_model_config_path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Unified Pipeline Orchestrator: Generate, Train, and Evaluate from YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., configs/superpoint.yaml)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name alias (e.g., 'SuperPoint', 'xfeat_object'). Auto-locates config if --config not passed.",
    )
    parser.add_argument(
        "--step", "-s",
        type=str,
        default="all",
        choices=["all", "generate", "train", "eval"],
        help="Pipeline stage to execute: 'generate', 'train', 'eval', or 'all' (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parsed configurations and commands without executing",
    )
    return parser.parse_known_args()


def run_pipeline() -> None:
    args, extra_args = parse_args()

    # 1. Resolve configuration path
    if args.config:
        cfg_path = Path(args.config)
    elif args.model:
        cfg_path = get_model_config_path(args.model)
    else:
        print("[!] Error: Either --config <path.yaml> or --model <name> must be specified.")
        sys.exit(1)

    cfg = load_config(cfg_path)
    model_name = cfg.get("model", {}).get("model_type", "Unknown")
    print("=" * 75)
    print(f"[*] Pipeline Orchestrator: Model '{model_name}' | Config: {cfg_path}")
    print(f"[*] Target Step: '{args.step}' | Extra Override Flags: {extra_args}")
    print("=" * 75)

    if args.dry_run:
        print("\n[+] Dry-run mode enabled. Parsed configuration sections:")
        for section in ("model", "data_generation", "train", "eval"):
            print(f"  [{section}]: {cfg.get(section, {})}")
        return

    steps_to_run = ["generate", "train", "eval"] if args.step == "all" else [args.step]
    t0 = time.time()

    # Step 1: Data Generation
    if "generate" in steps_to_run:
        print("\n" + "=" * 75)
        print(">>> [Stage 1: Dataset Generation]")
        print("=" * 75)
        from generate_dataset import main as gen_main

        gen_argv = ["--config", str(cfg_path), *extra_args]
        gen_main(gen_argv)

    # Step 2: Training
    if "train" in steps_to_run:
        print("\n" + "=" * 75)
        print(">>> [Stage 2: Model Training]")
        print("=" * 75)
        from train import main as train_main

        train_argv = ["--config", str(cfg_path), *extra_args]
        # Temporarily adapt sys.argv so train_applicable_arguments works seamlessly
        old_argv = sys.argv
        sys.argv = ["train.py", *train_argv]
        try:
            train_main()
        finally:
            sys.argv = old_argv

    # Step 3: Evaluation
    if "eval" in steps_to_run:
        print("\n" + "=" * 75)
        print(">>> [Stage 3: Evaluation & Benchmarking]")
        print("=" * 75)
        from eval import main as eval_main

        eval_argv = ["--config", str(cfg_path), *extra_args]
        eval_main(eval_argv)

    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print(f"[✓] Pipeline '{args.step}' successfully completed in {elapsed:.1f}s.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
