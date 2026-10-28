"""
Command-line interface to render network architecture diagrams for all models.
Usage:
    python scripts/generate_architecture_diagram.py [--model <model_name>] [--out <output_path>] [--all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from visualization.architecture_visualizer import render_all_models, render_model_architecture


def main() -> None:
    parser = argparse.ArgumentParser(description="Render publication-quality neural network architecture diagrams.")
    parser.add_argument("--model", type=str, default="xfeat_line_matcher",
                        choices=["xfeat_line_matcher", "xfeat_object", "mask_line", "line_transformer"],
                        help="Model architecture to visualize")
    parser.add_argument("--out", type=str, default=None, help="Target PNG file path")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI resolution (default: 180)")
    parser.add_argument("--all", action="store_true", help="Render all supported models in batch")

    args = parser.parse_args()

    if args.all:
        print("[*] Rendering architecture diagrams for ALL supported models...")
        results = render_all_models()
        for model_name, path in results.items():
            print(f"  [✓] {model_name:<20} -> {path}")
    else:
        print(f"[*] Rendering architecture diagram for [{args.model}]...")
        path = render_model_architecture(args.model, out_path=args.out, dpi=args.dpi)
        print(f"  [✓] Successfully exported to: {path}")


if __name__ == "__main__":
    main()
