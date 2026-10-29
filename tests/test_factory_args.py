"""Unused command-line flags are reported instead of ignored silently."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factory_args import warn_unused_arguments
from generate_dataset import applicable_arguments, build_parser
from train import parse_args, train_applicable_arguments


class TestUnusedFactoryArguments(unittest.TestCase):
    def test_homo_matcher_ignores_synthetic_and_cad_flags(self) -> None:
        parser = build_parser()
        ignored = warn_unused_arguments(
            parser,
            applicable_arguments("FCN_field"),
            "FCN_field",
            ["--task", "FCN_field", "--num-scenes", "10", "--min-matches", "3", "--num-workers", "20"],
        )
        self.assertEqual(ignored, ["--num-scenes", "--min-matches"])

    def test_structured3d_ignores_homo_only_flags(self) -> None:
        parser = build_parser()
        ignored = warn_unused_arguments(
            parser,
            applicable_arguments("structured3d"),
            "structured3d",
            ["--task", "structured3d", "--max-pairs", "4", "--weight", "w.tar", "--num-workers", "8"],
        )
        self.assertEqual(ignored, ["--max-pairs", "--weight"])

    def test_train_homo_matcher_ignores_object_loss_flags(self) -> None:
        old = sys.argv
        sys.argv = ["train.py", "--model", "FCN_field"]
        try:
            _args, parser = parse_args()
        finally:
            sys.argv = old
        ignored = warn_unused_arguments(
            parser,
            train_applicable_arguments("FCN_field"),
            "FCN_field",
            ["--model", "FCN_field", "--dustbin-weight", "0.3", "--cycle-weight", "1"],
        )
        self.assertEqual(ignored, ["--dustbin-weight"])


if __name__ == "__main__":
    unittest.main()
