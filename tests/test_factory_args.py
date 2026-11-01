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
    def test_planar_letters_ignores_cad_flags(self) -> None:
        parser = build_parser()
        ignored = warn_unused_arguments(
            parser,
            applicable_arguments("planar_letters"),
            "planar_letters",
            ["--task", "planar_letters", "--s3d-dir", "foo", "--min-matches", "3", "--num-scenes", "10"],
        )
        self.assertEqual(ignored, ["--s3d-dir", "--min-matches"])

    def test_structured3d_ignores_synthetic_flags(self) -> None:
        parser = build_parser()
        ignored = warn_unused_arguments(
            parser,
            applicable_arguments("structured3d"),
            "structured3d",
            ["--task", "structured3d", "--num-scenes", "10", "--random-colors", "--num-workers", "8"],
        )
        self.assertEqual(ignored, ["--num-scenes", "--random-colors"])

    def test_train_superpoint_ignores_object_loss_flags(self) -> None:
        _args, parser = parse_args(["--model", "SuperPoint"])
        ignored = warn_unused_arguments(
            parser,
            train_applicable_arguments("SuperPoint"),
            "SuperPoint",
            ["--model", "SuperPoint", "--dustbin-weight", "0.3", "--motion-weight", "0.5"],
        )
        self.assertEqual(ignored, ["--dustbin-weight", "--motion-weight"])


if __name__ == "__main__":
    unittest.main()
