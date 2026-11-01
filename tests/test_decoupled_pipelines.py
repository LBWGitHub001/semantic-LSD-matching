#!/usr/bin/env python3
"""
Unit and Integration Tests for Decoupled Model Architectures:
  1. Verifies that all 3 active model subpackages (superpoint, object_matcher, mask_line_matcher)
     have self-contained train.py and eval.py
  2. Verifies that each model's train() and evaluate() can be called programmatically
  3. Verifies that root train.py dispatches correctly to model subpackages
  4. Verifies that deleted models (FCN_field, line_transformer, semantic_line_matcher) are gone
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import TRAIN_DISPATCH, parse_args, train_applicable_arguments


class TestDecoupledModelArchitecture(unittest.TestCase):
    def test_active_model_folders_have_train_and_eval(self) -> None:
        """Active model packages must contain dedicated, self-contained train.py and eval.py."""
        model_dirs = [
            "superpoint",
            "object_matcher",
            "mask_line_matcher",
        ]
        for mdir in model_dirs:
            base = ROOT / "models" / mdir
            train_py = base / "train.py"
            eval_py = base / "eval.py"
            init_py = base / "__init__.py"

            self.assertTrue(base.is_dir(), f"Model directory missing: {base}")
            self.assertTrue(train_py.is_file(), f"Missing train.py in: {base}")
            self.assertTrue(eval_py.is_file(), f"Missing eval.py in: {base}")
            self.assertTrue(init_py.is_file(), f"Missing __init__.py in: {base}")

            # Verify callable exports
            pkg = importlib.import_module(f"models.{mdir}")
            self.assertTrue(hasattr(pkg, "train"), f"Package models.{mdir} does not export 'train'")
            self.assertTrue(hasattr(pkg, "evaluate"), f"Package models.{mdir} does not export 'evaluate'")
            self.assertTrue(callable(getattr(pkg, "train")))
            self.assertTrue(callable(getattr(pkg, "evaluate")))

    def test_deleted_model_directories_removed(self) -> None:
        """Deleted models FCN_field, line_transformer, semantic_line_matcher should not exist."""
        for mdir in ["FCN_field", "line_transformer", "semantic_line_matcher", "line_matcher"]:
            d = ROOT / "models" / mdir
            self.assertFalse(d.exists(), f"Deleted directory models/{mdir} should be completely removed")

    def test_train_dispatcher_coverage(self) -> None:
        """Every canonical model type in TRAIN_DISPATCH must resolve to an importable module."""
        for model_type, module_path in TRAIN_DISPATCH.items():
            mod = importlib.import_module(module_path)
            self.assertTrue(hasattr(mod, "train"), f"{module_path} missing 'train'")
            self.assertTrue(callable(getattr(mod, "train")))

    def test_root_train_applicable_arguments(self) -> None:
        """Root train.py parameter analysis must properly classify arguments for all families."""
        point_args = train_applicable_arguments("SuperPoint")
        self.assertIn("train_stage", point_args)
        self.assertIn("weight_desc", point_args)
        self.assertIn("tau_df", point_args)

        object_args = train_applicable_arguments("xfeat_object")
        self.assertIn("dustbin_weight", object_args)
        self.assertIn("min_area_ratio", object_args)

        mask_args = train_applicable_arguments("mask_line")
        self.assertIn("min_matches", mask_args)


if __name__ == "__main__":
    unittest.main()
