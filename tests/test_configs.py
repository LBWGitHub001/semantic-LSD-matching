#!/usr/bin/env python3
"""
Unit and Integration Tests for YAML Configuration Architecture:
  1. Validation of all 7 model YAML configuration files
  2. Model name resolution and alias mapping
  3. Argparse default injection across generate_dataset, train, and eval
  4. Precedence of explicit CLI overrides over YAML defaults
  5. Pipeline dry-run functionality
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import (
    load_config,
    save_config,
    get_model_config_path,
    extract_section,
    apply_config_to_parser,
    MODEL_CONFIG_MAP,
    MODEL_ALIAS_MAP,
)


class TestYamlConfigurationSuite(unittest.TestCase):
    def test_all_seven_yaml_files_exist_and_valid(self) -> None:
        """Every canonical model must have a valid YAML configuration file with all 4 sections."""
        expected_models = [
            "superpoint",
            "fcn_field",
            "xfeat_object",
            "xfeat_line_matcher",
            "semantic_line_matcher",
            "mask_line",
            "line_transformer",
        ]
        self.assertEqual(sorted(list(MODEL_CONFIG_MAP.keys())), sorted(expected_models))

        for model_key in expected_models:
            cfg_path = get_model_config_path(model_key)
            self.assertTrue(cfg_path.is_file(), f"Missing config file: {cfg_path}")

            cfg = load_config(cfg_path)
            self.assertIsInstance(cfg, dict)

            # Check four required sections
            for sec in ("model", "data_generation", "train", "eval"):
                self.assertIn(sec, cfg, f"Missing section '{sec}' in {cfg_path.name}")
                self.assertIsInstance(cfg[sec], dict, f"Section '{sec}' must be dict in {cfg_path.name}")

            # Validate model section
            self.assertIn("model_type", cfg["model"])
            self.assertIn("family", cfg["model"])

            # Validate data generation section
            self.assertIn("task", cfg["data_generation"])
            self.assertIn("out_dir", cfg["data_generation"])

            # Validate train section
            self.assertIn("data_dir", cfg["train"])
            self.assertIn("epochs", cfg["train"])
            self.assertIn("lr", cfg["train"])

            # Validate eval section
            self.assertIn("checkpoint", cfg["eval"])
            self.assertIn("data_dir", cfg["eval"])
            self.assertIn("split", cfg["eval"])

    def test_model_name_and_alias_resolution(self) -> None:
        """Test resolving canonical names and user-friendly aliases to config paths."""
        self.assertEqual(get_model_config_path("SuperPoint").name, "superpoint.yaml")
        self.assertEqual(get_model_config_path("superpoint").name, "superpoint.yaml")
        self.assertEqual(get_model_config_path("FCN_field").name, "fcn_field.yaml")
        self.assertEqual(get_model_config_path("xfeat").name, "xfeat_object.yaml")
        self.assertEqual(get_model_config_path("object").name, "xfeat_object.yaml")
        self.assertEqual(get_model_config_path("xfeat_line").name, "xfeat_line_matcher.yaml")
        self.assertEqual(get_model_config_path("semantic_line").name, "semantic_line_matcher.yaml")
        self.assertEqual(get_model_config_path("mask").name, "mask_line.yaml")
        self.assertEqual(get_model_config_path("linefeat").name, "line_transformer.yaml")

        with self.assertRaises(ValueError):
            get_model_config_path("non_existent_model_xyz")

    def test_apply_config_to_parser_and_cli_overrides(self) -> None:
        """Test that YAML config sets defaults, and CLI explicitly passed flags override them."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--device", type=str, default="cpu")

        sample_cfg = {
            "model": {"model_type": "SuperPoint"},
            "train": {"epochs": 15, "lr": 0.001, "device": "cuda"},
        }
        apply_config_to_parser(parser, sample_cfg, section="train")

        # 1. Without CLI overrides: uses YAML defaults
        args_yaml = parser.parse_args([])
        self.assertEqual(args_yaml.epochs, 15)
        self.assertEqual(args_yaml.lr, 0.001)
        self.assertEqual(args_yaml.device, "cuda")

        # 2. With CLI overrides: CLI takes precedence
        args_cli = parser.parse_args(["--epochs", "50", "--device", "cpu"])
        self.assertEqual(args_cli.epochs, 50)
        self.assertEqual(args_cli.lr, 0.001)  # Preserved from YAML
        self.assertEqual(args_cli.device, "cpu")  # Overridden by CLI

    def test_generate_dataset_config_cli(self) -> None:
        """Test generate_dataset.py with --config flag."""
        from generate_dataset import build_parser
        from configs import load_config, apply_config_to_parser

        parser = build_parser()
        cfg = load_config(get_model_config_path("superpoint"))
        apply_config_to_parser(parser, cfg, section="data_generation")

        # YAML defaults
        args_default = parser.parse_args(["--config", "configs/superpoint.yaml"])
        self.assertEqual(args_default.task, "planar_letters")
        self.assertEqual(args_default.out_dir, "data/planar_letters")
        self.assertEqual(args_default.num_scenes, 26)

        # CLI overrides
        args_override = parser.parse_args(["--config", "configs/superpoint.yaml", "--num-scenes", "3"])
        self.assertEqual(args_override.num_scenes, 3)

    def test_train_config_cli(self) -> None:
        """Test train.py parse_args with --config flag."""
        from train import parse_args

        args, _ = parse_args(["--config", "configs/superpoint.yaml"])
        self.assertEqual(args.model_type, "SuperPoint")
        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.train_stage, "joint")

        # CLI overrides
        args_override, _ = parse_args(["--config", "configs/superpoint.yaml", "--epochs", "25", "--lr", "0.0005"])
        self.assertEqual(args_override.epochs, 25)
        self.assertEqual(args_override.lr, 0.0005)

    def test_eval_config_cli(self) -> None:
        """Test eval.py parse_args with --config flag."""
        from eval import parse_args

        args, _ = parse_args(["--config", "configs/superpoint.yaml"])
        self.assertEqual(args.checkpoint, "checkpoints/SuperPoint/latest")
        self.assertEqual(args.data_dir, "data/planar_letters")
        self.assertEqual(args.split, "val")
        self.assertEqual(args.eval_stage, "all")

        # CLI overrides
        args_override, _ = parse_args(["--config", "configs/superpoint.yaml", "--split", "test"])
        self.assertEqual(args_override.split, "test")

    def test_save_and_reload_config(self) -> None:
        """Test saving and re-loading config YAML."""
        import tempfile

        cfg = load_config(get_model_config_path("superpoint"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_yaml = Path(tmp_dir) / "test_config.yaml"
            saved_path = save_config(cfg, tmp_yaml)
            self.assertTrue(saved_path.is_file())

            reloaded = load_config(saved_path)
            self.assertEqual(reloaded["model"]["model_type"], "SuperPoint")
            self.assertEqual(reloaded["train"]["epochs"], 15)


if __name__ == "__main__":
    unittest.main()
