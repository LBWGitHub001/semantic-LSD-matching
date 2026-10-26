"""
Comprehensive Unit Tests for XFeatLineMatcher.

Tests:
  1. PatchLineGeometryEncoder 5D conversion, masked pooling, and null_line_token fallback.
  2. RegionPatchExtractor full image patch cropping and line association.
  3. XFeatLineMatcher forward pass (patches mode & full images mode).
  4. Variable object & line counts (e.g., M0 != M1, 0 lines in patch).
  5. Loss computation and backward gradient verification.
  6. ModelFactory auto-detection and checkpoint serialization/loading.
"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_loss, build_model
from models.object_matcher import (
    XFeatLineMatcher,
    PatchLineGeometryEncoder,
    RegionPatchExtractor,
)
from evaluation import ModelFactory


class TestXFeatLineMatcher(unittest.TestCase):
    def test_line_encoder(self) -> None:
        encoder = PatchLineGeometryEncoder(line_dim=64)

        # 1. Test endpoints to 5D
        lines_4d = torch.tensor([[[[10.0, 20.0, 50.0, 60.0]]]])  # (1, 1, 1, 4)
        geom_5d = encoder.endpoints_to_5d(lines_4d)
        self.assertEqual(geom_5d.shape, (1, 1, 1, 5))
        self.assertTrue(torch.isclose(geom_5d[0, 0, 0, 0], torch.tensor(30.0)))
        self.assertTrue(torch.isclose(geom_5d[0, 0, 0, 1], torch.tensor(40.0)))

        # 2. Test forward with 3 patches: patch 0 has 2 lines, patch 1 has 0 lines, patch 2 has 1 line
        B, M, K = 1, 3, 4
        lines_tensor = torch.zeros((B, M, K, 5))
        mask_tensor = torch.zeros((B, M, K), dtype=torch.bool)

        # Patch 0: 2 lines
        lines_tensor[0, 0, 0] = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.866])
        lines_tensor[0, 0, 1] = torch.tensor([0.7, 0.8, 0.2, -0.5, 0.866])
        mask_tensor[0, 0, 0] = True
        mask_tensor[0, 0, 1] = True

        # Patch 1: 0 lines (all False mask)

        # Patch 2: 1 line
        lines_tensor[0, 2, 0] = torch.tensor([0.5, 0.5, 0.6, 1.0, 0.0])
        mask_tensor[0, 2, 0] = True

        tokens = encoder(lines_tensor, valid_mask=mask_tensor)
        self.assertEqual(tokens.shape, (B, M, 64))

        # Verify that patch 1 receives null_line_token exactly
        self.assertTrue(torch.allclose(tokens[0, 1], encoder.null_line_token))
        # Verify that patch 0 and 2 do NOT equal null_line_token
        self.assertFalse(torch.allclose(tokens[0, 0], encoder.null_line_token))
        self.assertFalse(torch.allclose(tokens[0, 2], encoder.null_line_token))

        # 3. Test gradient flow to null_line_token
        loss = tokens.sum()
        loss.backward()
        self.assertIsNotNone(encoder.null_line_token.grad)
        self.assertGreater(encoder.null_line_token.grad.norm().item(), 0)

    def test_patch_extractor_and_line_associator(self) -> None:
        B, H, W = 2, 480, 640
        images = torch.rand((B, 3, H, W))
        bboxes = torch.tensor([
            [[50.0, 60.0, 150.0, 160.0], [200.0, 200.0, 300.0, 350.0]],
            [[10.0, 10.0, 80.0, 80.0], [100.0, 100.0, 250.0, 250.0]],
        ])  # (2, 2, 4)

        # 1. Test crop_patches
        patches = RegionPatchExtractor.crop_patches(images, bboxes, patch_size=112)
        self.assertEqual(patches.shape, (B, 2, 3, 112, 112))

        # 2. Test associate_lines_with_bboxes
        lines = torch.tensor([
            [[60.0, 70.0, 140.0, 150.0], [500.0, 400.0, 520.0, 420.0]],
            [[15.0, 15.0, 70.0, 70.0], [150.0, 150.0, 200.0, 200.0]],
        ])
        patch_lines, valid_mask = RegionPatchExtractor.associate_lines_with_bboxes(lines, bboxes, max_lines_per_patch=4)
        self.assertEqual(patch_lines.shape, (B, 2, 4, 5))
        self.assertEqual(valid_mask.shape, (B, 2, 4))
        self.assertEqual(valid_mask[0, 0].sum().item(), 1)
        self.assertEqual(valid_mask[0, 1].sum().item(), 0)

    def test_xfeat_line_matcher_forward_and_backward(self) -> None:
        model = build_model("xfeat_line_matcher", d_model=128, num_layers=2, nhead=4)
        loss_fn = build_loss("xfeat_line_matcher")

        # Mode 1: Pre-cropped patches forward
        B, M0, M1 = 1, 5, 7
        patches0 = torch.rand((B, M0, 3, 112, 112))
        patches1 = torch.rand((B, M1, 3, 112, 112))
        patch_lines0 = torch.rand((B, M0, 8, 5))
        patch_lines1 = torch.rand((B, M1, 8, 5))
        bboxes0 = torch.rand((B, M0, 4)) * 400
        bboxes1 = torch.rand((B, M1, 4)) * 400

        out = model(
            patches0=patches0,
            patches1=patches1,
            patch_lines0=patch_lines0,
            patch_lines1=patch_lines1,
            bboxes0=bboxes0,
            bboxes1=bboxes1,
        )

        self.assertEqual(out["prob_matrix"].shape, (B, M0 + 1, M1 + 1))
        self.assertEqual(out["log_P"].shape, (B, M0 + 1, M1 + 1))
        self.assertEqual(out["tokens0"].shape, (B, M0, 128))
        self.assertEqual(out["tokens1"].shape, (B, M1, 128))

        row_sums = out["prob_matrix"][0, :M0, :M1].sum(dim=-1)
        self.assertTrue((row_sums <= 1.05).all())

        # Mode 2: Full-image forward
        img0 = torch.rand((B, 3, 240, 320))
        img1 = torch.rand((B, 3, 240, 320))
        lines0 = torch.rand((B, 12, 4)) * 200
        lines1 = torch.rand((B, 15, 4)) * 200

        out_full = model(
            img0=img0,
            img1=img1,
            bboxes0=bboxes0,
            bboxes1=bboxes1,
            lines0=lines0,
            lines1=lines1,
        )
        self.assertEqual(out_full["prob_matrix"].shape, (B, M0 + 1, M1 + 1))

        # Mode 3: Test Loss & Backward pass
        gt_matches = [[(0, 0), (1, 2), (2, 3)]]
        unmatched_0 = [[3, 4]]
        unmatched_1 = [[1, 4, 5, 6]]

        loss = loss_fn(out["log_P"], gt_matches, unmatched_0, unmatched_1)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        loss.backward()

        self.assertIsNotNone(model.visual_backbone.stage1[0].weight.grad)
        self.assertIsNotNone(model.line_encoder.line_mlp[0].weight.grad)
        self.assertIsNotNone(model.fusion_proj[0].weight.grad)
        self.assertIsNotNone(model.matcher.proj0.weight.grad)

    def test_model_factory_auto_detection(self) -> None:
        model = build_model("xfeat_line_matcher", d_model=128, use_pyramid=True, with_geometry=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_line_matcher.pth"
            torch.save({"model_state": model.state_dict(), "epoch": 1}, ckpt_path)

            dev = torch.device("cpu")
            loaded_model, info, _ = ModelFactory.create_from_checkpoint(ckpt_path, device=dev)

            self.assertEqual(info["model_type"], "xfeat_line_matcher")
            self.assertTrue(info["use_pyramid"])
            self.assertTrue(info["with_geometry"])
            self.assertIsInstance(loaded_model, XFeatLineMatcher)


if __name__ == "__main__":
    unittest.main()
