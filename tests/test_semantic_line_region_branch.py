"""Region-branch smoke test for SemanticLineMatcher (step05)."""

from __future__ import annotations

import unittest

import torch

from models.semantic_line_matcher.model import SemanticLineMatcher


class TestSemanticLineRegionBranch(unittest.TestCase):
    def test_region_forward_shapes(self) -> None:
        model = SemanticLineMatcher(
            feat_dim=32,
            d_model=64,
            num_layers=1,
            nhead=2,
            use_pyramid=False,
            with_geometry=True,
            enable_line_head=False,
        )
        B, M0, M1 = 1, 3, 4
        patches0 = torch.randn(B, M0, 3, 64, 64)
        patches1 = torch.randn(B, M1, 3, 64, 64)
        bboxes0 = torch.tensor([[[0, 0, 40, 40], [10, 10, 50, 50], [20, 5, 60, 45]]], dtype=torch.float32)
        bboxes1 = torch.tensor(
            [[[0, 0, 30, 30], [5, 5, 45, 45], [15, 15, 55, 55], [2, 20, 40, 60]]],
            dtype=torch.float32,
        )
        out = model(patches0, patches1, bboxes0=bboxes0, bboxes1=bboxes1, image_size=(120, 160))
        self.assertEqual(tuple(out["log_P"].shape), (B, M0 + 1, M1 + 1))
        self.assertEqual(tuple(out["P"].shape), (B, M0 + 1, M1 + 1))
        # Rows roughly sum to 1 in probability space after Sinkhorn
        row_sum = out["P"][0, :M0, :].sum(dim=-1)
        self.assertTrue(torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-2))


if __name__ == "__main__":
    unittest.main()
