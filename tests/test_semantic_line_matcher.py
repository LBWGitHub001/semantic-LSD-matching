"""Dual-output forward tests for SemanticLineMatcher."""

from __future__ import annotations

import unittest

import torch

from models.semantic_line_matcher.model import SemanticLineMatcher


class TestSemanticLineMatcherDual(unittest.TestCase):
    def test_dual_outputs(self) -> None:
        model = SemanticLineMatcher(
            feat_dim=32,
            line_dim=32,
            d_model=64,
            num_layers=1,
            nhead=2,
            use_pyramid=False,
            with_geometry=False,
            line_sample_points=4,
            enable_line_head=True,
        )
        B, M0, M1, K = 1, 2, 2, 3
        patches0 = torch.randn(B, M0, 3, 48, 48)
        patches1 = torch.randn(B, M1, 3, 48, 48)
        lines0 = torch.zeros(B, M0, K, 5)
        lines1 = torch.zeros(B, M1, K, 5)
        for lines in (lines0, lines1):
            lines[..., 0] = 0.5
            lines[..., 1] = 0.5
            lines[..., 2] = 0.25
            lines[..., 4] = 1.0
        mask0 = torch.ones(B, M0, K, dtype=torch.bool)
        mask0[0, 0, 2] = False
        mask1 = torch.ones(B, M1, K, dtype=torch.bool)
        out = model(
            patches0,
            patches1,
            patch_lines0=lines0,
            patch_lines1=lines1,
            line_mask0=mask0,
            line_mask1=mask1,
        )
        self.assertEqual(tuple(out["log_P"].shape), (1, M0 + 1, M1 + 1))
        n0 = int(mask0[0].sum().item())
        n1 = int(mask1[0].sum().item())
        self.assertEqual(tuple(out["log_P_line"].shape), (n0 + 1, n1 + 1))
        self.assertEqual(out["line_parents0"][0].numel(), n0)
        self.assertEqual(out["line_parents1"][0].numel(), n1)
        loss = out["log_P"].sum() + out["log_P_line"].sum()
        loss.backward()


if __name__ == "__main__":
    unittest.main()
