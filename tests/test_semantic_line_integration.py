"""End-to-end fake-batch: SemanticLineMatcher forward + four-term loss."""

from __future__ import annotations

import unittest

import torch

from models import build_loss, build_model


class TestSemanticLineIntegration(unittest.TestCase):
    def test_forward_and_four_losses(self) -> None:
        model = build_model(
            "semantic_line_matcher",
            feat_dim=32,
            line_dim=32,
            d_model=64,
            num_layers=1,
            nhead=2,
            use_pyramid=False,
            with_geometry=True,
            line_sample_points=4,
            enable_line_head=True,
        )
        loss_fn = build_loss(
            "semantic_line_matcher",
            dustbin_weight=0.2,
            region_weight=1.0,
            line_weight=1.0,
            sem_weight=0.1,
            geo_weight=0.2,
            adj_sigma=0.3,
        )

        B, M0, M1, K = 1, 3, 3, 2
        patches0 = torch.randn(B, M0, 3, 48, 48)
        patches1 = torch.randn(B, M1, 3, 48, 48)
        bboxes0 = torch.tensor([[[0, 0, 40, 40], [20, 10, 60, 50], [5, 30, 45, 70]]], dtype=torch.float32)
        bboxes1 = bboxes0.clone()
        lines = torch.zeros(B, M0, K, 5)
        lines[..., 0], lines[..., 1], lines[..., 2], lines[..., 4] = 0.5, 0.5, 0.2, 1.0
        mask = torch.ones(B, M0, K, dtype=torch.bool)
        centroids = torch.tensor([[[-0.5, -0.5], [0.5, -0.4], [0.0, 0.6]]], dtype=torch.float32)

        out = model(
            patches0,
            patches1,
            patch_lines0=lines,
            patch_lines1=lines.clone(),
            line_mask0=mask,
            line_mask1=mask.clone(),
            bboxes0=bboxes0,
            bboxes1=bboxes1,
            image_size=(80, 80),
        )
        self.assertEqual(tuple(out["log_P"].shape), (1, 4, 4))
        self.assertTrue(torch.is_tensor(out["log_P_line"]))

        n0 = int(mask[0].sum().item())
        loss, metrics = loss_fn(
            log_P_region=out["log_P"],
            log_P_line=out["log_P_line"],
            gt_matches=[[(0, 0), (1, 1)]],
            unmatched_0=[[2]],
            unmatched_1=[[2]],
            gt_line_pairs_flat=[[(0, 0)]],
            unmatched_lines_flat_0=[[1]],
            unmatched_lines_flat_1=[[1]],
            line_parents0=out["line_parents0"],
            line_parents1=out["line_parents1"],
            centroids0=centroids,
            centroids1=centroids.clone(),
            has_line_ids=True,
        )
        self.assertTrue(torch.isfinite(loss))
        for key in ("loss_region", "loss_line", "loss_sem", "loss_geo"):
            self.assertIn(key, metrics)
        loss.backward()
        grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        self.assertTrue(grad_ok)

    def test_sem_geo_zero_degrades_to_dual_kl(self) -> None:
        loss_fn = build_loss(
            "semantic_line",
            sem_weight=0.0,
            geo_weight=0.0,
            region_weight=1.0,
            line_weight=1.0,
        )
        log_P = torch.randn(1, 3, 3, requires_grad=True)
        log_line = torch.randn(3, 3, requires_grad=True)
        loss, metrics = loss_fn(
            log_P_region=log_P,
            log_P_line=log_line,
            gt_matches=[[(0, 0)]],
            unmatched_0=[[1]],
            unmatched_1=[[1]],
            gt_line_pairs_flat=[[(0, 0)]],
            unmatched_lines_flat_0=[[1]],
            unmatched_lines_flat_1=[[1]],
            has_line_ids=True,
        )
        self.assertEqual(metrics["loss_sem"], 0.0)
        self.assertEqual(metrics["loss_geo"], 0.0)
        loss.backward()


if __name__ == "__main__":
    unittest.main()
