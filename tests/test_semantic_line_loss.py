"""Tests for SemanticLineMatchingLoss (KL / JSD / geo added across steps)."""

from __future__ import annotations

import unittest

import torch

from models.semantic_line_matcher.loss import SemanticLineMatchingLoss, _assignment_kl


class TestSemanticLineKL(unittest.TestCase):
    def test_assignment_kl_prefers_correct(self) -> None:
        # 2x2 + dustbin
        log_P = torch.full((3, 3), -10.0)
        log_P[0, 0] = -0.1
        log_P[1, 1] = -0.1
        log_P[0, 2] = -5.0
        log_P[1, 2] = -5.0
        log_P[2, 0] = -5.0
        log_P[2, 1] = -5.0
        good = _assignment_kl(log_P, [(0, 0), (1, 1)], [], [], dustbin_weight=0.2)
        bad_log = log_P.clone()
        bad_log[0, 0] = -5.0
        bad_log[0, 1] = -0.1
        bad = _assignment_kl(bad_log, [(0, 0), (1, 1)], [], [], dustbin_weight=0.2)
        self.assertLess(float(good), float(bad))

    def test_forward_kl_backward(self) -> None:
        loss_fn = SemanticLineMatchingLoss(sem_weight=0.0, geo_weight=0.0)
        log_P = torch.randn(1, 3, 3, requires_grad=True)
        log_P_line = torch.randn(3, 3, requires_grad=True)
        # normalize-ish via clamp path inside
        total, metrics = loss_fn(
            log_P_region=log_P,
            log_P_line=log_P_line,
            gt_matches=[[(0, 0)]],
            unmatched_0=[[1]],
            unmatched_1=[[1]],
            gt_line_pairs_flat=[[(0, 0)]],
            unmatched_lines_flat_0=[[1]],
            unmatched_lines_flat_1=[[1]],
            has_line_ids=True,
        )
        total.backward()
        self.assertIn("loss_region", metrics)
        self.assertIn("loss_line", metrics)
        self.assertIsNotNone(log_P.grad)
        self.assertIsNotNone(log_P_line.grad)


class TestSemanticLineJSD(unittest.TestCase):
    def test_jsd_concentrated_smaller_than_dispersed(self) -> None:
        loss_fn = SemanticLineMatchingLoss(sem_weight=1.0, geo_weight=0.0, region_weight=0.0, line_weight=0.0)
        # 2 lines in region 0, 2 lines in region 1 on the other side + dustbin
        # Concentrated: both lines mass on semantic 0
        P_c = torch.zeros(3, 3)
        P_c[0, 0] = 0.9
        P_c[0, 2] = 0.1
        P_c[1, 0] = 0.85
        P_c[1, 2] = 0.15
        P_c[2, :] = 0.0
        P_c[2, 2] = 1.0
        log_c = torch.log(P_c.clamp(min=1e-8))
        parents0 = [torch.tensor([0, 0])]
        parents1 = [torch.tensor([0, 1])]
        jsd_c = loss_fn.semantic_jsd_loss(
            log_P_line=log_c,
            line_parents0=parents0,
            line_parents1=parents1,
            num_regions0=1,
            num_regions1=2,
            enabled=True,
        )
        # Dispersed: line0→sem0, line1→sem1
        P_d = torch.zeros(3, 3)
        P_d[0, 0] = 0.9
        P_d[0, 2] = 0.1
        P_d[1, 1] = 0.9
        P_d[1, 2] = 0.1
        P_d[2, 2] = 1.0
        log_d = torch.log(P_d.clamp(min=1e-8))
        jsd_d = loss_fn.semantic_jsd_loss(
            log_P_line=log_d,
            line_parents0=parents0,
            line_parents1=parents1,
            num_regions0=1,
            num_regions1=2,
            enabled=True,
        )
        self.assertLess(float(jsd_c), float(jsd_d))


if __name__ == "__main__":
    unittest.main()

