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


if __name__ == "__main__":
    unittest.main()
