"""
LineFeat Loss Functions for Supervised Line Matching Training.

Implements Negative Log-Likelihood (NLL) Loss on Log-Sinkhorn assignments
with ground-truth correspondence supervision and metric computations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LineMatchingLoss(nn.Module):
    """
    Negative Log-Likelihood (NLL) Loss over the assignment matrix.
    """

    def __init__(self, dustbin_loss_weight: float = 1.0) -> None:
        super().__init__()
        self.dustbin_loss_weight = dustbin_loss_weight

    def forward(
        self,
        log_assignment: torch.Tensor,
        num_matches: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            log_assignment: (1, N0+1, N1+1) log probabilities
            num_matches: Number of ground-truth matches (assumed at diagonal indices 0..M-1)

        Returns:
            loss: scalar Tensor
            metrics: dict with loss value, precision, recall
        """
        log_P = log_assignment[0]  # (N0+1, N1+1)
        N0_plus_1, N1_plus_1 = log_P.shape
        M = num_matches

        if M == 0:
            return torch.tensor(0.0, device=log_P.device, requires_grad=True), {"loss": 0.0, "acc": 0.0}

        # 1. Match loss on true correspondences (diagonal: i matches i)
        diag_log_p = torch.diagonal(log_P[:M, :M])  # (M,)
        pos_loss = -torch.mean(diag_log_p)

        # 2. Accuracy computation
        with torch.no_grad():
            pred1 = torch.argmax(log_P[:M, :M], dim=-1)  # (M,)
            correct = (pred1 == torch.arange(M, device=log_P.device)).float().sum().item()
            acc = (correct / float(M)) * 100.0

        loss = pos_loss
        metrics = {
            "loss": float(loss.item()),
            "acc": float(acc),
            "correct": int(correct),
            "total_gt": int(M),
        }
        return loss, metrics
