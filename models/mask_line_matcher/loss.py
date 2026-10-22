"""
LineFeat Loss Functions for Supervised Line Matching Training.

Implements:
  1. LineMatchingLoss: Negative Log-Likelihood (NLL) on Sinkhorn assignment distributions.
  2. MaskedMatrixLoss: Dual NLL + Masked Binary Cross-Entropy on the canonical 255x255 match matrix.
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


class MaskedMatrixLoss(nn.Module):
    """
    Supervision loss tailored for the canonical 255x255 match probability matrix.
    Combines Sinkhorn NLL with Masked Binary Cross-Entropy on the valid (N0 x N1) subregion.
    """

    def __init__(self, lambda_bce: float = 1.0) -> None:
        super().__init__()
        self.lambda_bce = lambda_bce

    def forward(
        self,
        matrix_255: torch.Tensor,
        log_assignment: torch.Tensor,
        gt_pairs: list[tuple[int, int]] | torch.Tensor,
        N0: int,
        N1: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            matrix_255: (B, 255, 255) predicted canonical probability matrix.
            log_assignment: (B, N0+1, N1+1) log Sinkhorn assignment distribution.
            gt_pairs: list of (idx0, idx1) true correspondence indices.
            N0: number of lines in View 0.
            N1: number of lines in View 1.

        Returns:
            total_loss: scalar Tensor
            metrics: dict with loss and accuracy
        """
        B = matrix_255.shape[0]
        device = matrix_255.device

        # 1. Construct target binary matrix for valid subregion
        target_sub = torch.zeros((B, N0, N1), device=device, dtype=torch.float32)
        if isinstance(gt_pairs, list):
            for i0, i1 in gt_pairs:
                if i0 < N0 and i1 < N1:
                    target_sub[:, i0, i1] = 1.0
        elif isinstance(gt_pairs, torch.Tensor):
            for p in gt_pairs:
                i0, i1 = int(p[0]), int(p[1])
                if i0 < N0 and i1 < N1:
                    target_sub[:, i0, i1] = 1.0

        M = len(gt_pairs)
        if M == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "acc": 0.0}

        # 2. Sinkhorn NLL on positive matches
        log_P_sub = log_assignment[:, :N0, :N1]
        pos_log_p = []
        for i0, i1 in gt_pairs:
            if i0 < N0 and i1 < N1:
                pos_log_p.append(log_P_sub[:, i0, i1])

        if len(pos_log_p) > 0:
            nll_loss = -torch.mean(torch.stack(pos_log_p))
        else:
            nll_loss = torch.tensor(0.0, device=device)

        # 3. Masked BCE on active (N0 x N1) subregion in FP32
        pred_sub = matrix_255[:, :N0, :N1].float().clamp(1e-6, 1.0 - 1e-6)
        with torch.amp.autocast(device.type, enabled=False):
            bce_loss = F.binary_cross_entropy(pred_sub, target_sub.float())

        total_loss = nll_loss + self.lambda_bce * bce_loss

        # 4. Metrics
        with torch.no_grad():
            pred_j = torch.argmax(log_P_sub[0], dim=1)
            correct = 0
            for i0, i1 in gt_pairs:
                if pred_j[i0].item() == i1:
                    correct += 1
            acc = (correct / float(M)) * 100.0

        metrics = {
            "loss": float(total_loss.item()),
            "nll_loss": float(nll_loss.item()),
            "bce_loss": float(bce_loss.item()),
            "acc": float(acc),
            "correct": int(correct),
            "total_gt": int(M),
        }
        return total_loss, metrics
