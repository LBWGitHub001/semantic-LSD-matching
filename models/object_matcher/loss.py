"""
Loss function for object-level bipartite matching.
"""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn


class ObjectMatchingLoss(nn.Module):
    """
    SuperGlue-style Negative Log-Likelihood (NLL) Loss over Sinkhorn probability matrix.
    Supervises co-visible matches and dustbin rejections with balanced weights.
    """

    def __init__(self, dustbin_weight: float = 0.3) -> None:
        super().__init__()
        # Dustbin weight: 0.2 ~ 0.4 (0.3 is our empirical sweet spot for balancing recall & precision)
        self.dustbin_weight = float(dustbin_weight)

    def forward(
        self,
        log_P: torch.Tensor,
        gt_matches: List[List[Tuple[int, int]]],
        unmatched_0: List[List[int]],
        unmatched_1: List[List[int]],
    ) -> torch.Tensor:
        B, M0_plus_1, M1_plus_1 = log_P.shape
        M0 = M0_plus_1 - 1
        M1 = M1_plus_1 - 1

        total_loss = torch.tensor(0.0, device=log_P.device)

        for b in range(B):
            matches = gt_matches[b]
            un0 = unmatched_0[b]
            un1 = unmatched_1[b]

            loss_pos = torch.tensor(0.0, device=log_P.device)
            loss_neg = torch.tensor(0.0, device=log_P.device)
            n_pos = len(matches)
            n_neg = len(un0) + len(un1)

            # 1. Matched pairs (Positive)
            for i, j in matches:
                loss_pos = loss_pos - log_P[b, i, j]

            # 2. View 0 dustbin (Negative)
            for i in un0:
                loss_neg = loss_neg - log_P[b, i, M1]

            # 3. View 1 dustbin (Negative)
            for j in un1:
                loss_neg = loss_neg - log_P[b, M0, j]

            # Independently normalize positive and negative losses
            mean_pos = (loss_pos / float(n_pos)) if n_pos > 0 else torch.tensor(0.0, device=log_P.device)
            mean_neg = (loss_neg / float(n_neg)) if n_neg > 0 else torch.tensor(0.0, device=log_P.device)

            # Balanced weighted summation
            total_loss = total_loss + mean_pos + self.dustbin_weight * mean_neg

        return total_loss / float(B)
