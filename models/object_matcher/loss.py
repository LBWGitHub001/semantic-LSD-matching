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
    Supervises co-visible matches and dustbin rejections.
    """

    def __init__(self) -> None:
        super().__init__()

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
        total_items = 0

        for b in range(B):
            matches = gt_matches[b]
            un0 = unmatched_0[b]
            un1 = unmatched_1[b]

            loss_b = torch.tensor(0.0, device=log_P.device)
            # 1. Matched pairs
            for i, j in matches:
                loss_b = loss_b - log_P[b, i, j]
                total_items += 1

            # 2. Unmatched in View 0 (dustbin column index M1)
            for i in un0:
                loss_b = loss_b - log_P[b, i, M1]
                total_items += 1

            # 3. Unmatched in View 1 (dustbin row index M0)
            for j in un1:
                loss_b = loss_b - log_P[b, M0, j]
                total_items += 1

            total_loss = total_loss + loss_b

        if total_items > 0:
            return total_loss / float(total_items)
        return total_loss
