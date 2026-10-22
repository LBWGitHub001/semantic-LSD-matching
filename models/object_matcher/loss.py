"""
Loss function for object-level bipartite matching.
Includes NLL matching loss and Motion Vector Consensus Loss (方案一：变换一致性损失).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class ObjectMatchingLoss(nn.Module):
    """
    SuperGlue-style Negative Log-Likelihood (NLL) Loss over Sinkhorn probability matrix
    coupled with Motion Vector Consensus Loss (变换一致性损失).
    Supervises co-visible matches, dustbin rejections, and penalizes candidates that
    violate the consensus 2D displacement field of the camera transformation.
    """

    def __init__(
        self,
        dustbin_weight: float = 0.3,
        motion_weight: float = 0.2,
        motion_margin: float = 0.15,
    ) -> None:
        super().__init__()
        # Dustbin weight: 0.2 ~ 0.4 (0.3 is our empirical sweet spot)
        self.dustbin_weight = float(dustbin_weight)
        # Transformation consistency weight (方案一：位移向量场共识损失权重)
        self.motion_weight = float(motion_weight)
        # Parallax tolerance margin in [-1, 1] normalized coordinates
        self.motion_margin = float(motion_margin)

    def forward(
        self,
        log_P: torch.Tensor,
        gt_matches: List[List[Tuple[int, int]]],
        unmatched_0: List[List[int]],
        unmatched_1: List[List[int]],
        centroids0: Optional[torch.Tensor] = None,
        centroids1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, M0_plus_1, M1_plus_1 = log_P.shape
        M0 = M0_plus_1 - 1
        M1 = M1_plus_1 - 1

        # Safe clamp to guarantee finite log probabilities
        log_P = torch.clamp(log_P, min=-100.0, max=0.0)
        total_nll_loss = torch.tensor(0.0, device=log_P.device)
        total_motion_loss = torch.tensor(0.0, device=log_P.device)

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

            total_nll_loss = total_nll_loss + mean_pos + self.dustbin_weight * mean_neg

            # 4. Motion Vector Consensus Loss (方案一：位移向量场共识损失)
            if self.motion_weight > 0.0 and centroids0 is not None and centroids1 is not None and M0 > 0 and M1 > 0:
                c0 = centroids0[b]  # (M0, 2)
                c1 = centroids1[b]  # (M1, 2)

                # Determine consensus motion vector of the camera transformation
                if n_pos > 0:
                    # Ground-truth guided consensus: mean displacement of true matches
                    disp_gt = torch.stack([c1[j] - c0[i] for i, j in matches], dim=0)  # (K, 2)
                    mean_disp = disp_gt.mean(dim=0, keepdim=True)  # (1, 2)
                else:
                    # Self-supervised fallback: probability-weighted mean displacement
                    P_sub = torch.exp(log_P[b, :M0, :M1])  # (M0, M1)
                    disp_all_fb = c1.unsqueeze(0) - c0.unsqueeze(1)  # (M0, M1, 2)
                    sum_P = P_sub.sum().clamp(min=1e-4)
                    mean_disp = (P_sub.unsqueeze(-1) * disp_all_fb).sum(dim=(0, 1), keepdim=True) / sum_P

                # Candidate displacement for all (i, j): c1[j] - c0[i]
                disp_all = c1.unsqueeze(0) - c0.unsqueeze(1)  # (M0, M1, 2)
                residuals = torch.norm(disp_all - mean_disp, dim=-1)  # (M0, M1)

                # Margin-based penalty: only penalize violations larger than parallax margin
                motion_penalty = (torch.clamp(residuals - self.motion_margin, min=0.0) ** 2).detach()

                # Penalize predicted probability mass assigned to inconsistent pairs
                P_real = torch.exp(log_P[b, :M0, :M1])  # (M0, M1)
                loss_motion_b = (P_real * motion_penalty).sum() / max(float(M0), 1.0)
                total_motion_loss = total_motion_loss + loss_motion_b

        avg_loss = (total_nll_loss + self.motion_weight * total_motion_loss) / float(B)
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            return torch.tensor(0.0, device=log_P.device, requires_grad=True)
        return avg_loss
