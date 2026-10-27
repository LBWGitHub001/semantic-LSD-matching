"""
SemanticLineMatchingLoss: region KL + line KL + semantic JSD + adjacency geo.

Hard-label KL(Y||P) is equivalent to SuperGlue-style assignment NLL with dustbin.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _assignment_kl(
    log_P: torch.Tensor,
    matches: Sequence[Tuple[int, int]],
    unmatched_0: Sequence[int],
    unmatched_1: Sequence[int],
    dustbin_weight: float,
) -> torch.Tensor:
    """KL(Y||P) / NLL for one assignment matrix (N0+1, N1+1) or (1, N0+1, N1+1)."""
    if log_P.ndim == 3:
        log_P = log_P[0]
    log_P = torch.clamp(log_P, min=-100.0, max=0.0)
    N0 = log_P.shape[0] - 1
    N1 = log_P.shape[1] - 1
    device = log_P.device

    loss_pos = torch.tensor(0.0, device=device)
    n_pos = 0
    for i, j in matches:
        if 0 <= i < N0 and 0 <= j < N1:
            loss_pos = loss_pos - log_P[i, j]
            n_pos += 1

    loss_neg = torch.tensor(0.0, device=device)
    n_neg = 0
    for i in unmatched_0:
        if 0 <= i < N0:
            loss_neg = loss_neg - log_P[i, N1]
            n_neg += 1
    for j in unmatched_1:
        if 0 <= j < N1:
            loss_neg = loss_neg - log_P[N0, j]
            n_neg += 1

    mean_pos = loss_pos / float(n_pos) if n_pos > 0 else torch.tensor(0.0, device=device)
    mean_neg = loss_neg / float(n_neg) if n_neg > 0 else torch.tensor(0.0, device=device)
    return mean_pos + dustbin_weight * mean_neg


class SemanticLineMatchingLoss(nn.Module):
    """Four-term loss; JSD / geo filled in later steps (weights default applied)."""

    def __init__(
        self,
        dustbin_weight: float = 0.2,
        region_weight: float = 1.0,
        line_weight: float = 1.0,
        sem_weight: float = 0.1,
        geo_weight: float = 0.2,
        adj_sigma: float = 0.3,
        motion_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dustbin_weight = float(dustbin_weight)
        self.region_weight = float(region_weight)
        self.line_weight = float(line_weight)
        self.sem_weight = float(sem_weight)
        self.geo_weight = float(geo_weight)
        self.adj_sigma = float(adj_sigma)
        # Accept but ignore legacy motion
        self.motion_weight = float(motion_weight)

    def _region_kl(
        self,
        log_P: torch.Tensor,
        gt_matches: List[List[Tuple[int, int]]],
        unmatched_0: List[List[int]],
        unmatched_1: List[List[int]],
    ) -> torch.Tensor:
        B = log_P.shape[0]
        total = torch.tensor(0.0, device=log_P.device)
        for b in range(B):
            total = total + _assignment_kl(
                log_P[b],
                gt_matches[b] if b < len(gt_matches) else [],
                unmatched_0[b] if b < len(unmatched_0) else [],
                unmatched_1[b] if b < len(unmatched_1) else [],
                self.dustbin_weight,
            )
        return total / float(max(B, 1))

    def _line_kl(
        self,
        log_P_line: Union[torch.Tensor, List[torch.Tensor], None],
        gt_line_pairs: List[List[Tuple[int, int]]],
        unmatched_flat_0: List[List[int]],
        unmatched_flat_1: List[List[int]],
    ) -> torch.Tensor:
        if log_P_line is None:
            return torch.tensor(0.0)

        if isinstance(log_P_line, torch.Tensor):
            mats = [log_P_line]
        else:
            mats = list(log_P_line)

        total = torch.tensor(0.0, device=mats[0].device)
        n = 0
        for b, mat in enumerate(mats):
            pairs = gt_line_pairs[b] if b < len(gt_line_pairs) else []
            un0 = unmatched_flat_0[b] if b < len(unmatched_flat_0) else []
            un1 = unmatched_flat_1[b] if b < len(unmatched_flat_1) else []
            if not pairs and not un0 and not un1:
                continue
            total = total + _assignment_kl(mat, pairs, un0, un1, self.dustbin_weight)
            n += 1
        if n == 0:
            return torch.tensor(0.0, device=mats[0].device, requires_grad=True) * 0.0
        return total / float(n)

    @staticmethod
    def _line_to_semantic_matrix(
        P_line: torch.Tensor,
        parents_other: torch.Tensor,
        num_regions_other: int,
    ) -> torch.Tensor:
        """
        Marginalize line↔line probs into line→semantic (+ dustbin).

        P_line: (N_self+1, N_other+1), parents_other: (N_other,)
        Returns M: (N_self, num_regions_other+1)
        """
        N_self = P_line.shape[0] - 1
        N_other = P_line.shape[1] - 1
        device = P_line.device
        dtype = P_line.dtype
        M = torch.zeros((N_self, num_regions_other + 1), device=device, dtype=dtype)
        if N_self == 0:
            return M
        # dustbin column
        M[:, -1] = P_line[:N_self, N_other]
        if N_other == 0 or parents_other.numel() == 0:
            return M
        for j in range(N_other):
            s = int(parents_other[j].item())
            if 0 <= s < num_regions_other:
                M[:, s] = M[:, s] + P_line[:N_self, j]
        # renormalize rows
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return M

    @staticmethod
    def _generalized_jsd(rows: torch.Tensor) -> torch.Tensor:
        """
        rows: (K, C) probability rows for lines in the same region.
        JSD = sum_k w_k KL(p_k || m), w=1/K, m=mean(p).
        """
        K = rows.shape[0]
        if K <= 1:
            return rows.new_zeros(())
        w = 1.0 / float(K)
        m = rows.mean(dim=0).clamp(min=1e-8)
        m = m / m.sum()
        # KL(p||m) = sum p (log p - log m)
        p = rows.clamp(min=1e-8)
        p = p / p.sum(dim=-1, keepdim=True)
        kl = (p * (torch.log(p) - torch.log(m.unsqueeze(0)))).sum(dim=-1)
        return (w * kl).sum()

    def semantic_jsd_loss(
        self,
        log_P_line: Union[torch.Tensor, List[torch.Tensor], None] = None,
        line_parents0: Optional[List[torch.Tensor]] = None,
        line_parents1: Optional[List[torch.Tensor]] = None,
        num_regions0: int = 0,
        num_regions1: int = 0,
        device: Optional[torch.device] = None,
        enabled: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Bidirectional line→semantic generalized JSD within each parent region.
        Encourages lines of the same semantic to agree on the opposite semantic.
        """
        if not enabled or log_P_line is None or line_parents0 is None or line_parents1 is None:
            return torch.tensor(0.0, device=device or torch.device("cpu"))

        if isinstance(log_P_line, torch.Tensor):
            mats = [log_P_line]
        else:
            mats = list(log_P_line)

        total = torch.tensor(0.0, device=mats[0].device)
        n_groups = 0
        for b, log_mat in enumerate(mats):
            P = torch.exp(torch.clamp(log_mat, min=-100.0, max=0.0))
            p0 = line_parents0[b] if b < len(line_parents0) else torch.zeros(0, dtype=torch.long, device=P.device)
            p1 = line_parents1[b] if b < len(line_parents1) else torch.zeros(0, dtype=torch.long, device=P.device)

            M_ab = self._line_to_semantic_matrix(P, p1, num_regions1)
            M_ba = self._line_to_semantic_matrix(P.transpose(0, 1), p0, num_regions0)

            for r in range(num_regions0):
                idx = (p0 == r).nonzero(as_tuple=False).view(-1)
                if idx.numel() <= 1:
                    continue
                total = total + self._generalized_jsd(M_ab[idx])
                n_groups += 1
            for r in range(num_regions1):
                idx = (p1 == r).nonzero(as_tuple=False).view(-1)
                if idx.numel() <= 1:
                    continue
                total = total + self._generalized_jsd(M_ba[idx])
                n_groups += 1

        if n_groups == 0:
            return torch.tensor(0.0, device=mats[0].device)
        return total / float(n_groups)

    def _soft_adjacency(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Soft adjacency A[i,k]=exp(-||ci-ck||/σ), diagonal zero.
        centroids: (M, 2)
        """
        M = centroids.shape[0]
        if M == 0:
            return centroids.new_zeros((0, 0))
        diff = centroids.unsqueeze(0) - centroids.unsqueeze(1)  # (M,M,2)
        dist = torch.linalg.norm(diff, dim=-1)
        A = torch.exp(-dist / max(self.adj_sigma, 1e-6))
        A = A - torch.diag(torch.diag(A))
        return A

    @staticmethod
    def _row_l1_normalize(A: torch.Tensor) -> torch.Tensor:
        if A.numel() == 0:
            return A
        return A / A.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    def adjacency_geo_loss(
        self,
        log_P_region: Optional[torch.Tensor] = None,
        centroids0: Optional[torch.Tensor] = None,
        centroids1: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        enabled: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Adjacency preservation: neighbors stay neighbors across views.
        A0 ≈ P A1 P^T with soft centroid adjacency and soft matching P.
        """
        if not enabled or log_P_region is None or centroids0 is None or centroids1 is None:
            return torch.tensor(0.0, device=device or torch.device("cpu"))

        B = log_P_region.shape[0]
        total = torch.tensor(0.0, device=log_P_region.device)
        n = 0
        for b in range(B):
            log_P = log_P_region[b]
            M0 = log_P.shape[0] - 1
            M1 = log_P.shape[1] - 1
            if M0 < 2 or M1 < 2:
                continue
            c0 = centroids0[b] if centroids0.ndim == 3 else centroids0
            c1 = centroids1[b] if centroids1.ndim == 3 else centroids1
            if c0.shape[0] < 2 or c1.shape[0] < 2:
                continue
            P = torch.exp(torch.clamp(log_P[:M0, :M1], min=-100.0, max=0.0))
            A0 = self._soft_adjacency(c0[:M0])
            A1 = self._soft_adjacency(c1[:M1])
            A0_hat = P @ A1 @ P.transpose(0, 1)
            diff = self._row_l1_normalize(A0) - self._row_l1_normalize(A0_hat)
            total = total + (diff ** 2).sum()
            n += 1
        if n == 0:
            return torch.tensor(0.0, device=log_P_region.device)
        return total / float(n)

    def forward(
        self,
        log_P_region: torch.Tensor,
        log_P_line: Union[torch.Tensor, List[torch.Tensor], None] = None,
        gt_matches: Optional[List[List[Tuple[int, int]]]] = None,
        unmatched_0: Optional[List[List[int]]] = None,
        unmatched_1: Optional[List[List[int]]] = None,
        gt_line_pairs_flat: Optional[List[List[Tuple[int, int]]]] = None,
        unmatched_lines_flat_0: Optional[List[List[int]]] = None,
        unmatched_lines_flat_1: Optional[List[List[int]]] = None,
        line_parents0: Optional[List[torch.Tensor]] = None,
        line_parents1: Optional[List[torch.Tensor]] = None,
        centroids0: Optional[torch.Tensor] = None,
        centroids1: Optional[torch.Tensor] = None,
        has_line_ids: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gt_matches = gt_matches or [[]]
        unmatched_0 = unmatched_0 or [[]]
        unmatched_1 = unmatched_1 or [[]]
        gt_line_pairs_flat = gt_line_pairs_flat or [[]]
        unmatched_lines_flat_0 = unmatched_lines_flat_0 or [[]]
        unmatched_lines_flat_1 = unmatched_lines_flat_1 or [[]]

        loss_r = self._region_kl(log_P_region, gt_matches, unmatched_0, unmatched_1)
        if has_line_ids:
            loss_l = self._line_kl(
                log_P_line, gt_line_pairs_flat, unmatched_lines_flat_0, unmatched_lines_flat_1
            )
        else:
            loss_l = torch.tensor(0.0, device=log_P_region.device)

        loss_s = self.semantic_jsd_loss(
            log_P_line=log_P_line,
            line_parents0=line_parents0,
            line_parents1=line_parents1,
            num_regions0=log_P_region.shape[1] - 1,
            num_regions1=log_P_region.shape[2] - 1,
            device=log_P_region.device,
            enabled=has_line_ids and self.sem_weight > 0,
        )
        loss_g = self.adjacency_geo_loss(
            log_P_region=log_P_region,
            centroids0=centroids0,
            centroids1=centroids1,
            device=log_P_region.device,
            enabled=self.geo_weight > 0,
        )

        total = (
            self.region_weight * loss_r
            + self.line_weight * loss_l
            + self.sem_weight * loss_s
            + self.geo_weight * loss_g
        )
        metrics = {
            "loss": float(total.detach().item()) if total.numel() else 0.0,
            "loss_region": float(loss_r.detach().item()) if torch.is_tensor(loss_r) else 0.0,
            "loss_line": float(loss_l.detach().item()) if torch.is_tensor(loss_l) else 0.0,
            "loss_sem": float(loss_s.detach().item()) if torch.is_tensor(loss_s) else 0.0,
            "loss_geo": float(loss_g.detach().item()) if torch.is_tensor(loss_g) else 0.0,
        }
        if torch.isnan(total) or torch.isinf(total):
            total = torch.tensor(0.0, device=log_P_region.device, requires_grad=True)
        return total, metrics
