"""SemanticLineMatchingLoss scaffold (four-term loss; filled in steps 07–09)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticLineMatchingLoss(nn.Module):
    """Region KL + line KL + semantic JSD + adjacency geo."""

    def __init__(
        self,
        dustbin_weight: float = 0.2,
        region_weight: float = 1.0,
        line_weight: float = 1.0,
        sem_weight: float = 0.1,
        geo_weight: float = 0.2,
        adj_sigma: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dustbin_weight = float(dustbin_weight)
        self.region_weight = float(region_weight)
        self.line_weight = float(line_weight)
        self.sem_weight = float(sem_weight)
        self.geo_weight = float(geo_weight)
        self.adj_sigma = float(adj_sigma)

    def forward(self, *args, **kwargs):  # pragma: no cover - filled in steps 07–09
        raise NotImplementedError("SemanticLineMatchingLoss.forward is implemented in later steps")
