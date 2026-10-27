"""SemanticLineMatcher scaffold (region + line dual Sinkhorn; filled in later steps)."""

from __future__ import annotations

import torch.nn as nn


class SemanticLineMatcher(nn.Module):
    """
    Dual-output matcher: region Sinkhorn + line Sinkhorn.
    Line descriptors = Proj([along-line texture || region feature]).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._placeholder = nn.Identity()
        # Concrete modules added in steps 04–06.

    def forward(self, *args, **kwargs):  # pragma: no cover - filled in step05/06
        raise NotImplementedError("SemanticLineMatcher.forward is implemented in later steps")
