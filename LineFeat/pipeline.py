"""
LineFeat High-Level Inference Pipeline & API.

Provides an easy-to-use, unified Python interface for cross-image line matching:
  - Supports image file paths or in-memory numpy BGR arrays
  - Automatic 2D line detection via LSD (or user-provided lines)
  - End-to-end Line-Transformer matching with trained weights
  - Built-in visualization and export utilities
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from LineFeat.matcher import LineMatcher
from LineFeat.model import LineMatcherNet


@dataclass
class LineMatchResult:
    """Structured result of cross-image line matching."""

    lines0: np.ndarray  # (N0, 4) [x1, y1, x2, y2]
    lines1: np.ndarray  # (N1, 4)
    matches: list[tuple[int, int, float]]  # List of (idx0, idx1, similarity_score)
    matched_lines0: np.ndarray  # (K, 4) Coordinates of matched lines in View 0
    matched_lines1: np.ndarray  # (K, 4) Coordinates of matched lines in View 1
    match_scores: np.ndarray  # (K,) Similarity score for each match
    image0: np.ndarray  # (H0, W0, 3) BGR
    image1: np.ndarray  # (H1, W1, 3) BGR

    def __len__(self) -> int:
        return len(self.matches)

    def visualize(
        self,
        max_draw: int = 50,
        show_connecting_lines: bool = True,
    ) -> np.ndarray:
        """Render side-by-side matching visualization image."""
        return LineMatcher.draw_matches(
            self.image0,
            self.image1,
            self.lines0,
            self.lines1,
            self.matches,
            max_draw=max_draw,
            show_connecting_lines=show_connecting_lines,
        )

    def save_visualization(
        self,
        path: str | Path,
        max_draw: int = 50,
        show_connecting_lines: bool = True,
    ) -> Path:
        """Render and save visualization image to disk."""
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        vis = self.visualize(max_draw=max_draw, show_connecting_lines=show_connecting_lines)
        cv2.imwrite(str(out_path), vis)
        return out_path


class LineMatchPipeline:
    """
    High-level end-to-end line matching pipeline.

    Example:
        >>> from LineFeat import LineMatchPipeline
        >>> pipeline = LineMatchPipeline("checkpoints/linefeat/best_model.pth")
        >>> result = pipeline.match("image1.jpg", "image2.jpg")
        >>> print(f"Found {len(result)} matches!")
        >>> result.save_visualization("outputs/matches.jpg")
    """

    def __init__(
        self,
        weights: str | Path = "checkpoints/linefeat/best_model.pth",
        device: str | torch.device | None = None,
        similarity_threshold: float = 0.85,
        d_model: int = 128,
        num_layers: int = 4,
    ) -> None:
        """
        Args:
            weights: Path to trained LineMatcherNet checkpoint.
            device: 'cuda', 'cpu', or None (auto-detect).
            similarity_threshold: Minimum cosine similarity threshold for matching.
            d_model: Feature dimension of LineMatcherNet.
            num_layers: Number of Transformer layers.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.similarity_threshold = similarity_threshold

        # Model
        self.model = LineMatcherNet(
            d_model=d_model,
            num_layers=num_layers,
            nhead=4,
            freeze_backbone=True,
        ).to(self.device).eval()

        # Load weights
        weights_path = Path(weights).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state_dict)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _prepare_image(self, img_input: str | Path | np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        if isinstance(img_input, (str, Path)):
            path = Path(img_input)
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image from path: {path}")
        elif isinstance(img_input, np.ndarray):
            bgr = img_input.copy()
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}. Expected str, Path, or np.ndarray.")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # (H, W, 3) -> (1, 3, H, W) normalized
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        t_norm = (t - self.mean) / self.std
        return bgr, t_norm

    @torch.no_grad()
    def match(
        self,
        image0: str | Path | np.ndarray,
        image1: str | Path | np.ndarray,
        lines0: np.ndarray,
        lines1: np.ndarray,
        threshold: float | None = None,
    ) -> LineMatchResult:
        """
        Pure Line Matching between image0 and image1 using pre-detected lines.

        Args:
            image0: File path or BGR numpy array of View 0.
            image1: File path or BGR numpy array of View 1.
            lines0: (N0, 4) line segment coordinates [x1, y1, x2, y2].
            lines1: (N1, 4) line segment coordinates [x1, y1, x2, y2].
            threshold: Optional override for cosine similarity threshold.

        Returns:
            LineMatchResult object containing matches and visualization methods.
        """
        if lines0 is None or lines1 is None:
            raise ValueError("lines0 and lines1 must be provided as (N, 4) line segment coordinates.")

        lines0 = np.asarray(lines0, dtype=np.float32)
        lines1 = np.asarray(lines1, dtype=np.float32)
        thresh = threshold if threshold is not None else self.similarity_threshold

        bgr0, t0 = self._prepare_image(image0)
        bgr1, t1 = self._prepare_image(image1)

        N0, N1 = len(lines0), len(lines1)
        if N0 == 0 or N1 == 0:
            return LineMatchResult(
                lines0=lines0,
                lines1=lines1,
                matches=[],
                matched_lines0=np.zeros((0, 4), dtype=np.float32),
                matched_lines1=np.zeros((0, 4), dtype=np.float32),
                match_scores=np.zeros((0,), dtype=np.float32),
                image0=bgr0,
                image1=bgr1,
            )

        l0_t = torch.from_numpy(lines0).unsqueeze(0).to(self.device)
        l1_t = torch.from_numpy(lines1).unsqueeze(0).to(self.device)

        # Forward pass
        out = self.model(t0, t1, l0_t, l1_t)
        scores = out["scores"][0].cpu().numpy()  # (N0, N1)
        log_P = out["log_assignment"][0].cpu().numpy()  # (N0+1, N1+1)

        P_mat = log_P[:N0, :N1]
        best1 = np.argmax(P_mat, axis=1)  # (N0,)
        best0 = np.argmax(P_mat, axis=0)  # (N1,)

        matches: list[tuple[int, int, float]] = []
        for i in range(N0):
            j = int(best1[i])
            if int(best0[j]) != i:
                continue

            sim = float(scores[i, j])
            if sim < thresh:
                continue

            matches.append((i, j, sim))

        matches.sort(key=lambda m: m[2], reverse=True)

        if len(matches) > 0:
            matched_l0 = lines0[[m[0] for m in matches]]
            matched_l1 = lines1[[m[1] for m in matches]]
            matched_s = np.array([m[2] for m in matches], dtype=np.float32)
        else:
            matched_l0 = np.zeros((0, 4), dtype=np.float32)
            matched_l1 = np.zeros((0, 4), dtype=np.float32)
            matched_s = np.zeros((0,), dtype=np.float32)

        return LineMatchResult(
            lines0=lines0,
            lines1=lines1,
            matches=matches,
            matched_lines0=matched_l0,
            matched_lines1=matched_l1,
            match_scores=matched_s,
            image0=bgr0,
            image1=bgr1,
        )
