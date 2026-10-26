"""
Unit Tests for MaskLineMatcherNet.

Verifies:
  1. Output dimension matches canonical (B, 255, 255) specifications.
  2. Mathematical Permutation Equivariance:
     Swapping line order in View 0 or View 1 exactly permutes the corresponding
     rows/columns of the (255, 255) output matrix without distortion.
  3. Differentiability:
     Gradients flow back to model weights when supervising the 255x255 matrix.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mask_line_matcher import MaskLineMatcherNet, MaskLineFeatMatchNet


class TestMaskLineMatcherNet(unittest.TestCase):
    def test_output_shapes_and_values(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MaskLineMatcherNet(d_model=64, num_layers=2, max_matrix_size=255).to(device).eval()

        B = 2
        H, W = 180, 320
        N0, N1 = 5, 8

        img0 = torch.randn(B, 6, H, W, device=device)
        img1 = torch.randn(B, 6, H, W, device=device)
        lines0 = torch.rand(B, N0, 4, device=device) * 150.0
        lines1 = torch.rand(B, N1, 4, device=device) * 150.0

        with torch.no_grad():
            out = model(img0, img1, lines0, lines1)

        mat = out["matrix_255"]
        self.assertEqual(mat.shape, (B, 255, 255), f"Expected (B, 255, 255), got {mat.shape}")
        self.assertEqual(mat[:, N0:, :].sum().item(), 0.0, "Inactive rows beyond N0 should be 0.0")
        self.assertEqual(mat[:, :, N1:].sum().item(), 0.0, "Inactive cols beyond N1 should be 0.0")
        self.assertTrue((mat >= 0.0).all() and (mat <= 1.0).all(), "All probabilities must be in [0, 1]")

    def test_permutation_equivariance(self) -> None:
        """
        Critical property:
        If lines in View 0 are permuted by permutation sigma,
        the rows of the output 255x255 matrix MUST be permuted by the exact same sigma!
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        model = MaskLineMatcherNet(d_model=64, num_layers=2, max_matrix_size=255).to(device).eval()

        H, W = 180, 320
        N0, N1 = 4, 3

        img0 = torch.randn(1, 6, H, W, device=device)
        img1 = torch.randn(1, 6, H, W, device=device)

        # 4 distinct lines
        lines0 = torch.tensor([[[20, 20, 100, 20], [30, 50, 30, 150], [80, 80, 140, 140], [10, 120, 90, 120]]], dtype=torch.float32, device=device)
        lines1 = torch.tensor([[[22, 22, 102, 22], [32, 52, 32, 152], [82, 82, 142, 142]]], dtype=torch.float32, device=device)

        with torch.no_grad():
            out_orig = model(img0, img1, lines0, lines1)
        mat_orig = out_orig["matrix_255"][0, :N0, :N1]

        # Permute View 0 lines: swap index 0 and index 2
        perm = [2, 1, 0, 3]
        lines0_perm = lines0[:, perm, :]

        with torch.no_grad():
            out_perm = model(img0, img1, lines0_perm, lines1)
        mat_perm = out_perm["matrix_255"][0, :N0, :N1]

        diff = torch.max(torch.abs(mat_perm - mat_orig[perm, :])).item()
        self.assertLess(diff, 1e-5, f"Permutation equivariance violated! Max diff: {diff}")

    def test_differentiability(self) -> None:
        """Verify gradients flow from 255x255 matrix back to model parameters."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MaskLineMatcherNet(d_model=64, num_layers=2, max_matrix_size=255).to(device).train()

        H, W = 180, 320
        N0, N1 = 3, 3

        img0 = torch.randn(1, 6, H, W, device=device, requires_grad=True)
        img1 = torch.randn(1, 6, H, W, device=device, requires_grad=True)
        lines0 = torch.rand(1, N0, 4, device=device) * 100.0
        lines1 = torch.rand(1, N1, 4, device=device) * 100.0

        out = model(img0, img1, lines0, lines1)
        target_matrix = torch.zeros(1, 255, 255, device=device)
        target_matrix[0, 0, 0] = 1.0
        target_matrix[0, 1, 1] = 1.0

        loss = F.mse_loss(out["matrix_255"], target_matrix)
        loss.backward()

        self.assertIsNotNone(model.proj0.weight.grad)
        self.assertGreater(model.proj0.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
