"""Unit tests for AlongLineTextureEncoder."""

from __future__ import annotations

import unittest

import torch

from models.semantic_line_matcher.line_encoder import AlongLineTextureEncoder, endpoints_from_5d


class TestAlongLineTextureEncoder(unittest.TestCase):
    def test_endpoints_from_5d_horizontal(self) -> None:
        # center (0.5,0.5), length 0.4, θ=0 → sin2=0, cos2=1
        lines = torch.tensor([[[0.5, 0.5, 0.4, 0.0, 1.0]]])
        p1, p2 = endpoints_from_5d(lines)
        self.assertAlmostEqual(float(p1[0, 0, 1]), 0.5, places=5)
        self.assertAlmostEqual(float(p2[0, 0, 1]), 0.5, places=5)
        self.assertAlmostEqual(float((p2 - p1)[0, 0, 0]), 0.4, places=5)

    def test_forward_shape_and_grad(self) -> None:
        enc = AlongLineTextureEncoder(in_channels=16, out_dim=32, num_samples=8)
        fmap = torch.randn(4, 16, 28, 28, requires_grad=True)
        lines = torch.zeros(4, 3, 5)
        lines[..., 0] = 0.5
        lines[..., 1] = 0.5
        lines[..., 2] = 0.3
        lines[..., 4] = 1.0
        mask = torch.tensor(
            [[True, True, False], [True, False, False], [True, True, True], [False, False, False]]
        )
        out = enc(fmap, lines, valid_mask=mask)
        self.assertEqual(out.shape, (4, 3, 32))
        out.sum().backward()
        self.assertIsNotNone(fmap.grad)


if __name__ == "__main__":
    unittest.main()
