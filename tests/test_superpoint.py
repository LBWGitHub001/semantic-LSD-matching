"""Descriptor-only SuperPoint: 5 input channels and a 256-D head."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_loss, build_model, model_family


class TestSuperPoint(unittest.TestCase):
    def test_forward_shapes_and_unit_descriptors(self) -> None:
        model = build_model("SuperPoint")
        image = torch.rand(2, 5, 64, 64)
        pred = model(image)
        self.assertNotIn("semi", pred)
        self.assertNotIn("score", pred)
        self.assertFalse(hasattr(model, "convPb"))
        self.assertEqual(model.conv1a.in_channels, 5)
        self.assertEqual(tuple(pred["desc"].shape), (2, 256, 8, 8))
        self.assertEqual(tuple(pred["descriptor"].shape), (2, 256, 64, 64))
        norm = pred["descriptor"].norm(dim=1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))
        self.assertEqual(model_family("superpoint"), "point")

    def test_loss_backward(self) -> None:
        model = build_model("SuperPoint")
        loss_fn = build_loss("SuperPoint")
        image0 = torch.rand(1, 5, 64, 64)
        image1 = torch.rand(1, 5, 64, 64)
        pred0 = model(image0)
        pred1 = model(image1)
        homography = torch.eye(3).unsqueeze(0)
        out = loss_fn(pred0, pred1, homography)
        out["loss"].backward()
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertIsNotNone(model.convDb.weight.grad)
        self.assertNotIn("lp", out)


if __name__ == "__main__":
    unittest.main()
