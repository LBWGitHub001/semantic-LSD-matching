"""SuperPoint with descriptor head and endpoint detector head."""

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
        self.assertIn("semi", pred)
        self.assertIn("heatmap", pred)
        self.assertTrue(hasattr(model, "convPb"))
        self.assertTrue(hasattr(model, "convPa"))
        self.assertEqual(model.conv1a.in_channels, 5)
        self.assertEqual(tuple(pred["semi"].shape), (2, 65, 8, 8))
        self.assertEqual(tuple(pred["heatmap"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(pred["desc"].shape), (2, 256, 8, 8))
        self.assertEqual(tuple(pred["descriptor"].shape), (2, 256, 64, 64))
        norm = pred["descriptor"].norm(dim=1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))
        self.assertEqual(model_family("superpoint"), "point")

    def test_descriptor_loss_backward_compat(self) -> None:
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
        self.assertIn("ld", out)

    def test_joint_line_geometric_loss(self) -> None:
        model = build_model("SuperPoint")
        loss_fn = build_loss("SuperPoint")
        image0 = torch.rand(1, 5, 64, 64)
        image1 = torch.rand(1, 5, 64, 64)
        # 模拟真实的对数距离场 [0, 1] 与角度场 [0, 1]
        image0[:, 3:4] = torch.rand(1, 1, 64, 64)
        image0[:, 4:5] = torch.rand(1, 1, 64, 64)
        image1[:, 3:4] = torch.rand(1, 1, 64, 64)
        image1[:, 4:5] = torch.rand(1, 1, 64, 64)

        pred0 = model(image0)
        pred1 = model(image1)
        homography = torch.eye(3).unsqueeze(0)
        points0 = torch.tensor([[12.0, 18.0], [40.0, 48.0]])
        points1 = torch.tensor([[12.0, 18.0], [40.0, 48.0]])

        out = loss_fn(
            pred0,
            pred1,
            homography,
            image0=image0,
            image1=image1,
            points0=points0,
            points1=points1,
        )
        for key in ("loss", "ld", "l_det", "l_orphan", "l_len", "l_prem"):
            self.assertIn(key, out)
            self.assertTrue(torch.isfinite(out[key]))

        out["loss"].backward()
        # 验证梯度同时回传至描述子分支与端点检测分支
        self.assertIsNotNone(model.convDb.weight.grad)
        self.assertIsNotNone(model.convPb.weight.grad)
        self.assertTrue(torch.isfinite(model.convPb.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
