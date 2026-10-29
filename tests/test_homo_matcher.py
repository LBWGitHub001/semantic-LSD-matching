"""Ridge geometry, homography head, and factory registration."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_dataset import DatasetGeneratorFactory
from models import build_loss, build_model, model_family
from models.FCN_field.model import weighted_dlt
from models.FCN_field.ridge import extract_ridge_lines, pack_input


class TestFCNField(unittest.TestCase):
    def test_ridge_follows_distance_field(self) -> None:
        height, width = 80, 160
        distance = np.full((height, width), 5.0, dtype=np.float32)
        distance[40, 10:150] = 0.0
        angle = np.zeros((height, width), dtype=np.float32)
        endpoints, samples = extract_ridge_lines(distance, angle, min_length=20, min_pixels=8)
        self.assertGreater(len(endpoints), 0)
        self.assertEqual(samples.shape[1], 16)
        ys = endpoints[:, :, 1]
        self.assertTrue(np.all(np.abs(ys - 40) <= 1.5))

    def test_pack_input_has_five_channels(self) -> None:
        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        df = np.ones((8, 8), dtype=np.float32)
        angle = np.zeros((8, 8), dtype=np.float32)
        packed = pack_input(bgr, df, angle)
        self.assertEqual(packed.shape, (5, 8, 8))

    def test_identity_correspondences_give_identity(self) -> None:
        source = torch.tensor([[[10.0, 10.0], [80.0, 10.0], [80.0, 60.0], [10.0, 60.0], [40.0, 30.0]]])
        weight = torch.ones(1, 5)
        homography = weighted_dlt(source, source.clone(), weight)
        ones = torch.ones(source.shape[0], source.shape[1], 1)
        homo = torch.cat([source, ones], dim=-1)
        mapped = torch.einsum("bij,bnj->bni", homography, homo)
        mapped = mapped[..., :2] / mapped[..., 2:3]
        self.assertTrue(torch.allclose(mapped, source, atol=0.05), mapped[0])
        self.assertAlmostEqual(float(homography[0, 2, 2]), 1.0, places=4)

    def test_forward_and_loss_backward(self) -> None:
        torch.manual_seed(0)
        model = build_model("FCN_field", grid_size=4, feat_dim=32)
        loss_fn = build_loss("FCN_field", lambda1=1.0, lambda2=1.0)
        image0 = torch.rand(2, 5, 64, 64)
        image1 = torch.rand(2, 5, 64, 64)
        pred = model(image0, image1)
        self.assertEqual(tuple(pred["velocity12"].shape), (2, 2, 64, 64))
        self.assertEqual(tuple(pred["magnitude12"].shape), (2, 64, 64))
        self.assertNotIn("H12", pred)
        ridges = torch.zeros(2, 2, 8, 2)
        ridges[..., 0] = torch.linspace(4, 50, 8)
        ridges[..., 1] = 20
        batch = {
            "ridges0": ridges,
            "ridge_mask": torch.tensor([[True, False], [True, True]]),
            "depth0": torch.ones(2, 1, 64, 64),
            "depth1": torch.ones(2, 1, 64, 64) * 2,
        }
        out = loss_fn(pred, batch)
        out["loss"].backward()
        self.assertTrue(out["loss"].requires_grad)
        self.assertEqual(tuple(pred["velocity12"].shape), image0.shape[:1] + (2,) + image0.shape[-2:])
        self.assertIsNotNone(model.velocity_head.weight.grad)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(model_family("FCN_field"), "homography")

    def test_dataset_factory_registration(self) -> None:
        factory = DatasetGeneratorFactory.create(task="FCN_field")
        self.assertEqual([component.name for component in factory.components], ["FCN_field"])
        from evaluation import EvaluatorFactory

        evaluator = EvaluatorFactory.create("homography")
        self.assertEqual(evaluator.__class__.__name__, "HomographyEvaluator")


if __name__ == "__main__":
    unittest.main()
