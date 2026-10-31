"""Point-match samples pack five channels and the figure marks wrong neighbors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.planar_letters import build_planar_letters_dataset
from datasets.point_match_dataset import PointMatchDataset
from visualization.point_matches import match_accuracy, save_point_match_figure


def _fake_infer(bgrs):
    return [
        (np.full(image.shape[:2], 1.5, np.float32), np.full(image.shape[:2], 0.4, np.float32))
        for image in bgrs
    ]


class TestPointMatchTrainingPieces(unittest.TestCase):
    def test_dataset_and_figure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            build_planar_letters_dataset(root, num_scenes=1, views_per_scene=2, seed=0, infer_batch=_fake_infer)
            sample = PointMatchDataset(root, split="train")[0]
            self.assertEqual(tuple(sample["image0"].shape), (5, 240, 320))
            # 检验 channel 3 (归一化对数距离场) 在 [0, 1] 范围内
            dist_ch = sample["image0"][3].numpy()
            self.assertTrue(np.all(dist_ch >= 0.0) and np.all(dist_ch <= 1.0))
            self.assertEqual(sample["points0"].shape, sample["points1"].shape)
            self.assertGreater(sample["points0"].shape[0], 3)
            descriptor = torch.zeros(1, 8, 240, 320)
            descriptor[0, 0] = 1
            accuracy, choice = match_accuracy(descriptor, descriptor, sample["points0"], sample["points1"])
            self.assertEqual(choice.shape[0], sample["points0"].shape[0])
            self.assertGreaterEqual(accuracy, 0.0)
            figure = root / "match.jpg"
            save_point_match_figure(
                figure,
                sample["bgr0"].numpy(),
                sample["bgr1"].numpy(),
                sample["points0"].numpy(),
                sample["points1"].numpy(),
                choice.numpy(),
            )
            self.assertTrue(figure.is_file())
            self.assertGreater(figure.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
