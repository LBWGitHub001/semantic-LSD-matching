"""Cross-view corners of a synthetic shape share one 3D point."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synthetic_shapes import build_descriptor_shapes_dataset, render_scene
from generate_dataset import DatasetGeneratorFactory


def _unproject(keypoint: dict, view: dict, intrinsics: np.ndarray, world_point: np.ndarray) -> np.ndarray:
    camera = view["R"] @ world_point + view["t"]
    depth = float(camera[2])
    x = (keypoint["x"] - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    y = (keypoint["y"] - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    return view["R"].T @ (np.array([x, y, depth], dtype=np.float32) - view["t"])


class TestSyntheticShapes(unittest.TestCase):
    def test_shared_corner_is_one_point_and_background_is_noisy(self) -> None:
        scene = render_scene("cube", 2, np.random.default_rng(0))
        self.assertGreaterEqual(len(scene["pairs"]), 1)
        pair = scene["pairs"][0]
        self.assertGreaterEqual(len(pair["matches"]), 3)
        view0 = scene["views"][pair["view0"]]
        view1 = scene["views"][pair["view1"]]
        self.assertGreater(float(view0["image"].std()), 8.0)
        world = scene["points_world"]
        for match in pair["matches"]:
            point = world[match["id"]]
            back0 = _unproject(view0["keypoints"][match["index0"]], view0, scene["K"], point)
            back1 = _unproject(view1["keypoints"][match["index1"]], view1, scene["K"], point)
            self.assertLess(float(np.linalg.norm(back0 - point)), 1e-3)
            self.assertLess(float(np.linalg.norm(back0 - back1)), 1e-3)

    def test_dataset_writes_preview_and_registers(self) -> None:
        factory = DatasetGeneratorFactory.create(task="descriptor_shapes")
        self.assertEqual([component.name for component in factory.components], ["descriptor_shapes"])
        with tempfile.TemporaryDirectory() as folder:
            stats = build_descriptor_shapes_dataset(folder, num_scenes=7, views_per_scene=2, seed=1)
            self.assertEqual(stats["scenes"], 7)
            self.assertGreater(stats["pairs"], 0)
            preview = cv2.imread(str(Path(folder) / "preview.jpg"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(preview)
            self.assertGreater(int((preview[:, :, 1] > 200).sum()), 10)
            split = json.loads((Path(folder) / "train_split.json").read_text(encoding="utf-8"))
            self.assertGreater(split["count"], 0)


if __name__ == "__main__":
    unittest.main()
