"""A letter point mapped by the saved homography lands on the same id."""

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

from datasets.planar_letters import build_planar_letters_dataset, render_scene
from generate_dataset import DatasetGeneratorFactory


def _by_id(points: list[dict]) -> dict[int, dict]:
    return {int(point["id"]): point for point in points}


class TestPlanarLetters(unittest.TestCase):
    def test_homography_moves_a_point_onto_the_same_id(self) -> None:
        scene = render_scene("A", 3, np.random.default_rng(0))
        self.assertGreaterEqual(len(scene["pairs"]), 1)
        self.assertGreater(float(scene["views"][0]["image"].std()), 8.0)
        canonical = _by_id(scene["views"][0]["keypoints"])
        warped = scene["views"][1]
        homography = warped["H_from_canonical"]
        inverse = np.linalg.inv(homography)
        self.assertLess(float(np.max(np.abs(inverse @ homography - np.eye(3)))), 1e-6)
        for point in warped["keypoints"]:
            source = canonical[int(point["id"])]
            homo = np.array([source["x"], source["y"], 1.0])
            mapped = homography @ homo
            mapped = mapped[:2] / mapped[2]
            self.assertLess(abs(float(mapped[0]) - point["x"]), 1e-4)
            self.assertLess(abs(float(mapped[1]) - point["y"]), 1e-4)

    def test_dataset_writes_preview_and_registers(self) -> None:
        factory = DatasetGeneratorFactory.create(task="planar_letters")
        self.assertEqual([component.name for component in factory.components], ["planar_letters"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = {"n": 0}

            def fake_infer(bgrs):
                calls["n"] += len(bgrs)
                self.assertEqual(bgrs[0].shape[2], 3)
                return [
                    (np.full(image.shape[:2], 1.5, np.float32), np.full(image.shape[:2], 0.4, np.float32))
                    for image in bgrs
                ]

            stats = build_planar_letters_dataset(
                root, num_scenes=4, views_per_scene=3, seed=1, infer_batch=fake_infer
            )
            self.assertEqual(stats["scenes"], 4)
            self.assertGreater(stats["pairs"], 0)
            self.assertEqual(calls["n"], 12)
            field = np.load(root / "scene_00000" / "view_00.npz")
            self.assertEqual(set(field.files), {"df", "angle"})
            image = cv2.imread(str(root / "scene_00000" / "view_00.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(field["df"].shape, image.shape[:2])
            # 已有同名 npz 时不再推理。
            before = calls["n"]
            build_planar_letters_dataset(root, num_scenes=1, views_per_scene=3, seed=1, infer_batch=fake_infer)
            self.assertEqual(calls["n"], before)
            preview = cv2.imread(str(root / "preview.jpg"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(preview)
            self.assertGreater(int((preview[:, :, 1] > 200).sum()), 10)
            transforms = json.loads((root / "scene_00000" / "transforms.json").read_text(encoding="utf-8"))
            self.assertEqual(transforms["letter"], "A")
            self.assertEqual(len(transforms["views"][0]["H_to_canonical"]), 3)


if __name__ == "__main__":
    unittest.main()
