"""
Unit tests for the refactored visualization suite.
Verifies matching visualizer, architecture visualizer, and curve visualizer.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import cv2

from visualization import (
    blend_mask,
    draw_dashed_line,
    draw_mask_boundary,
    plot_training_trajectory,
    render_model_architecture,
    visualize_line_matches,
    visualize_object_matches,
    visualize_semantic_region_matches,
)
from visualization.matching_visualizer import save_pair_visualization, segment_from_line5d


class TestVisualizationSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_primitives(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:60, 20:60] = 1

        blended = blend_mask(img.copy(), mask, (0, 255, 0), alpha=0.5)
        self.assertEqual(blended.shape, (100, 100, 3))
        self.assertTrue(np.any(blended > 0))

        # Contours
        contour_img = img.copy()
        draw_mask_boundary(contour_img, mask, (255, 0, 0), thickness=2)
        self.assertTrue(np.any(contour_img > 0))

        # Dashed line
        line_img = img.copy()
        draw_dashed_line(line_img, (10, 10), (90, 90), (0, 255, 255), thickness=1, dash_length=4)
        self.assertTrue(np.any(line_img > 0))

    def test_semantic_region_visualizer(self) -> None:
        bgr0 = np.full((120, 160, 3), 100, dtype=np.uint8)
        bgr1 = np.full((120, 160, 3), 100, dtype=np.uint8)
        masks0 = np.zeros((2, 120, 160), dtype=np.float32)
        masks1 = np.zeros((2, 120, 160), dtype=np.float32)
        masks0[0, 10:40, 10:40] = 1.0
        masks0[1, 50:80, 50:80] = 1.0
        masks1[0, 12:42, 12:42] = 1.0
        masks1[1, 52:82, 52:82] = 1.0

        regs0 = [
            {"centroid_2d": [25, 25], "category": "table", "label": "table_0"},
            {"centroid_2d": [65, 65], "category": "chair", "label": "chair_1"},
        ]
        regs1 = [
            {"centroid_2d": [27, 27], "category": "table", "label": "table_0"},
            {"centroid_2d": [67, 67], "category": "chair", "label": "chair_1"},
        ]

        out_file = self.tmp_path / "semantic_vis_test.jpg"
        visualize_semantic_region_matches(
            bgr0=bgr0,
            bgr1=bgr1,
            masks0=masks0,
            masks1=masks1,
            regs0=regs0,
            regs1=regs1,
            pred_matches=[(0, 0)],
            gt_matches=[(0, 0), (1, 1)],
            pred_un0=[1],
            pred_un1=[1],
            out_path=out_file,
            title="Unit Test Matcher",
        )
        self.assertTrue(out_file.is_file())
        self.assertGreater(out_file.stat().st_size, 1000)

    def test_object_matches_visualizer(self) -> None:
        img0 = np.full((100, 100, 3), 200, dtype=np.uint8)
        img1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        objs0 = [{"bbox": [10, 10, 40, 40], "category": "sofa", "centroid": [25, 25]}]
        objs1 = [{"bbox": [15, 15, 45, 45], "category": "sofa", "centroid": [30, 30]}]

        out_file = self.tmp_path / "object_vis_test.jpg"
        visualize_object_matches(
            raw_img0=img0,
            raw_img1=img1,
            objs0=objs0,
            objs1=objs1,
            pred_matches=[(0, 0)],
            gt_matches=[(0, 0)],
            pred_un0=[],
            pred_un1=[],
            out_path=out_file,
        )
        self.assertTrue(out_file.is_file())
        self.assertGreater(out_file.stat().st_size, 1000)

    def test_line_matches_visualizer(self) -> None:
        bgr0 = np.full((80, 100, 3), 40, dtype=np.uint8)
        bgr1 = np.full((80, 100, 3), 50, dtype=np.uint8)
        # Square bbox matches the 112 patch with no letterbox offset.
        seg = segment_from_line5d([0.5, 0.5, 0.4, 0.0, 1.0], [0, 0, 112, 112], patch_size=112)
        self.assertIsNotNone(seg)
        self.assertAlmostEqual(seg[0][1], 56.0, places=4)
        self.assertAlmostEqual(seg[1][1], 56.0, places=4)
        self.assertAlmostEqual(seg[0][0], 33.6, places=4)
        self.assertAlmostEqual(seg[1][0], 78.4, places=4)

        lines = np.zeros((1, 2, 5), dtype=np.float32)
        lines[0, 0] = [0.5, 0.5, 0.4, 0.0, 1.0]
        mask = np.zeros((1, 2), dtype=bool)
        mask[0, 0] = True
        bboxes = [[10.0, 8.0, 70.0, 68.0]]

        out_file = self.tmp_path / "line_vis_test.jpg"
        visualize_line_matches(
            bgr0=bgr0,
            bgr1=bgr1,
            patch_lines0=lines,
            patch_lines1=lines.copy(),
            line_mask0=mask,
            line_mask1=mask.copy(),
            bboxes0=bboxes,
            bboxes1=bboxes,
            pred_matches=[(0, 0)],
            gt_matches=[(0, 0)],
            out_path=out_file,
            title="Line Unit Test",
            has_gt=True,
        )
        self.assertTrue(out_file.is_file())
        self.assertGreater(out_file.stat().st_size, 1000)
        canvas = cv2.imread(str(out_file))
        header_h, gap, pad = 50, 30, 12
        expected_h = header_h + 80 + header_h + 80 + pad
        expected_w = 100 + 100 + gap
        self.assertEqual(canvas.shape[:2], (expected_h, expected_w))

        no_gt = self.tmp_path / "line_vis_nogt.jpg"
        visualize_line_matches(
            bgr0=bgr0,
            bgr1=bgr1,
            patch_lines0=lines,
            patch_lines1=lines,
            line_mask0=mask,
            line_mask1=mask,
            bboxes0=bboxes,
            bboxes1=bboxes,
            pred_matches=[(0, 0)],
            gt_matches=[(0, 0)],
            out_path=no_gt,
            has_gt=False,
        )
        self.assertTrue(no_gt.is_file())

    def test_save_pair_visualization_dispatches_line_canvas(self) -> None:
        image = np.full((40, 50, 3), 30, dtype=np.uint8)
        lines = np.zeros((1, 1, 5), dtype=np.float32)
        lines[0, 0] = [0.5, 0.5, 0.3, 0.0, 1.0]
        mask = np.ones((1, 1), dtype=bool)
        prob = np.zeros((2, 2), dtype=np.float32)
        prob[0, 0] = 0.9
        prob[1, 1] = 0.9
        batch = {
            "raw_img0": image,
            "raw_img1": image.copy(),
            "masks0": np.zeros((1, 40, 50), dtype=np.float32),
            "masks1": np.zeros((1, 40, 50), dtype=np.float32),
            "patch_lines0": lines,
            "patch_lines1": lines.copy(),
            "line_mask0": mask,
            "line_mask1": mask.copy(),
            "bboxes0": [[0.0, 0.0, 40.0, 40.0]],
            "bboxes1": [[0.0, 0.0, 40.0, 40.0]],
            "gt_matches": [[(0, 0)]],
            "gt_line_pairs_flat": [[(0, 0)]],
            "has_line_ids": True,
        }
        out_file = self.tmp_path / "dispatch_line.jpg"
        wrote = save_pair_visualization(
            batch,
            {"P_line": prob, "prob_matrix": prob},
            out_file,
            title="dispatch",
            pred_matches=[(0, 0)],
            pred_un0=[],
            pred_un1=[],
        )
        self.assertTrue(wrote)
        self.assertTrue(out_file.is_file())

    def test_curve_visualizer(self) -> None:
        history = [
            {"epoch": 1, "train_loss": 1.2, "val_f1": 60.0, "val_prec": 58.0, "val_rec": 62.0},
            {"epoch": 2, "train_loss": 0.8, "val_f1": 72.0, "val_prec": 70.0, "val_rec": 74.0},
        ]
        out_file = self.tmp_path / "curve_test.png"
        plot_training_trajectory(history, out_file, title="Test Trajectory")
        self.assertTrue(out_file.is_file())
        self.assertGreater(out_file.stat().st_size, 1000)

    def test_architecture_visualizer(self) -> None:
        out_file = self.tmp_path / "arch_test.png"
        render_model_architecture("xfeat_line_matcher", out_path=out_file, dpi=80)
        self.assertTrue(out_file.is_file())
        self.assertGreater(out_file.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
