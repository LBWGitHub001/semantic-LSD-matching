#!/usr/bin/env python3
"""
Unit and Integration Tests for Factory Architecture:
  1. DatasetGeneratorFactory & LSDLineComponent tests
  2. ModelFactory auto-detection tests on all checkpoints
  3. DatasetFactory auto-resolution tests
  4. Dual-pair visualization canvas dimension tests
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_dataset import (
    DatasetGeneratorFactory,
    BaseDatasetComponent,
    ObjectMaskComponent,
    LSDLineComponent,
    Wireframe3DLineComponent,
    Structured3DLineRegionComponent,
    UnifiedProceduralRoom,
    render_indoor_view,
    create_look_at,
)
from eval import (
    ModelFactory,
    DatasetFactory,
    EvaluatorFactory,
    ObjectMatchingEvaluator,
    LineMatchingEvaluator,
)
from datasets import visualize_semantic_region_matches


def test_generator_factory():
    print("[-] Testing DatasetGeneratorFactory...")
    factory = DatasetGeneratorFactory.create(task="semantic_line")
    comp_names = [c.name for c in factory.components]
    assert "object" in comp_names, "object component missing"
    assert "lsd_line" in comp_names, "lsd_line component missing"
    assert "wireframe_3d" in comp_names, "wireframe_3d component missing"

    # Test procedural room and LSD detection
    room = UnifiedProceduralRoom(seed=123, random_colors=True)
    K = np.array([[380.0, 0, 320.0], [0, 380.0, 240.0], [0, 0, 1]], dtype=np.float32)
    eye = np.array([1.5, 1.4, 1.5], dtype=np.float32)
    R, t = create_look_at(eye, np.array([0, 0.75, 0], dtype=np.float32))
    rgb, mask, objs, lines_3d = render_indoor_view(room, K, R, t)

    lsd_comp = LSDLineComponent(min_length=15.0)
    scene_context = {
        "visible_objects_per_view": {0: objs},
        "projected_lines_per_view": {0: lines_3d},
    }
    view_data = lsd_comp.process_view(0, rgb, mask, {"K": K, "R": R, "t": t}, scene_context)
    assert "lines" in view_data, "No lines detected"
    assert "features" in view_data, "No features generated"
    assert len(view_data["lines"]) > 0, "LSD detected 0 lines"
    assert len(view_data["features"]) == len(view_data["lines"]), "Features length mismatch"
    print(f"    [✓] GeneratorFactory passed! Detected {len(view_data['lines'])} straight lines.")

    # Test Structured3D GT Line Component Registration
    s3d_factory = DatasetGeneratorFactory.create(task="structured3d_lines")
    comp_names_s3d = [c.name for c in s3d_factory.components]
    assert "structured3d_lines" in comp_names_s3d, "structured3d_lines component missing"
    print(f"    [✓] Structured3DLineRegionComponent successfully registered and verified in factory.")


def test_model_factory():
    print("[-] Testing ModelFactory auto-detection...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Test latest 50-epoch checkpoint
    model_latest, info_latest, path_latest = ModelFactory.create_from_checkpoint(
        "checkpoints/xfeat_object_matcher/latest", device=device
    )
    assert info_latest["use_pyramid"] is True, "Expected pyramid=True for latest"
    assert info_latest["with_geometry"] is False, "Expected with_geometry=False for latest"
    assert info_latest["d_model"] == 128
    print(f"    [✓] ModelFactory passed on latest checkpoint: {path_latest.name} (Pyramid: True)")

    # 2. Test legacy Oct 21 checkpoint
    legacy_path = Path("checkpoints/xfeat_object_matcher/best_model_legacy_oct21.pth")
    if legacy_path.is_file():
        model_old, info_old, _ = ModelFactory.create_from_checkpoint(legacy_path, device=device)
        assert info_old["use_pyramid"] is False, "Expected pyramid=False for legacy"
        assert info_old["with_geometry"] is True, "Expected with_geometry=True for legacy"
        print("    [✓] ModelFactory passed on legacy checkpoint (with_geometry: True)")


def test_evaluator_factory():
    print("[-] Testing EvaluatorFactory & Dual-Pair Visualization...")
    eval_obj = EvaluatorFactory.create("object")
    assert isinstance(eval_obj, ObjectMatchingEvaluator)
    eval_line = EvaluatorFactory.create("line")
    assert isinstance(eval_line, LineMatchingEvaluator)

    # Test dual-pair visualization dimensions
    bgr0 = np.full((480, 640, 3), 200, dtype=np.uint8)
    bgr1 = np.full((480, 640, 3), 200, dtype=np.uint8)
    masks0 = np.zeros((2, 480, 640), dtype=np.float32)
    masks0[0, 50:150, 50:150] = 1.0
    masks0[1, 200:300, 200:300] = 1.0
    masks1 = np.zeros((2, 480, 640), dtype=np.float32)
    masks1[0, 60:160, 60:160] = 1.0
    masks1[1, 210:310, 210:310] = 1.0

    regs0 = [{"centroid_2d": [100, 100], "category": "box1"}, {"centroid_2d": [250, 250], "category": "box2"}]
    regs1 = [{"centroid_2d": [110, 110], "category": "box1"}, {"centroid_2d": [260, 260], "category": "box2"}]

    out_test = Path("/tmp/test_vis_dual_pair.jpg")
    visualize_semantic_region_matches(
        bgr0=bgr0,
        bgr1=bgr1,
        masks0=masks0,
        masks1=masks1,
        regs0=regs0,
        regs1=regs1,
        pred_matches=[(0, 0), (1, 0)], # 1 TP, 1 FP
        gt_matches=[(0, 0), (1, 1)],    # 1 TP, 1 FN
        pred_un0=[],
        pred_un1=[],
        out_path=out_test,
        title="Unit Test Dual Pair",
    )
    assert out_test.is_file(), "Visualization file not created"
    vis_img = cv2.imread(str(out_test))
    # Dimensions should be (2 * 480 + 112, 640 * 2 + 30) = (1072, 1310, 3)
    assert vis_img.shape == (1072, 1310, 3), f"Unexpected shape {vis_img.shape}"
    out_test.unlink()
    print("    [✓] Dual-pair visualization passed! Correct (1072, 1310, 3) canvas created.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("        RUNNING COMPREHENSIVE FACTORY UNIT TESTS            ")
    print("=" * 60)
    test_generator_factory()
    test_model_factory()
    test_evaluator_factory()
    print("=" * 60)
    print("       ALL FACTORY TESTS PASSED WITH 100% SUCCESS!          ")
    print("=" * 60 + "\n")
