"""SuperPoint with descriptor head and endpoint detector head."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import build_loss, build_model, model_family
from models.superpoint.loss import (
    SuperPointDescriptorLoss,
    SuperPointDetectorLoss,
    SuperPointJointLoss,
    SuperPointLoss,
    descriptor_hinge_loss,
    detector_bce_loss,
    orphan_endpoint_loss,
    max_length_line_loss,
    premature_ray_probing_loss,
    extract_candidate_endpoints,
    interpolate_and_sample_lines,
    compute_line_validity_matrix,
)


class TestSuperPoint(unittest.TestCase):
    def test_forward_shapes_and_unit_descriptors(self) -> None:
        model = build_model("SuperPoint")
        image = torch.rand(2, 5, 64, 64)
        pred = model(image)
        self.assertIn("semi", pred)
        self.assertIn("heatmap", pred)
        self.assertIn("endpoint_mask", pred)
        self.assertIn("mask", pred)
        self.assertTrue(hasattr(model, "convPb"))
        self.assertTrue(hasattr(model, "convPa"))
        self.assertEqual(model.conv1a.in_channels, 5)
        self.assertEqual(tuple(pred["semi"].shape), (2, 65, 8, 8))
        self.assertEqual(tuple(pred["heatmap"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(pred["endpoint_mask"].shape), (2, 1, 64, 64))
        # 验证概率掩膜数值严格在 [0, 1] 之间
        self.assertTrue((pred["endpoint_mask"] >= 0.0).all())
        self.assertTrue((pred["endpoint_mask"] <= 1.0).all())
        self.assertEqual(tuple(pred["desc"].shape), (2, 256, 8, 8))
        self.assertEqual(tuple(pred["descriptor"].shape), (2, 256, 64, 64))
        norm = pred["descriptor"].norm(dim=1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-4))
        self.assertEqual(model_family("superpoint"), "point")

        # 验证 extract_endpoints
        pts, sc = model.extract_endpoints(pred["endpoint_mask"], threshold=0.01, top_k=16)
        self.assertEqual(tuple(pts.shape), (2, 16, 2))
        self.assertEqual(tuple(sc.shape), (2, 16))

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

    def test_individual_loss_member_methods(self) -> None:
        """测试 SuperPointLoss 类的各分项损失成员方法与辅助几何算子。"""
        loss_fn = SuperPointLoss()
        b, h, w = 1, 64, 64
        heatmap = torch.rand(b, 1, h, w, requires_grad=True)
        df_map = torch.rand(b, 1, h, w)
        af_map = torch.rand(b, 1, h, w)

        # 1. 类成员方法: 候选端点提取
        scores, points = loss_fn.extract_candidate_endpoints(heatmap)
        self.assertEqual(tuple(scores.shape), (b, 16))
        self.assertEqual(tuple(points.shape), (b, 16, 2))

        # 2. 类成员方法: 沿线插值与采样
        s_df, s_af, dist, diff, p1, p2 = loss_fn.interpolate_and_sample_lines(points, df_map, af_map)
        self.assertEqual(tuple(s_df.shape), (b, 16, 16, 10))
        self.assertEqual(tuple(dist.shape), (b, 16, 16))

        # 3. 类成员方法: 合法度矩阵判定
        v_matrix = loss_fn.compute_line_validity_matrix(s_df, s_af, dist)
        self.assertEqual(tuple(v_matrix.shape), (b, 16, 16))
        self.assertTrue((v_matrix >= 0.0).all() and (v_matrix <= 1.0).all())

        # 4. 类成员方法: orphan_endpoint_loss
        l_orphan = loss_fn.orphan_endpoint_loss(scores, v_matrix)
        self.assertTrue(torch.isfinite(l_orphan))

        # 5. 类成员方法: max_length_line_loss
        l_len = loss_fn.max_length_line_loss(scores, dist, v_matrix, h, w)
        self.assertTrue(torch.isfinite(l_len))

        # 6. 类成员方法: premature_ray_probing_loss
        l_prem = loss_fn.premature_ray_probing_loss(scores, dist, diff, p1, p2, v_matrix, df_map)
        self.assertTrue(torch.isfinite(l_prem))

        # 7. 类成员方法: detector_loss
        points_gt = torch.tensor([[10.0, 15.0], [30.0, 45.0]])
        l_det = loss_fn.detector_loss(heatmap, points_gt)
        self.assertTrue(torch.isfinite(l_det))

        # 8. 类成员方法: descriptor_loss
        desc0 = torch.randn(b, 256, 8, 8)
        desc1 = torch.randn(b, 256, 8, 8)
        homo = torch.eye(3).unsqueeze(0)
        l_desc = loss_fn.descriptor_loss(desc0, desc1, homo)
        self.assertTrue(torch.isfinite(l_desc))

        # 9. 组合求导验证梯度流向 heatmap
        combined = l_orphan + 0.5 * l_len + 0.5 * l_prem + l_det
        combined.backward()
        self.assertIsNotNone(heatmap.grad)
        self.assertTrue(torch.isfinite(heatmap.grad).all())

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

    def test_point_matching_evaluator_metrics_and_visualize(self) -> None:
        """测试 PointMatchingEvaluator 评测与可视化诊断大图生成。"""
        import tempfile
        from evaluation.evaluators import PointMatchingEvaluator

        model = build_model("SuperPoint")
        device = torch.device("cpu")
        evaluator = PointMatchingEvaluator()

        # 构造模拟测试 batch
        mock_batch = {
            "image0": torch.rand(1, 5, 64, 64),
            "image1": torch.rand(1, 5, 64, 64),
            "homography": torch.eye(3).unsqueeze(0),
            "points0": torch.tensor([[10.0, 15.0], [25.0, 30.0]]),
            "points1": torch.tensor([[10.0, 15.0], [25.0, 30.0]]),
            "bgr0": torch.zeros((64, 64, 3), dtype=torch.uint8),
            "bgr1": torch.zeros((64, 64, 3), dtype=torch.uint8),
            "pair_id": "test_scene_v0_v1",
        }
        loader = [mock_batch]

        # 验证 evaluate 输出包含四大维度关键指标
        metrics = evaluator.evaluate(model, loader, device)
        for key in (
            "loss", "loss_desc", "loss_det", "loss_orphan", "loss_len", "loss_prem",
            "match_acc", "mma_1px", "mma_2px", "mma_3px",
            "corner_repeatability_3_0px", "valid_lines_per_view", "mean_line_length",
        ):
            self.assertIn(key, metrics)
            self.assertTrue(isinstance(metrics[key], (float, int)))

        # 验证 visualize 生成诊断图
        with tempfile.TemporaryDirectory() as tmp_dir:
            vis_paths = evaluator.visualize(model, loader, device, Path(tmp_dir), num_vis=1)
            self.assertEqual(len(vis_paths), 4)  # 生成 detection, matching, pca 与 composite
            for vp in vis_paths:
                self.assertTrue(vp.is_file())
                self.assertGreater(vp.stat().st_size, 1000)

    def test_train_stage_parameter_freezing(self) -> None:
        """测试模型根据阶段自动冻结/激活目标分支参数。"""
        model = build_model("SuperPoint")

        # 1. 描述子阶段
        params_desc = model.get_trainable_parameters("descriptor")
        self.assertTrue(model.convDa.weight.requires_grad)
        self.assertTrue(model.convDb.weight.requires_grad)
        self.assertFalse(model.convPa.weight.requires_grad)
        self.assertFalse(model.convPb.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in params_desc))

        # 2. 检测器阶段
        params_det = model.get_trainable_parameters("detector")
        self.assertFalse(model.convDa.weight.requires_grad)
        self.assertFalse(model.convDb.weight.requires_grad)
        self.assertTrue(model.convPa.weight.requires_grad)
        self.assertTrue(model.convPb.weight.requires_grad)
        self.assertTrue(all(p.requires_grad for p in params_det))

        # 3. 联合阶段
        params_joint = model.get_trainable_parameters("joint")
        self.assertTrue(model.convDa.weight.requires_grad)
        self.assertTrue(model.convPa.weight.requires_grad)
        self.assertEqual(len(params_joint), len(list(model.parameters())))

    def test_decoupled_loss_classes_forward_backward(self) -> None:
        """测试 SuperPointDescriptorLoss 与 SuperPointDetectorLoss 的独立前向与反向传播。"""
        model = build_model("SuperPoint")
        desc_loss = SuperPointDescriptorLoss()
        det_loss = SuperPointDetectorLoss()

        image0 = torch.rand(1, 5, 64, 64)
        image1 = torch.rand(1, 5, 64, 64)
        pred0 = model(image0)
        pred1 = model(image1)
        homography = torch.eye(3).unsqueeze(0)

        # 1. 独立描述子损失测试
        out_desc = desc_loss(pred0, pred1, homography)
        self.assertIn("loss", out_desc)
        self.assertIn("ld", out_desc)
        out_desc["loss"].backward(retain_graph=True)
        self.assertIsNotNone(model.convDb.weight.grad)
        self.assertIsNone(model.convPb.weight.grad)

        # 2. 独立检测损失测试
        model.zero_grad()
        out_det = det_loss(
            pred0, pred1, homography,
            image0=image0, image1=image1,
            points0=torch.tensor([[10.0, 15.0]]),
            points1=torch.tensor([[10.0, 15.0]]),
        )
        for key in ("loss", "l_det", "l_orphan", "l_len", "l_prem"):
            self.assertIn(key, out_det)
        out_det["loss"].backward()
        self.assertIsNotNone(model.convPb.weight.grad)

    def test_decoupled_visualizations_detection_and_matching(self) -> None:
        """测试独立生成【检测: 原始掩膜+直线】与【描述: 配对情况】可视化图像。"""
        import tempfile
        from visualization.point_matches import (
            save_detection_mask_lines_figure,
            save_descriptor_matching_figure,
        )

        model = build_model("SuperPoint")
        loss_fn = SuperPointLoss()

        img0 = torch.rand(1, 5, 64, 64)
        img1 = torch.rand(1, 5, 64, 64)
        pred0 = model(img0)
        pred1 = model(img1)
        pts0 = torch.tensor([[12.0, 18.0], [30.0, 40.0]])
        pts1 = torch.tensor([[12.0, 18.0], [30.0, 40.0]])
        homo = torch.eye(3).unsqueeze(0)
        bgr0 = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr1 = np.zeros((64, 64, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            p_dir = Path(tmp_dir)
            det_path = p_dir / "test_detection.jpg"
            save_detection_mask_lines_figure(
                det_path,
                bgr0, bgr1,
                pred0, pred1,
                img0[:, 3:4], img1[:, 3:4],
                img0[:, 4:5], img1[:, 4:5],
                loss_fn,
                pair_id="sample_01",
            )
            self.assertTrue(det_path.is_file())
            self.assertGreater(det_path.stat().st_size, 500)

            match_path = p_dir / "test_matching.jpg"
            save_descriptor_matching_figure(
                match_path,
                bgr0, bgr1,
                pred0["descriptor"], pred1["descriptor"],
                pts0, pts1, homo,
                pair_id="sample_01",
            )
            self.assertTrue(match_path.is_file())
            self.assertGreater(match_path.stat().st_size, 500)

    def test_extract_endpoints_with_nms(self) -> None:
        """测试 NMS 抑制端点周围冗余聚簇，正确提取真实局部极大值。"""
        model = build_model("SuperPoint")
        # 构造一个 64x64 的模拟概率掩膜，在 (15, 15) 和 (45, 45) 放置两个强峰，并在周围放置次高峰
        mask = torch.zeros(1, 1, 64, 64)
        mask[0, 0, 15, 15] = 0.95
        mask[0, 0, 14, 15] = 0.85
        mask[0, 0, 15, 16] = 0.80

        mask[0, 0, 45, 45] = 0.90
        mask[0, 0, 46, 45] = 0.82

        points, scores = model.extract_endpoints(mask, threshold=0.1, nms_radius=3, top_k=5)
        # 验证经过 NMS 后，两个峰值附近的次高峰被成功抑制，仅保留独立的两个峰值
        p0 = points[0, 0].tolist()
        p1 = points[0, 1].tolist()
        s0 = float(scores[0, 0])
        s1 = float(scores[0, 1])

        self.assertAlmostEqual(s0, 0.95, places=3)
        self.assertAlmostEqual(s1, 0.90, places=3)
        self.assertEqual(p0, [15.0, 15.0])
        self.assertEqual(p1, [45.0, 45.0])
        # 第三名应该是 0.0 (因已被 NMS 抑制且其他区域为 0)
        self.assertAlmostEqual(float(scores[0, 2]), 0.0, places=4)

    def test_line_consistent_descriptor_loss(self) -> None:
        """测试描述子线一致性损失 (方案 A: 同一线相似，跨图同线相同，异线推离)。"""
        from models.superpoint.loss import SuperPointDescriptorLoss

        desc_loss = SuperPointDescriptorLoss()
        model = build_model("SuperPoint")
        image0 = torch.rand(1, 5, 64, 64)
        image1 = torch.rand(1, 5, 64, 64)
        pred0 = model(image0)
        pred1 = model(image1)
        homo = torch.eye(3).unsqueeze(0)

        # 构造两条线段
        lines0 = torch.tensor([[[10.0, 10.0, 30.0, 10.0], [10.0, 30.0, 30.0, 30.0]]])
        lines1 = torch.tensor([[[10.0, 10.0, 30.0, 10.0], [10.0, 30.0, 30.0, 30.0]]])

        out = desc_loss(pred0, pred1, homo, lines0=lines0, lines1=lines1, image0=image0, image1=image1)
        for key in ("loss", "ld", "l_intra", "l_cross", "l_inter", "l_bg"):
            self.assertIn(key, out)
            self.assertTrue(torch.isfinite(out[key]))

        # 反向传播验证梯度正常回传至描述子卷积头
        out["loss"].backward()
        self.assertIsNotNone(model.convDb.weight.grad)
        self.assertTrue(torch.isfinite(model.convDb.weight.grad).all())

    def test_line_descriptor_pca_visualization(self) -> None:
        """测试线一致性描述子 PCA 伪彩流形图生成。"""
        import tempfile
        from visualization.point_matches import save_line_descriptor_pca_figure

        model = build_model("SuperPoint")
        img0 = torch.rand(1, 5, 64, 64)
        img1 = torch.rand(1, 5, 64, 64)
        pred0 = model(img0)
        pred1 = model(img1)
        bgr0 = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr1 = np.zeros((64, 64, 3), dtype=np.uint8)
        lines0 = torch.tensor([[[10.0, 10.0, 30.0, 10.0]]])
        lines1 = torch.tensor([[[10.0, 10.0, 30.0, 10.0]]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pca_path = Path(tmp_dir) / "test_pca.jpg"
            save_line_descriptor_pca_figure(
                pca_path, bgr0, bgr1, pred0["descriptor"], pred1["descriptor"],
                lines0=lines0, lines1=lines1, pair_id="test_pair",
            )
            self.assertTrue(pca_path.is_file())
            self.assertGreater(pca_path.stat().st_size, 500)

    def test_progressive_stages_and_weights(self) -> None:
        """测试 5 阶段逐步损失训练架构与其动态权重切换。"""
        from models.superpoint.loss import STAGE_LOSS_WEIGHTS

        loss_fn = SuperPointLoss()
        expected_stages = [
            "stage1_det_warmup",
            "stage2_det_geom",
            "stage3_desc_intra",
            "stage4_desc_cross",
            "stage5_joint",
        ]
        for s in expected_stages:
            self.assertIn(s, STAGE_LOSS_WEIGHTS)

        # 1. 验证 stage1_det_warmup
        loss_fn.set_stage("stage1_det_warmup")
        self.assertEqual(loss_fn.weight_det, 1.0)
        self.assertEqual(loss_fn.weight_orphan, 0.0)
        self.assertEqual(loss_fn.weight_intra, 0.0)

        # 2. 验证 stage2_det_geom
        loss_fn.set_stage("stage2_det_geom")
        self.assertEqual(loss_fn.weight_det, 1.0)
        self.assertEqual(loss_fn.weight_orphan, 1.0)
        self.assertEqual(loss_fn.weight_len, 0.5)
        self.assertEqual(loss_fn.weight_premature, 0.5)
        self.assertEqual(loss_fn.weight_intra, 0.0)

        # 3. 验证 stage3_desc_intra
        loss_fn.set_stage("stage3_desc_intra")
        self.assertEqual(loss_fn.weight_det, 0.0)
        self.assertEqual(loss_fn.weight_intra, 1.0)
        self.assertEqual(loss_fn.weight_bg, 0.3)
        self.assertEqual(loss_fn.weight_cross, 0.0)

        # 4. 验证 stage5_joint
        loss_fn.set_stage("stage5_joint")
        self.assertEqual(loss_fn.weight_det, 1.0)
        self.assertEqual(loss_fn.weight_intra, 1.0)
        self.assertEqual(loss_fn.weight_cross, 1.5)

    def test_trainer_and_eval_helpers(self) -> None:
        """测试解耦后的 SuperPointTrainerHelper 与 SuperPointEvalHelper。"""
        import tempfile
        from models.superpoint.utils import SuperPointTrainerHelper, SuperPointEvalHelper

        with tempfile.TemporaryDirectory() as tmp_dir:
            p_tmp = Path(tmp_dir)
            trainer_helper = SuperPointTrainerHelper(
                data_dir="data/planar_letters",
                save_dir=str(p_tmp / "ckpt"),
                epochs=1,
                device="cpu",
                train_stage="stage2_det_geom",
                progressive=True,
                curriculum=[
                    {"epochs": [1, 1], "stage": "stage1_det_warmup"},
                ],
            )
            model, loss_fn, optimizer, train_loader, val_loader = trainer_helper.setup()
            stage = trainer_helper.update_stage(model, loss_fn, epoch=1)
            self.assertEqual(stage, "stage1_det_warmup")
            self.assertEqual(loss_fn.weight_det, 1.0)
            self.assertEqual(loss_fn.weight_orphan, 0.0)

            eval_helper = SuperPointEvalHelper(
                data_dir="data/planar_letters",
                checkpoint=str(p_tmp / "non_existent.pth"),
                out_dir=str(p_tmp / "eval_out"),
                num_vis=0,
                device="cpu",
                eval_stage="all",
            )
            e_model, criterion, loader = eval_helper.setup()
            self.assertIsNotNone(e_model)
            self.assertIsNotNone(criterion)

    def test_pure_line_detector_loss(self) -> None:
        """测试纯直线场找线检测层: 仅保留距离场像素比较与角度场极差比较，其余全部锁定。"""
        from models.superpoint.loss import SuperPointLoss, loss_line_distance, loss_line_angle

        # 1. 验证 detector 阶段权重
        loss_fn = SuperPointLoss(stage="detector")
        self.assertEqual(loss_fn.weight_dist, 1.0)
        self.assertEqual(loss_fn.weight_ang, 1.0)
        self.assertEqual(loss_fn.weight_det, 0.0)
        self.assertEqual(loss_fn.weight_orphan, 0.0)
        self.assertEqual(loss_fn.weight_len, 0.0)
        self.assertEqual(loss_fn.weight_premature, 0.0)
        self.assertEqual(loss_fn.weight_intra, 0.0)
        self.assertEqual(loss_fn.weight_cross, 0.0)
        self.assertEqual(loss_fn.weight_desc, 0.0)

        # 2. 验证模型分支参数锁定
        model = build_model("SuperPoint")
        model.set_train_stage("detector")
        self.assertTrue(model.convPa.weight.requires_grad)
        self.assertTrue(model.convPb.weight.requires_grad)
        self.assertFalse(model.convDa.weight.requires_grad)
        self.assertFalse(model.convDb.weight.requires_grad)

        # 3. 前向计算与反向梯度验证
        img0 = torch.rand(1, 5, 64, 64)
        img1 = torch.rand(1, 5, 64, 64)
        pred0 = model(img0)
        pred1 = model(img1)
        homo = torch.eye(3).unsqueeze(0)

        out = loss_fn(pred0, pred1, homo, image0=img0, image1=img1)
        self.assertIn("l_dist", out)
        self.assertIn("l_ang", out)
        self.assertTrue(torch.isfinite(out["l_dist"]))
        self.assertTrue(torch.isfinite(out["l_ang"]))
        self.assertTrue(torch.isfinite(out["loss"]))

        out["loss"].backward()
        # 检测头具有梯度，描述子头无梯度
        self.assertIsNotNone(model.convPb.weight.grad)
        self.assertTrue(torch.isfinite(model.convPb.weight.grad).all())
        self.assertIsNone(model.convDb.weight.grad)

        # 4. 独立测试数学算子
        sampled_df = torch.ones(1, 4, 4, 10)  # 完全在距离脊线上 (1.0)
        sampled_af = torch.full((1, 4, 4, 10), 0.5)  # 角度恒定
        pair_scores = torch.full((1, 4, 4), 0.8)

        l_dist = loss_line_distance(sampled_df, pair_scores, tau_df=0.55)
        l_ang = loss_line_angle(sampled_af, pair_scores, tau_ang=0.10)
        self.assertTrue(torch.isfinite(l_dist))
        self.assertTrue(torch.isfinite(l_ang))




if __name__ == "__main__":
    unittest.main()
