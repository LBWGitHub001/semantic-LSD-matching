"""
RESTful API Handlers and WebSocket Handlers for the Web Management Interface.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import tornado.web
import tornado.websocket

from review_runs import find_all_runs
from web.process_manager import TASK_MANAGER, ROOT_DIR


class BaseApiHandler(tornado.web.RequestHandler):
    """Base API handler with JSON response helpers and CORS support."""

    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")

    def options(self, *args: Any, **kwargs: Any) -> None:
        self.set_status(204)
        self.finish()

    def write_json(self, data: Any, status: int = 200) -> None:
        self.set_status(status)
        self.write(json.dumps(data, ensure_ascii=False))

    def write_error_json(self, message: str, status: int = 400) -> None:
        self.write_json({"error": True, "message": message}, status=status)

    def get_json_body(self) -> dict[str, Any]:
        try:
            return json.loads(self.request.body.decode("utf-8"))
        except Exception:
            return {}


class StatusHandler(BaseApiHandler):
    """GET /api/status - Returns system and active process status."""

    def get(self) -> None:
        status = TASK_MANAGER.get_status()
        self.write_json(status)


class RunsHandler(BaseApiHandler):
    """GET /api/runs - Lists all historical training experiments."""

    def get(self) -> None:
        ckpt_dir = ROOT_DIR / "checkpoints"
        raw_runs = find_all_runs(ckpt_dir)

        runs_summary = []
        for r in raw_runs:
            cfg = r.get("config", {})
            exp_info = cfg.get("experiment_info", {})
            hparams = cfg.get("hyperparameters", {})
            best_val = cfg.get("best_validation") or {}
            test_bench = cfg.get("test_benchmark") or {}
            r_dir = r.get("run_dir")

            has_report = (r_dir / "report.md").is_file() if r_dir else False
            has_curves = (r_dir / "curves.png").is_file() if r_dir else False
            has_best = (r_dir / "best_model.pth").is_file() if r_dir else False

            runs_summary.append({
                "run_name": r.get("run_name"),
                "run_tag": r.get("run_tag", r.get("run_name")),
                "model_type": r.get("model_type"),
                "run_dir": str(r_dir),
                "rel_dir": str(r_dir.relative_to(ROOT_DIR)) if r_dir else "",
                "start_time": exp_info.get("start_time", "N/A"),
                "end_time": exp_info.get("end_time"),
                "epochs": hparams.get("epochs", 0),
                "lr": hparams.get("learning_rate"),
                "dustbin_weight": hparams.get("dustbin_weight"),
                "motion_weight": hparams.get("motion_loss_weight"),
                "best_epoch": best_val.get("epoch"),
                "val_f1": best_val.get("f1"),
                "val_precision": best_val.get("precision"),
                "val_recall": best_val.get("recall"),
                "test_raw_f1": test_bench.get("raw_visual", {}).get("f1") if isinstance(test_bench, dict) else None,
                "test_sem_f1": test_bench.get("semantic_constrained", {}).get("f1") if isinstance(test_bench, dict) else None,
                "has_report": has_report,
                "has_curves": has_curves,
                "has_best_model": has_best,
            })

        self.write_json({"runs": runs_summary, "total": len(runs_summary)})


class RunDetailHandler(BaseApiHandler):
    """GET /api/runs/<run_name> - Fetches full metadata, report markdown, and metrics trajectory."""

    def get(self, run_name: str) -> None:
        ckpt_dir = ROOT_DIR / "checkpoints"
        raw_runs = find_all_runs(ckpt_dir)

        matched = None
        for r in raw_runs:
            if (
                r["run_name"] == run_name
                or r.get("run_tag") == run_name
                or run_name in str(r["run_dir"])
                or r["run_dir"].name == run_name
                or (r.get("run_tag") and run_name in r["run_tag"])
            ):
                matched = r
                break

        if not matched:
            self.write_error_json(f"Run '{run_name}' not found", status=404)
            return

        r_dir = matched["run_dir"]
        cfg = matched["config"]
        history = matched.get("history", [])

        # Read report.md if available
        report_md_path = r_dir / "report.md"
        report_content = ""
        if report_md_path.is_file():
            try:
                report_content = report_md_path.read_text(encoding="utf-8")
            except Exception:
                pass

        # Curves relative path
        curves_rel = None
        if (r_dir / "curves.png").is_file():
            curves_rel = f"/files/{r_dir.relative_to(ROOT_DIR)}/curves.png"

        # Checkpoint files in this run
        run_ckpts = []
        for pth in sorted(r_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
            run_ckpts.append({
                "name": pth.name,
                "rel_path": str(pth.relative_to(ROOT_DIR)),
                "size_mb": round(pth.stat().st_size / (1024 * 1024), 2),
                "is_best": "best_model" in pth.name,
                "modified": pth.stat().st_mtime,
            })

        # Visualization images in this run
        vis_dir = r_dir / "visualizations"
        vis_images = []
        if vis_dir.is_dir():
            for img_p in sorted(vis_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True):
                vis_images.append({
                    "name": img_p.name,
                    "url": f"/files/{img_p.relative_to(ROOT_DIR)}",
                    "size_kb": round(img_p.stat().st_size / 1024, 1),
                    "modified": img_p.stat().st_mtime,
                })

        self.write_json({
            "run_name": matched["run_name"],
            "run_tag": matched.get("run_tag", matched["run_name"]),
            "model_type": matched["model_type"],
            "run_dir": str(r_dir),
            "config": cfg,
            "history": history,
            "report_markdown": report_content,
            "curves_url": curves_rel,
            "checkpoints": run_ckpts,
            "visualizations": vis_images,
        })


class CompareRunsHandler(BaseApiHandler):
    """GET /api/runs/compare?run1=...&run2=... - Compares two runs side by side."""

    def get(self) -> None:
        r1 = self.get_query_argument("run1", "")
        r2 = self.get_query_argument("run2", "")
        if not r1 or not r2:
            self.write_error_json("Please provide both run1 and run2 query parameters")
            return

        all_runs = find_all_runs(ROOT_DIR / "checkpoints")
        m1 = next((r for r in all_runs if r["run_name"] == r1 or r.get("run_tag") == r1 or r1 in str(r["run_dir"]) or (r.get("run_tag") and r1 in r["run_tag"])), None)
        m2 = next((r for r in all_runs if r["run_name"] == r2 or r.get("run_tag") == r2 or r2 in str(r["run_dir"]) or (r.get("run_tag") and r2 in r["run_tag"])), None)

        if not m1 or not m2:
            self.write_error_json("One or both runs could not be found", status=404)
            return

        cfg1 = m1["config"]
        cfg2 = m2["config"]
        hp1 = cfg1.get("hyperparameters", {})
        hp2 = cfg2.get("hyperparameters", {})
        bv1 = cfg1.get("best_validation") or {}
        bv2 = cfg2.get("best_validation") or {}
        tb1 = cfg1.get("test_benchmark") or {}
        tb2 = cfg2.get("test_benchmark") or {}

        comparison_table = [
            {"feature": "Experiment Tag", "run1": m1.get("run_tag", m1["run_name"]), "run2": m2.get("run_tag", m2["run_name"])},
            {"feature": "Model Architecture", "run1": m1["model_type"], "run2": m2["model_type"]},
            {"feature": "Total Epochs", "run1": hp1.get("epochs"), "run2": hp2.get("epochs")},
            {"feature": "Learning Rate", "run1": hp1.get("learning_rate"), "run2": hp2.get("learning_rate")},
            {"feature": "Dustbin Weight", "run1": hp1.get("dustbin_weight"), "run2": hp2.get("dustbin_weight")},
            {"feature": "Motion Loss Weight", "run1": hp1.get("motion_loss_weight"), "run2": hp2.get("motion_loss_weight")},
            {"feature": "Min Area Ratio", "run1": hp1.get("min_area_ratio"), "run2": hp2.get("min_area_ratio")},
            {"feature": "Pyramid Features", "run1": hp1.get("all_cli_args", {}).get("use_pyramid", True), "run2": hp2.get("all_cli_args", {}).get("use_pyramid", True)},
            {"feature": "Best Val Epoch", "run1": bv1.get("epoch"), "run2": bv2.get("epoch")},
            {"feature": "Best Val F1", "run1": f"{bv1.get('f1', 0):.2f}%" if bv1.get('f1') else "-", "run2": f"{bv2.get('f1', 0):.2f}%" if bv2.get('f1') else "-"},
            {"feature": "Best Val Precision", "run1": f"{bv1.get('precision', 0):.2f}%" if bv1.get('precision') else "-", "run2": f"{bv2.get('precision', 0):.2f}%" if bv2.get('precision') else "-"},
            {"feature": "Best Val Recall", "run1": f"{bv1.get('recall', 0):.2f}%" if bv1.get('recall') else "-", "run2": f"{bv2.get('recall', 0):.2f}%" if bv2.get('recall') else "-"},
            {"feature": "Test Raw F1", "run1": f"{tb1.get('raw_visual', {}).get('f1', 0):.2f}%" if isinstance(tb1, dict) and tb1.get('raw_visual') else "-", "run2": f"{tb2.get('raw_visual', {}).get('f1', 0):.2f}%" if isinstance(tb2, dict) and tb2.get('raw_visual') else "-"},
            {"feature": "Test Semantic F1", "run1": f"{tb1.get('semantic_constrained', {}).get('f1', 0):.2f}%" if isinstance(tb1, dict) and tb1.get('semantic_constrained') else "-", "run2": f"{tb2.get('semantic_constrained', {}).get('f1', 0):.2f}%" if isinstance(tb2, dict) and tb2.get('semantic_constrained') else "-"},
        ]

        # Check if pre-rendered compare plot exists
        cmp_plot_name = f"compare_{m1['run_name']}_vs_{m2['run_name']}.png"
        cmp_plot_path = ROOT_DIR / "outputs" / cmp_plot_name
        cmp_url = f"/files/outputs/{cmp_plot_name}" if cmp_plot_path.is_file() else None

        self.write_json({
            "run1_name": m1["run_name"],
            "run2_name": m2["run_name"],
            "table": comparison_table,
            "curve_url": cmp_url,
            "run1_history": m1.get("history", []),
            "run2_history": m2.get("history", []),
        })


class CheckpointsHandler(BaseApiHandler):
    """GET /api/checkpoints - Discovers all checkpoint files across checkpoints directory."""

    def get(self) -> None:
        ckpt_root = ROOT_DIR / "checkpoints"
        models: list[dict[str, Any]] = []

        if ckpt_root.is_dir():
            for pth in sorted(ckpt_root.glob("**/*.pth"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel = pth.relative_to(ROOT_DIR)
                is_best = "best_model" in pth.name
                size_mb = round(pth.stat().st_size / (1024 * 1024), 2)
                mtime = pth.stat().st_mtime
                models.append({
                    "path": str(pth),
                    "rel_path": str(rel),
                    "name": pth.name,
                    "folder": pth.parent.name,
                    "is_best": is_best,
                    "size_mb": size_mb,
                    "modified": mtime,
                })

        self.write_json({"checkpoints": models, "total": len(models)})


class DatasetsHandler(BaseApiHandler):
    """GET /api/datasets - Discovers datasets in data/ and provides pair counts."""

    def get(self) -> None:
        data_root = ROOT_DIR / "data"
        datasets = []

        if data_root.is_dir():
            for d in data_root.iterdir():
                if d.is_dir():
                    train_split = d / "train_split.json"
                    val_split = d / "val_split.json"
                    test_split = d / "test_split.json"

                    train_count = 0
                    val_count = 0
                    test_count = 0

                    if train_split.is_file():
                        try:
                            with open(train_split, "r") as f:
                                train_count = len(json.load(f))
                        except Exception:
                            pass
                    if val_split.is_file():
                        try:
                            with open(val_split, "r") as f:
                                val_count = len(json.load(f))
                        except Exception:
                            pass
                    if test_split.is_file():
                        try:
                            with open(test_split, "r") as f:
                                test_count = len(json.load(f))
                        except Exception:
                            pass

                    datasets.append({
                        "name": d.name,
                        "path": str(d),
                        "rel_path": f"data/{d.name}",
                        "train_pairs": train_count,
                        "val_pairs": val_count,
                        "test_pairs": test_count,
                        "total_pairs": train_count + val_count + test_count,
                    })

        self.write_json({"datasets": datasets})


class GalleryHandler(BaseApiHandler):
    """GET /api/gallery - Directly lists all visualization images across the system."""

    def get(self) -> None:
        images = []
        seen = set()

        # 1. Output directory visualizations (eval samples)
        outputs_dir = ROOT_DIR / "outputs" / "semantic_matching_vis"
        if outputs_dir.is_dir():
            for img_p in outputs_dir.glob("*.jpg"):
                name = img_p.name
                rel_url = f"/files/outputs/semantic_matching_vis/{name}"
                if rel_url not in seen:
                    seen.add(rel_url)
                    images.append({
                        "name": name,
                        "url": rel_url,
                        "size_kb": round(img_p.stat().st_size / 1024, 1),
                        "modified": img_p.stat().st_mtime,
                        "source": "Benchmark Output",
                    })

        # 2. Checkpoint run visualizations (test samples during training)
        ckpt_dir = ROOT_DIR / "checkpoints"
        if ckpt_dir.is_dir():
            for img_p in ckpt_dir.glob("**/visualizations/*.jpg"):
                rel_p = img_p.relative_to(ROOT_DIR)
                rel_url = f"/files/{rel_p}"
                if rel_url not in seen:
                    seen.add(rel_url)
                    run_tag = img_p.parent.parent.name
                    images.append({
                        "name": f"{run_tag}_{img_p.name}",
                        "url": rel_url,
                        "size_kb": round(img_p.stat().st_size / 1024, 1),
                        "modified": img_p.stat().st_mtime,
                        "source": f"Run: {run_tag}",
                    })

        # Sort newest first
        images.sort(key=lambda x: x["modified"], reverse=True)

        # Return up to 160 images for fast frontend rendering
        self.write_json({"images": images[:160], "total": len(images)})


def get_model_preview_data(model_type: str, use_pyramid: bool = True, d_model: int = 128) -> dict[str, Any]:
    """Generates detailed architectural breakdown and PyTorch module structure for a given model."""
    if model_type in ["xfeat_line_matcher", "xfeat_line", "line_matcher"]:
        pyr_str = "启用多尺度 FPN 特征金字塔 (P1/P2/P3 ➔ 192D)" if use_pyramid else "单尺度特征输出 (64D)"
        vis_dim = 192 if use_pyramid else 64
        total_fusion = vis_dim + 64 + 64

        pytorch_repr = f"""XFeatLineMatcher(
  (visual_backbone): {'XFeatPyramidBackbone' if use_pyramid else 'XFeatBackbone'}(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(32)
    (relu): ReLU(inplace=True)
    (layer1): Sequential(BasicBlock(32, 64, stride=2))   # Stride 4 (P1 fine geometric details)
    (layer2): Sequential(BasicBlock(64, 64, stride=2))   # Stride 8 (P2 part-level shapes)
    (layer3): Sequential(BasicBlock(64, 64, stride=2))   # Stride 16 (P3 high-level semantics)
    {'# Lateral 1x1 convolutions & Top-Down Feature Fusion Pyramid' if use_pyramid else ''}
    (lateral1): Conv2d(64, 64, kernel_size=(1, 1))
    (lateral2): Conv2d(64, 64, kernel_size=(1, 1))
    (smooth1): Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
    (smooth2): Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
  )
  (visual_pool): AdaptiveAvgPool2d(output_size=(1, 1))  # Pool to ({vis_dim}D)
  (line_encoder): PatchLineGeometryEncoder(
    # 输入: LSD亚像素直线 5D [xc/W, yc/H, L/max(W,H), sin(2θ), cos(2θ)]
    (mlp): Sequential(
      (0): Linear(in_features=5, out_features=64, bias=True)
      (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=64, out_features=64, bias=True)
      (4): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (5): ReLU(inplace=True)
    )
    (pool): MaxPool1d(over K lines ➔ 64D patch line descriptor)
  )
  (geo_encoder): ObjectGeometryEncoder(
    # 输入: 目标区域边界框 4D [cx/W, cy/H, w/W, h/H]
    (mlp): Sequential(
      (0): Linear(in_features=4, out_features=64, bias=True)
      (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (2): ReLU(inplace=True)
      (3): Linear(in_features=64, out_features=64, bias=True)
      (4): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (5): ReLU(inplace=True)
    )
  )
  (fusion_proj): Sequential(
    # 模态拼接: [Visual({vis_dim}D) || Line(64D) || BBox(64D)] ➔ {total_fusion}D
    (0): Linear(in_features={total_fusion}, out_features={d_model}, bias=True)
    (1): LayerNorm(({d_model},), eps=1e-05, elementwise_affine=True)
    (2): ReLU(inplace=True)
  )
  (transformer): ObjectGraphTransformer(
    (layers): ModuleList(
      (0-1): 2 x TransformerCrossLayer(
        (self_attn): MultiheadAttention(embed_dim={d_model}, num_heads=4, batch_first=True)
        (cross_attn): MultiheadAttention(embed_dim={d_model}, num_heads=4, batch_first=True)
        (norm1): LayerNorm(({d_model},), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm(({d_model},), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm(({d_model},), eps=1e-05, elementwise_affine=True)
        (ffn): Sequential(
          (0): Linear(in_features={d_model}, out_features={d_model*2}, bias=True)
          (1): ReLU(inplace=True)
          (2): Linear(in_features={d_model*2}, out_features={d_model}, bias=True)
        )
      )
    )
  )
  (matcher): ObjectSinkhornMatcher(
    (proj): Linear(in_features={d_model}, out_features={d_model}, bias=True)
    (dustbin_score): Parameter(shape=torch.Size([1]))  # 扩展垃圾桶维度以吸收无对应物
    (sinkhorn): LogSinkhornOptimalTransport(iters=20)
  )
)"""
        return {
            "model_type": "xfeat_line_matcher",
            "name": "XFeatLineMatcher",
            "tagline": "多模态多尺度直线与目标区域联合对齐网络 (SOTA 79.2%)",
            "params_total": "1,722,081 (约 1.72 M 参数)",
            "trainable_params": "1,722,081",
            "diagram_url": "/static/img/xfeat_line_matcher_architecture.png",
            "input_shapes": [
                {"name": "候选目标切片 (Region Patches)", "shape": "(B, M, 3, 112, 112)", "desc": "RGB三通道切片，经过自适应扩边裁剪插值"},
                {"name": "关联LSD直线 (Associated Lines)", "shape": "(B, M, K, 5)", "desc": "每切片至多K=8根，5D亚像素归一化几何向量"},
                {"name": "切片边界框 (Bounding Boxes)", "shape": "(B, M, 4)", "desc": "切片在原图中的[cx/W, cy/H, w/W, h/H]全局几何布局"}
            ],
            "pipeline_stages": [
                {"stage": "1. 视觉特征抽取", "dim": f"{vis_dim}D", "desc": pyr_str},
                {"stage": "2. 直线几何编码", "dim": "64D", "desc": "5D端点参数 ➔ 两层MLP ➔ K根线最大池化"},
                {"stage": "3. 空间边界框编码", "dim": "64D", "desc": "4D归一化坐标 ➔ 两层MLP映射"},
                {"stage": "4. 多模态融合投影", "dim": f"{d_model}D", "desc": f"将拼接后的{total_fusion}D向量投影归一化为{d_model}D Token"},
                {"stage": "5. 图注意力推理", "dim": f"{d_model}D", "desc": "2层图交叉注意力机制 (Self-Attn + Cross-Attn)"},
                {"stage": "6. 最优传输匹配", "dim": "(M0+1, M1+1)", "desc": "带扩展垃圾桶维度的Log-Sinkhorn双随机最优传输求解"}
            ],
            "pytorch_repr": pytorch_repr,
        }
    elif model_type in ["xfeat_object", "xfeat", "object"]:
        return {
            "model_type": "xfeat_object",
            "name": "XFeatObjectMatcher",
            "tagline": "纯物体区域多尺度视觉外观与空间几何基线网络",
            "params_total": "1,680,449 (约 1.68 M 参数)",
            "trainable_params": "1,680,449",
            "input_shapes": [
                {"name": "候选目标切片 (Region Patches)", "shape": "(B, M, 3, 112, 112)", "desc": "RGB三通道切片"},
                {"name": "切片边界框 (Bounding Boxes)", "shape": "(B, M, 4)", "desc": "切片在原图中的空间几何坐标"}
            ],
            "pipeline_stages": [
                {"stage": "1. 视觉特征抽取", "dim": "192D", "desc": "XFeat FPN 特征金字塔三层池化"},
                {"stage": "2. 空间布局编码", "dim": "64D", "desc": "边界框 4D ➔ MLP(64D)"},
                {"stage": "3. 融合投影", "dim": f"{d_model}D", "desc": "256D ➔ 128D LayerNorm + ReLU"},
                {"stage": "4. 图Transformer", "dim": f"{d_model}D", "desc": "2层跨视角交叉注意力"},
                {"stage": "5. 最优传输", "dim": "(M0+1, M1+1)", "desc": "Log-Sinkhorn 20次迭代双随机对齐"}
            ],
            "pytorch_repr": f"""XFeatObjectMatcher(
  (visual_backbone): XFeatPyramidBackbone(P1, P2, P3 ➔ 192D)
  (visual_pool): AdaptiveAvgPool2d(output_size=(1, 1))
  (geo_encoder): ObjectGeometryEncoder(MLP: 4D ➔ 64D)
  (fusion_proj): Sequential(Linear(256, {d_model}), LayerNorm({d_model}), ReLU())
  (transformer): ObjectGraphTransformer(2 Layers, 4 Heads, embed_dim={d_model})
  (matcher): ObjectSinkhornMatcher(LogSinkhorn, iters=20, dustbin=True)
)""",
        }
    elif model_type in ["mask_line"]:
        return {
            "model_type": "mask_line",
            "name": "MaskLineFeatMatchNet",
            "tagline": "6通道几何场（RGB + EDT距离场 + AFM朝向场）线段网络",
            "params_total": "3,245,184 (约 3.25 M 参数)",
            "trainable_params": "3,245,184",
            "input_shapes": [
                {"name": "多通道融合图像", "shape": "(B, 6, H, W)", "desc": "3通道RGB + 1通道欧氏距离场(EDT) + 2通道角度朝向场(AFM)"},
                {"name": "线段几何端点", "shape": "(B, N, 4)", "desc": "直线两端点坐标 [x1, y1, x2, y2]"}
            ],
            "pipeline_stages": [
                {"stage": "1. 几何场输入卷积", "dim": "64D", "desc": "Conv2d 6通道输入 ResNet 浅层骨干"},
                {"stage": "2. 线段端点编码", "dim": "64D", "desc": "LineEndpointMLP 对两端点提取位置编码"},
                {"stage": "3. 特征融合与分类", "dim": "128D", "desc": "特征图局部采样融合与相似度判定"},
                {"stage": "4. 匹配矩阵", "dim": "(N0, N1)", "desc": "DualSoftmax 双向互最近邻门限过滤"}
            ],
            "pytorch_repr": """MaskLineFeatMatchNet(
  (backbone): Conv6chResNet(in_channels=6, base_channels=64)
  (line_encoder): LineEndpointMLP(in_features=4, out_features=64)
  (fusion): FeatureMapSampler(channels=64)
  (matcher): DualSoftmaxLineMatcher(temperature=0.1)
)""",
        }
    else:  # line_transformer
        return {
            "model_type": "line_transformer",
            "name": "LineMatcherNet",
            "tagline": "端点几何向量与局部特征插值采样 Transformer 基线",
            "params_total": "1,148,928 (约 1.15 M 参数)",
            "trainable_params": "1,148,928",
            "input_shapes": [
                {"name": "直线端点坐标", "shape": "(B, N, 4)", "desc": "直线两端点坐标向量"},
                {"name": "局部采样点", "shape": "(B, N, 7, 2)", "desc": "沿直线均匀采样的7个局部坐标"}
            ],
            "pipeline_stages": [
                {"stage": "1. 端点几何编码", "dim": "128D", "desc": "LineGeometryEncoder MLP"},
                {"stage": "2. 局部视觉采样", "dim": "128D", "desc": "在骨干特征图上沿线段双线性插值采样"},
                {"stage": "3. Transformer", "dim": "128D", "desc": "TransformerEncoder 2层"},
                {"stage": "4. 最近邻匹配", "dim": "(N0, N1)", "desc": "距离阈值双向几何过滤"}
            ],
            "pytorch_repr": """LineMatcherNet(
  (line_encoder): LineGeometryEncoder(4 ➔ 128)
  (sampler): LineFeatureSampler(samples=7)
  (transformer): TransformerEncoder(d_model=128, nhead=4, num_layers=2)
  (matcher): MutualNearestNeighborMatcher(dist_thresh=0.2)
)""",
        }


class ModelPreviewHandler(BaseApiHandler):
    """GET /api/model/preview - Returns detailed model architecture specification."""

    def get(self) -> None:
        model_type = self.get_query_argument("model", "xfeat_line_matcher")
        use_pyramid = self.get_query_argument("pyramid", "true").lower() in ["true", "1", "yes"]
        d_model = int(self.get_query_argument("d_model", "128"))

        data = get_model_preview_data(model_type, use_pyramid, d_model)
        self.write_json(data)


class TrainHandler(BaseApiHandler):
    """POST /api/train/start and POST /api/train/stop - Controls training execution."""

    def post(self) -> None:
        action = self.get_argument("action", "start")
        if action == "stop":
            stopped, msg = TASK_MANAGER.stop_current_task()
            self.write_json({"success": stopped, "message": msg})
            return

        # Start Training
        body = self.get_json_body()
        model_type = body.get("model", "xfeat_line_matcher")
        data_dir = body.get("data_dir", "data/xfeat_line_matcher")
        epochs = body.get("epochs", 15)
        lr = body.get("lr", 5e-4)
        dustbin_weight = body.get("dustbin_weight", 0.1)
        motion_weight = body.get("motion_weight", 0.4)
        min_area_ratio = body.get("min_area_ratio", 0.40)
        data_ratio = float(body.get("data_ratio", 1.0))
        use_pyramid = body.get("pyramid", True)
        use_amp = body.get("amp", True)
        save_interval = body.get("save_interval", 5)
        num_workers = body.get("num_workers", 4)
        resume = body.get("resume")

        cmd = [
            "train.py",
            "--model", str(model_type),
            "--data-dir", str(data_dir),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--dustbin-weight", str(dustbin_weight),
            "--motion-weight", str(motion_weight),
            "--min-area-ratio", str(min_area_ratio),
            "--num-workers", str(num_workers),
            "--save-interval", str(save_interval),
        ]
        if data_ratio < 1.0:
            cmd.extend(["--data-ratio", str(data_ratio)])
        if use_pyramid:
            cmd.append("--pyramid")
        else:
            cmd.append("--no-pyramid")
        if use_amp:
            cmd.append("--amp")
        if resume:
            cmd.extend(["--resume", str(resume)])

        success, msg, task_info = TASK_MANAGER.start_task("train", cmd)
        if success:
            self.write_json({"success": True, "message": msg, "task": task_info})
        else:
            self.write_error_json(msg, status=409)


class EvalHandler(BaseApiHandler):
    """POST /api/eval/start - Controls evaluation benchmark execution."""

    def post(self) -> None:
        action = self.get_argument("action", "start")
        if action == "stop":
            stopped, msg = TASK_MANAGER.stop_current_task()
            self.write_json({"success": stopped, "message": msg})
            return

        body = self.get_json_body()
        checkpoint = body.get("checkpoint", "checkpoints/xfeat_line_matcher/20261024_140825/best_model.pth")
        data_dir = body.get("data_dir", "data/xfeat_line_matcher")
        split = body.get("split", "test")
        min_area_ratio = body.get("min_area_ratio", 0.40)
        num_vis = body.get("num_vis", 6)

        cmd = [
            "eval.py",
            "--checkpoint", str(checkpoint),
            "--data-dir", str(data_dir),
            "--split", str(split),
            "--min-area-ratio", str(min_area_ratio),
            "--num-vis", str(num_vis),
        ]

        success, msg, task_info = TASK_MANAGER.start_task("eval", cmd)
        if success:
            self.write_json({"success": True, "message": msg, "task": task_info})
        else:
            self.write_error_json(msg, status=409)


class DatasetGenHandler(BaseApiHandler):
    """POST /api/dataset/generate - Starts dataset generation process."""

    def post(self) -> None:
        action = self.get_argument("action", "start")
        if action == "stop":
            stopped, msg = TASK_MANAGER.stop_current_task()
            self.write_json({"success": stopped, "message": msg})
            return

        body = self.get_json_body()
        task = body.get("task", "structured3d_lines")
        out_dir = body.get("out_dir", "data/xfeat_line_matcher")
        s3d_dir = body.get("s3d_dir", "data/Structured3D")
        max_scenes = body.get("max_scenes")
        num_workers = body.get("num_workers", 4)
        min_area_ratio = body.get("min_area_ratio", 0.40)

        cmd = [
            "generate_dataset.py",
            "--task", str(task),
            "--out-dir", str(out_dir),
            "--s3d-dir", str(s3d_dir),
            "--num-workers", str(num_workers),
            "--min-area-ratio", str(min_area_ratio),
        ]
        if max_scenes:
            cmd.extend(["--max-scenes", str(max_scenes)])

        success, msg, task_info = TASK_MANAGER.start_task("dataset", cmd)
        if success:
            self.write_json({"success": True, "message": msg, "task": task_info})
        else:
            self.write_error_json(msg, status=409)


class ActiveLogsHandler(BaseApiHandler):
    """GET /api/logs/active - Returns recent logs of the active task."""

    def get(self) -> None:
        self.write_json({
            "logs": TASK_MANAGER.get_active_logs(),
            "status": TASK_MANAGER.get_status(),
        })


class HistoricalLogsHandler(BaseApiHandler):
    """GET /api/logs/history?run=<run_name> - Reads train.log of a past run."""

    def get(self) -> None:
        run_name = self.get_query_argument("run", "")
        if not run_name:
            self.write_error_json("Missing run parameter")
            return
        all_runs = find_all_runs(ROOT_DIR / "checkpoints")
        matched = next((r for r in all_runs if r["run_name"] == run_name or run_name in str(r["run_dir"])), None)
        if not matched:
            self.write_error_json("Run not found", status=404)
            return
        log_path = matched["run_dir"] / "train.log"
        lines = []
        if log_path.is_file():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                pass
        self.write_json({"run_name": run_name, "lines": lines, "total": len(lines)})


class LogsWebSocketHandler(tornado.websocket.WebSocketHandler):
    """WebSocket /ws/logs - Real-time terminal log stream."""

    def check_origin(self, origin: str) -> bool:
        return True

    def open(self) -> None:
        # Crucial: capture the actual running server event loop in the main thread!
        self.loop = tornado.ioloop.IOLoop.current()
        TASK_MANAGER.subscribe(self)
        # Send current task status & recent backlog
        status = TASK_MANAGER.get_status()
        self.write_message_safe({
            "type": "init",
            "status": status,
            "backlog": TASK_MANAGER.get_active_logs()[-300:],
        })

    def on_close(self) -> None:
        TASK_MANAGER.unsubscribe(self)

    def on_message(self, message: str) -> None:
        try:
            msg = json.loads(message)
            if msg.get("type") == "ping":
                self.write_message_safe({"type": "pong"})
        except Exception:
            pass

    def send_message_sync(self, message: dict[str, Any]) -> None:
        """Called by ProcessManager background thread."""
        try:
            if hasattr(self, "loop") and self.loop:
                self.loop.add_callback(self.write_message_safe, message)
            else:
                tornado.ioloop.IOLoop.instance().add_callback(self.write_message_safe, message)
        except Exception:
            pass

    def write_message_safe(self, message: dict[str, Any]) -> None:
        try:
            self.write_message(json.dumps(message, ensure_ascii=False))
        except Exception:
            pass
