"""
Unit and Integration Tests for Web API and Process Manager.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import tornado.testing
from web.server import make_app
from web.process_manager import TASK_MANAGER


class TestWebApi(tornado.testing.AsyncHTTPTestCase):
    def get_app(self) -> tornado.web.Application:
        return make_app()

    def test_status_endpoint(self) -> None:
        response = self.fetch("/api/status")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("is_busy", data)
        self.assertIn("gpu", data)
        self.assertIn("python", data)
        self.assertIn("resources", data)
        self.assertIn("cpu", data["resources"])
        self.assertIn("ram", data["resources"])
        self.assertIn("pct", data["resources"]["cpu"])
        self.assertIn("pct", data["resources"]["ram"])

    def test_runs_endpoint(self) -> None:
        response = self.fetch("/api/runs")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("runs", data)
        self.assertGreaterEqual(len(data["runs"]), 5)
        first = data["runs"][0]
        self.assertIn("run_name", first)
        self.assertIn("run_tag", first)
        self.assertIn("model_type", first)
        self.assertIn("+", first["run_tag"])
        self.assertIn("【", first["run_tag"])
        self.assertIn("】", first["run_tag"])

    def test_run_detail_endpoint(self) -> None:
        response = self.fetch("/api/runs/20261023_233349")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertEqual(data["run_name"], "20261023_233349")
        self.assertIn("config", data)
        self.assertIn("report_markdown", data)
        self.assertGreaterEqual(len(data["history"]), 12)

    def test_checkpoints_endpoint(self) -> None:
        response = self.fetch("/api/checkpoints")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("checkpoints", data)
        self.assertGreater(len(data["checkpoints"]), 0)

    def test_datasets_endpoint(self) -> None:
        response = self.fetch("/api/datasets")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("datasets", data)
        names = [d["name"] for d in data["datasets"]]
        self.assertIn("xfeat_line_matcher", names)

    def test_gallery_endpoint(self) -> None:
        response = self.fetch("/api/gallery")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("images", data)
        self.assertGreater(len(data["images"]), 0)

    def test_compare_endpoint(self) -> None:
        response = self.fetch("/api/runs/compare?run1=20261023_215042&run2=20261023_233349")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("table", data)
        self.assertGreater(len(data["table"]), 5)

    def test_logs_endpoints(self) -> None:
        # Active logs
        res_act = self.fetch("/api/logs/active")
        self.assertEqual(res_act.code, 200)
        data_act = json.loads(res_act.body.decode("utf-8"))
        self.assertIn("logs", data_act)

        # Historical logs
        res_hist = self.fetch("/api/logs/history?run=20261023_233349")
        self.assertEqual(res_hist.code, 200)
        data_hist = json.loads(res_hist.body.decode("utf-8"))
        self.assertIn("lines", data_hist)

    def test_index_html_serving(self) -> None:
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        html = response.body.decode("utf-8")
        self.assertIn("Semantic LSD Matching", html)
        self.assertIn("tab-history", html)
        self.assertIn("tab-train", html)
        self.assertIn("previewCurrentModel", html)
        self.assertIn("model-preview-modal", html)

    def test_train_live_html_serving(self) -> None:
        response = self.fetch("/train_live")
        self.assertEqual(response.code, 200)
        html = response.body.decode("utf-8")
        self.assertIn("实时训练监控", html)
        self.assertIn("stopLiveTraining", html)

    def test_review_html_serving(self) -> None:
        response = self.fetch("/review?run=20261026_145058")
        self.assertEqual(response.code, 200)
        html = response.body.decode("utf-8")
        self.assertIn("实验深度复盘", html)
        self.assertIn("eval-terminal-logs", html)
        self.assertIn("gallery-grid-container", html)

    def test_model_preview_endpoint(self) -> None:
        response = self.fetch("/api/model/preview?model=xfeat_line_matcher&pyramid=true&d_model=128")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertEqual(data["model_type"], "xfeat_line_matcher")
        self.assertIn("params_total", data)
        self.assertIn("input_shapes", data)
        self.assertIn("pipeline_stages", data)
        self.assertIn("diagram_url", data)
        self.assertIn("pytorch_repr", data)
        self.assertIn("XFeatLineMatcher", data["pytorch_repr"])
        self.assertIn("layer_graph", data)
        graph = data["layer_graph"]
        self.assertGreater(len(graph["nodes"]), 40)
        conv = next(node for node in graph["nodes"] if node["id"] == "s1_conv")
        self.assertEqual(conv["shape_out"], "(N, 32, 56×56)")
        self.assertIn("下采样", conv["meaning"])

    def test_semantic_line_preview_and_index_option(self) -> None:
        response = self.fetch("/api/model/preview?model=semantic_line_matcher&pyramid=true&d_model=128")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertEqual(data["model_type"], "semantic_line_matcher")
        self.assertIn("SemanticLineMatcher", data["pytorch_repr"])
        self.assertIn("layer_graph", data)
        self.assertGreater(len(data["layer_graph"]["nodes"]), 20)

        html = self.fetch("/").body.decode("utf-8")
        self.assertIn('value="semantic_line_matcher"', html)
        self.assertIn("train-semantic-weights", html)
        self.assertEqual(graph["param_count"], 1722081)


    def test_favicon_serving(self) -> None:
        response = self.fetch("/favicon.ico")
        self.assertEqual(response.code, 200)
        self.assertIn("image", response.headers.get("Content-Type", "").lower())


if __name__ == "__main__":
    unittest.main()
