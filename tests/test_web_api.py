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
        self.assertIn("model_type", first)

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
        self.assertIn("tab-dashboard", html)
        self.assertIn("tab-train", html)


if __name__ == "__main__":
    unittest.main()
