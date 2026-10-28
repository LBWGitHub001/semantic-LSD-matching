"""
Unit and Integration Tests for Training Queue API and Label Management.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import tornado.testing
from web.server import make_app
from web.process_manager import TASK_QUEUE_MANAGER
from review_runs import format_run_tag


class TestQueueApi(tornado.testing.AsyncHTTPTestCase):
    def get_app(self) -> tornado.web.Application:
        return make_app()

    def setUp(self) -> None:
        super().setUp()
        # Save auto_run state and queue items
        self._orig_auto_run = TASK_QUEUE_MANAGER.auto_run
        self._orig_queue = list(TASK_QUEUE_MANAGER.queue)
        TASK_QUEUE_MANAGER.set_auto_run(False)
        TASK_QUEUE_MANAGER.queue.clear()

    def tearDown(self) -> None:
        TASK_QUEUE_MANAGER.queue = list(self._orig_queue)
        TASK_QUEUE_MANAGER.set_auto_run(self._orig_auto_run)
        super().tearDown()

    def test_queue_get(self) -> None:
        response = self.fetch("/api/queue")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body.decode("utf-8"))
        self.assertIn("auto_run", data)
        self.assertIn("pending_count", data)
        self.assertIn("tasks", data)
        self.assertIsInstance(data["tasks"], list)

    def test_queue_add_and_delete(self) -> None:
        # 1. Add task with custom label
        payload = {
            "model": "xfeat_line_matcher",
            "data_dir": "data/xfeat_line_matcher",
            "epochs": 10,
            "lr": 0.0003,
            "dustbin_weight": 0.15,
            "motion_weight": 0.35,
            "min_area_ratio": 0.40,
            "data_ratio": 0.8,
            "pyramid": True,
            "amp": True,
            "label": "test_label_experiment",
        }
        res_add = self.fetch(
            "/api/queue?action=add",
            method="POST",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_add.code, 200)
        data_add = json.loads(res_add.body.decode("utf-8"))
        self.assertTrue(data_add["success"])
        task = data_add["task"]
        self.assertEqual(task["label"], "test_label_experiment")
        self.assertEqual(task["status"], "PENDING")
        self.assertEqual(task["current_stage"], "IDLE")
        self.assertEqual(task["config"]["epochs"], 10)
        self.assertEqual(task["config"]["data_ratio"], 0.8)
        task_id = task["task_id"]

        # 2. Check it appears in GET /api/queue
        res_get = self.fetch("/api/queue")
        self.assertEqual(res_get.code, 200)
        data_get = json.loads(res_get.body.decode("utf-8"))
        self.assertEqual(data_get["pending_count"], 1)
        found = any(t["task_id"] == task_id for t in data_get["tasks"])
        self.assertTrue(found)

        # 3. Delete task
        res_del = self.fetch(
            "/api/queue?action=delete",
            method="POST",
            body=json.dumps({"task_id": task_id}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_del.code, 200)
        data_del = json.loads(res_del.body.decode("utf-8"))
        self.assertTrue(data_del["success"])
        self.assertEqual(data_del["queue_data"]["pending_count"], 0)

    def test_queue_reorder(self) -> None:
        # Add two tasks
        res1 = self.fetch(
            "/api/queue?action=add",
            method="POST",
            body=json.dumps({"model": "mask_line", "label": "task_A"}),
            headers={"Content-Type": "application/json"},
        )
        t1_id = json.loads(res1.body.decode("utf-8"))["task"]["task_id"]

        res2 = self.fetch(
            "/api/queue?action=add",
            method="POST",
            body=json.dumps({"model": "xfeat_line_matcher", "label": "task_B"}),
            headers={"Content-Type": "application/json"},
        )
        t2_id = json.loads(res2.body.decode("utf-8"))["task"]["task_id"]

        # Initial order: [t1_id, t2_id]
        res_get = self.fetch("/api/queue")
        tasks = json.loads(res_get.body.decode("utf-8"))["tasks"]
        self.assertEqual([t["task_id"] for t in tasks], [t1_id, t2_id])

        # Move t2_id up
        res_reorder = self.fetch(
            "/api/queue?action=reorder",
            method="POST",
            body=json.dumps({"task_id": t2_id, "direction": "up"}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_reorder.code, 200)
        data_reorder = json.loads(res_reorder.body.decode("utf-8"))
        self.assertTrue(data_reorder["success"])
        tasks_after = data_reorder["queue_data"]["tasks"]
        self.assertEqual([t["task_id"] for t in tasks_after], [t2_id, t1_id])

    def test_queue_toggle_auto_and_clear(self) -> None:
        # 1. Toggle auto on then off
        res_toggle = self.fetch(
            "/api/queue?action=toggle_auto",
            method="POST",
            body=json.dumps({"enabled": True}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_toggle.code, 200)
        data_toggle = json.loads(res_toggle.body.decode("utf-8"))
        self.assertTrue(data_toggle["auto_run"])

        # Turn auto off before adding pending task
        res_toggle_off = self.fetch(
            "/api/queue?action=toggle_auto",
            method="POST",
            body=json.dumps({"enabled": False}),
            headers={"Content-Type": "application/json"},
        )
        self.assertFalse(json.loads(res_toggle_off.body.decode("utf-8"))["auto_run"])

        # 2. Add pending task while auto_run is False
        self.fetch(
            "/api/queue?action=add",
            method="POST",
            body=json.dumps({"model": "line_transformer", "label": "task_to_clear"}),
            headers={"Content-Type": "application/json"},
        )
        res_clear = self.fetch(
            "/api/queue?action=clear_pending",
            method="POST",
            body=json.dumps({}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(res_clear.code, 200)
        data_clear = json.loads(res_clear.body.decode("utf-8"))
        self.assertEqual(data_clear["cleared_count"], 1)
        self.assertEqual(data_clear["queue_data"]["pending_count"], 0)

    def test_run_tag_formatting_with_label(self) -> None:
        # Test review_runs.format_run_tag with and without label
        cfg_with_label = {
            "model": {"model_type": "xfeat_line_matcher", "use_pyramid": True, "d_model": 128},
            "dataset": {"dataset_dir": "data/synthetic_cad"},
            "experiment_info": {
                "label": "ablation_v3",
            },
        }
        tag_with_label = format_run_tag(cfg_with_label, "20261026_214500")
        self.assertIn("【ablation_v3】", tag_with_label)
        self.assertIn("【20261026_214500】", tag_with_label)
        self.assertTrue(tag_with_label.startswith("xfeat_line_matcher+pyramid_d128+synthetic_cad"))

        cfg_without_label = {
            "model": {"model_type": "mask_line", "use_pyramid": False, "d_model": 128},
            "dataset": {"dataset_dir": "data/synthetic_cad"},
            "experiment_info": {},
        }
        tag_without_label = format_run_tag(cfg_without_label, "20261026_214500")
        self.assertNotIn("【】", tag_without_label)
        self.assertIn("【20261026_214500】", tag_without_label)
        self.assertTrue(tag_without_label.startswith("mask_line+no_pyramid_d128+synthetic_cad"))


if __name__ == "__main__":
    unittest.main()
