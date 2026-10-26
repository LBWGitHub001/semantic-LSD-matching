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
            if r["run_name"] == run_name or run_name in str(r["run_dir"]):
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

        self.write_json({
            "run_name": matched["run_name"],
            "model_type": matched["model_type"],
            "run_dir": str(r_dir),
            "config": cfg,
            "history": history,
            "report_markdown": report_content,
            "curves_url": curves_rel,
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
        m1 = next((r for r in all_runs if r["run_name"] == r1 or r1 in str(r["run_dir"])), None)
        m2 = next((r for r in all_runs if r["run_name"] == r2 or r2 in str(r["run_dir"])), None)

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
    """GET /api/gallery - Lists visualization images from outputs/ and runs."""

    def get(self) -> None:
        outputs_dir = ROOT_DIR / "outputs" / "semantic_matching_vis"
        images = []

        if outputs_dir.is_dir():
            for img_p in sorted(outputs_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True):
                name = img_p.name
                rel_url = f"/files/outputs/semantic_matching_vis/{name}"
                size_kb = round(img_p.stat().st_size / 1024, 1)

                images.append({
                    "name": name,
                    "url": rel_url,
                    "size_kb": size_kb,
                    "modified": img_p.stat().st_mtime,
                })

        self.write_json({"images": images, "total": len(images)})


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
        use_pyramid = body.get("pyramid", True)
        use_amp = body.get("amp", True)
        save_interval = body.get("save_interval", 5)
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
            "--save-interval", str(save_interval),
        ]
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
