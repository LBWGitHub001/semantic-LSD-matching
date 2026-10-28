"""
Background Process and Task Manager for Semantic LSD Matching.
Handles training, evaluation, and dataset generation processes with real-time log streaming,
graceful cancellation (SIGINT), and status tracking.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_conda_python() -> str:
    """Finds the best python interpreter, prioritizing the project's conda environment."""
    candidates = [
        "/home/lbw/Apps/anaconda3/envs/LSD-matching/bin/python",
        sys.executable,
        "python3",
    ]
    for c in candidates:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    return sys.executable


class ProcessTask:
    """Represents a single managed process task."""

    def __init__(
        self,
        task_id: str,
        task_type: str,  # 'train', 'eval', 'dataset'
        command: list[str],
        cwd: Path | str = ROOT_DIR,
        on_line: Optional[Callable[[str], None]] = None,
        on_finish: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.task_id = task_id
        self.task_type = task_type
        self.command = command
        self.cwd = Path(cwd)
        self.on_line = on_line
        self.on_finish = on_finish

        self.process: Optional[subprocess.Popen] = None
        self.status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
        self.exit_code: Optional[int] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.log_buffer: deque[str] = deque(maxlen=3000)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the process in a background thread."""
        self.status = "RUNNING"
        self.start_time = time.time()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Ensure matplotlib doesn't hang waiting for display
        env["MPLBACKEND"] = "Agg"

        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid,  # Create process group for clean signal handling
        )

        self._thread = threading.Thread(target=self._read_output, daemon=True)
        self._thread.start()

    def _read_output(self) -> None:
        """Reads stdout/stderr line by line."""
        if not self.process or not self.process.stdout:
            return

        for line in iter(self.process.stdout.readline, ""):
            # Strip trailing newline for clean display, keep content
            line_str = line.rstrip("\r\n")
            self.log_buffer.append(line_str)
            if self.on_line:
                try:
                    self.on_line(line_str)
                except Exception:
                    pass

        self.process.stdout.close()
        self.exit_code = self.process.wait()
        self.end_time = time.time()

        if self.status == "CANCELLED":
            pass
        elif self.exit_code == 0:
            self.status = "COMPLETED"
        else:
            self.status = "FAILED"

        if self.on_finish:
            try:
                self.on_finish(self.exit_code)
            except Exception:
                pass

    def stop(self, timeout: float = 8.0) -> bool:
        """Sends SIGINT to the process group to allow graceful cleanup and saving checkpoints."""
        if not self.process or self.process.poll() is not None:
            return True

        self.status = "CANCELLED"
        try:
            # Send SIGINT to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        except ProcessLookupError:
            return True
        except Exception as e:
            print(f"[ProcessManager] Error sending SIGINT: {e}")

        # Wait for graceful exit
        t_start = time.time()
        while time.time() - t_start < timeout:
            if self.process.poll() is not None:
                return True
            time.sleep(0.3)

        # Force terminate if still alive
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=2.0)
            return True
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-serializable status dictionary."""
        duration = 0.0
        if self.start_time:
            end = self.end_time or time.time()
            duration = round(end - self.start_time, 1)

        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "command": " ".join(self.command),
            "pid": self.process.pid if self.process else None,
            "exit_code": self.exit_code,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S") if self.start_time else None,
            "duration_sec": duration,
            "log_lines_count": len(self.log_buffer),
        }


class TaskManager:
    """Singleton task manager coordinates running jobs and broadcasting logs."""

    def __init__(self) -> None:
        self.active_task: Optional[ProcessTask] = None
        self.history_tasks: list[dict[str, Any]] = []
        self._subscribers: set[Any] = set()
        self._lock = threading.RLock()
        self._subscribers_lock = threading.Lock()

    def subscribe(self, subscriber: Any) -> None:
        with self._subscribers_lock:
            self._subscribers.add(subscriber)

    def unsubscribe(self, subscriber: Any) -> None:
        with self._subscribers_lock:
            self._subscribers.discard(subscriber)

    def broadcast(self, message: dict[str, Any]) -> None:
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for s in subscribers:
            try:
                s.send_message_sync(message)
            except Exception:
                with self._subscribers_lock:
                    self._subscribers.discard(s)

    def on_task_line(self, line: str) -> None:
        task_id = ""
        with self._lock:
            if self.active_task:
                task_id = self.active_task.task_id
        self.broadcast({
            "type": "log",
            "task_id": task_id,
            "line": line,
        })

    def on_task_finish(self, exit_code: int) -> None:
        task_info = None
        with self._lock:
            if self.active_task:
                task_info = self.active_task.to_dict()
                self.history_tasks.append(task_info)
        if task_info:
            self.broadcast({
                "type": "task_finished",
                "task": task_info,
            })
            if getattr(self, "queue_manager", None):
                self.queue_manager.on_task_completed(exit_code)

    def start_task(self, task_type: str, cmd_args: list[str]) -> tuple[bool, str, dict[str, Any]]:
        """Starts a new background task if none is currently running."""
        with self._lock:
            if self.active_task and self.active_task.status == "RUNNING":
                return False, f"Another task ({self.active_task.task_type}: {self.active_task.task_id}) is currently running.", {}

            python_bin = get_conda_python()
            # Pass -u to ensure unbuffered real-time stdout streaming
            full_cmd = [python_bin, "-u"] + cmd_args
            task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            task = ProcessTask(
                task_id=task_id,
                task_type=task_type,
                command=full_cmd,
                cwd=ROOT_DIR,
                on_line=self.on_task_line,
                on_finish=self.on_task_finish,
            )
            self.active_task = task
            task.start()
            task_dict = task.to_dict()

        # Broadcast outside the state lock to prevent any lock contention
        self.broadcast({
            "type": "task_started",
            "task": task_dict,
        })
        return True, "Task started successfully", task_dict

    def stop_current_task(self) -> tuple[bool, str]:
        """Stops the currently active task gracefully."""
        with self._lock:
            if not self.active_task or self.active_task.status != "RUNNING":
                return False, "No running task to stop."
            active = self.active_task

        stopped = active.stop(timeout=6.0)
        task_info = active.to_dict()
        self.broadcast({
            "type": "task_stopped",
            "task": task_info,
        })
        if getattr(self, "queue_manager", None):
            self.queue_manager.on_task_completed(exit_code=-1)
        return stopped, "Task stop signal sent."

    def get_status(self) -> dict[str, Any]:
        """Returns overall system and task status with full hardware metrics."""
        active = self.active_task.to_dict() if self.active_task else None
        res_info = self.get_system_resources()
        return {
            "active_task": active,
            "is_busy": bool(self.active_task and self.active_task.status == "RUNNING"),
            "resources": res_info,
            "gpu": res_info["gpu"],
            "python": get_conda_python(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_system_resources(self) -> dict[str, Any]:
        """Collects real-time CPU, RAM, Swap, and GPU metrics."""
        # 1. CPU instant percentage & load
        cpu_pct = 0.0
        try:
            with open("/proc/stat", "r") as f:
                first = f.readline()
            fields = [float(x) for x in first.split()[1:]]
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            total = sum(fields)
            if hasattr(self, "_prev_cpu") and self._prev_cpu:
                prev_tot, prev_idle = self._prev_cpu
                dt = total - prev_tot
                di = idle - prev_idle
                if dt > 0:
                    cpu_pct = round(max(0.0, min(100.0, 100.0 * (1.0 - (di / dt)))), 1)
            self._prev_cpu = (total, idle)
        except Exception:
            pass

        load1, load5, load15 = (0.0, 0.0, 0.0)
        try:
            load1, load5, load15 = os.getloadavg()
        except Exception:
            pass
        cpu_count = os.cpu_count() or 1

        # If instant sampling gave 0 on first call, fallback to load1 / cpu_count
        if cpu_pct <= 0.0 and load1 > 0:
            cpu_pct = min(100.0, round((load1 / cpu_count) * 100, 1))

        # 2. RAM & Swap from /proc/meminfo
        ram_total_gb = 0.0
        ram_used_gb = 0.0
        ram_pct = 0.0
        swap_total_gb = 0.0
        swap_used_gb = 0.0
        swap_pct = 0.0
        try:
            mem = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].split()[0])
            tot = mem.get("MemTotal", 0)
            avail = mem.get("MemAvailable", 0)
            used = tot - avail
            ram_total_gb = round(tot / (1024 * 1024), 2)
            ram_used_gb = round(used / (1024 * 1024), 2)
            ram_pct = round((used / tot) * 100, 1) if tot > 0 else 0

            sw_tot = mem.get("SwapTotal", 0)
            sw_free = mem.get("SwapFree", 0)
            sw_used = sw_tot - sw_free
            swap_total_gb = round(sw_tot / (1024 * 1024), 2)
            swap_used_gb = round(sw_used / (1024 * 1024), 2)
            swap_pct = round((sw_used / sw_tot) * 100, 1) if sw_tot > 0 else 0
        except Exception:
            pass

        # 3. GPU from nvidia-smi
        gpu_info = self.get_gpu_info()

        return {
            "cpu": {
                "pct": cpu_pct,
                "count": cpu_count,
                "load_1m": round(load1, 2),
                "load_5m": round(load5, 2),
                "load_15m": round(load15, 2),
            },
            "ram": {
                "total_gb": ram_total_gb,
                "used_gb": ram_used_gb,
                "pct": ram_pct,
                "swap_total_gb": swap_total_gb,
                "swap_used_gb": swap_used_gb,
                "swap_pct": swap_pct,
            },
            "gpu": gpu_info,
        }

    def get_gpu_info(self) -> dict[str, Any]:
        """Retrieves GPU status via nvidia-smi if available."""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split("\n")[0].split(",")]
                if len(parts) >= 5:
                    total = float(parts[1])
                    used = float(parts[2])
                    power = float(parts[5]) if len(parts) >= 6 else 0.0
                    return {
                        "available": True,
                        "name": parts[0],
                        "memory_total_mb": total,
                        "memory_used_mb": used,
                        "memory_pct": round((used / total) * 100, 1) if total > 0 else 0,
                        "utilization_pct": float(parts[3]),
                        "temperature_c": float(parts[4]),
                        "power_w": round(power, 1),
                    }
        except Exception:
            pass
        return {"available": False, "name": "CPU Mode"}

    def get_active_logs(self) -> list[str]:
        if self.active_task:
            return list(self.active_task.log_buffer)
        return []


# Global singleton instance
TASK_MANAGER = TaskManager()


class QueuedTrainingTask:
    """Represents a scheduled or queued training task with user label and config."""

    def __init__(
        self,
        task_id: str,
        label: str,
        config: dict[str, Any],
        status: str = "PENDING",
        current_stage: str = "IDLE",  # IDLE, TRAIN, EVAL, DONE
        created_at: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        exit_code: Optional[int] = None,
        error_msg: Optional[str] = None,
    ) -> None:
        self.task_id = task_id
        self.label = label.strip() if label else ""
        self.config = config
        self.status = status  # PENDING, RUNNING, COMPLETED, CANCELLED, FAILED
        self.current_stage = current_stage
        self.created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.started_at = started_at
        self.finished_at = finished_at
        self.exit_code = exit_code
        self.error_msg = error_msg

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "label": self.label,
            "config": self.config,
            "status": self.status,
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error_msg": self.error_msg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueuedTrainingTask:
        return cls(
            task_id=data.get("task_id", ""),
            label=data.get("label", ""),
            config=data.get("config", {}),
            status=data.get("status", "PENDING"),
            current_stage=data.get("current_stage", "IDLE"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            exit_code=data.get("exit_code"),
            error_msg=data.get("error_msg"),
        )


class TaskQueueManager:
    """Coordinates the persistent training task queue and sequential execution."""

    def __init__(self, storage_path: Path | str = ROOT_DIR / "outputs" / "training_queue.json") -> None:
        self.storage_path = Path(storage_path)
        self.queue: list[QueuedTrainingTask] = []
        self.auto_run: bool = False
        self.current_queue_task_id: Optional[str] = None
        self._lock = threading.RLock()
        self.load_from_disk()

    def load_from_disk(self) -> None:
        with self._lock:
            if not self.storage_path.is_file():
                return
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.auto_run = data.get("auto_run", False)
                    items = data.get("queue", [])
                    self.queue = [QueuedTrainingTask.from_dict(d) for d in items]
                    for t in self.queue:
                        if t.status == "RUNNING":
                            t.status = "CANCELLED"
            except Exception as e:
                print(f"[!] Failed to load training queue from disk: {e}")

    def save_to_disk(self) -> None:
        with self._lock:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "auto_run": self.auto_run,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "queue": [t.to_dict() for t in self.queue],
                }
                with open(self.storage_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[!] Failed to save training queue to disk: {e}")

    def add_task(self, label: str, config: dict[str, Any]) -> QueuedTrainingTask:
        with self._lock:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_id = f"task_{ts}_{len(self.queue) + 1}"
            task = QueuedTrainingTask(task_id=task_id, label=label, config=config)
            self.queue.append(task)
            self.save_to_disk()
            self._broadcast_update()

            if self.auto_run:
                threading.Thread(target=self.dispatch_next, daemon=True).start()
            return task

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            for i, t in enumerate(self.queue):
                if t.task_id == task_id:
                    if t.status == "RUNNING":
                        TASK_MANAGER.stop_current_task()
                    del self.queue[i]
                    self.save_to_disk()
                    self._broadcast_update()
                    return True
            return False

    def reorder_task(self, task_id: str, direction: str) -> bool:
        with self._lock:
            idx = -1
            for i, t in enumerate(self.queue):
                if t.task_id == task_id:
                    idx = i
                    break
            if idx == -1:
                return False

            target_idx = idx - 1 if direction == "up" else idx + 1
            if 0 <= target_idx < len(self.queue):
                self.queue[idx], self.queue[target_idx] = self.queue[target_idx], self.queue[idx]
                self.save_to_disk()
                self._broadcast_update()
                return True
            return False

    def clear_pending(self) -> int:
        with self._lock:
            orig_len = len(self.queue)
            self.queue = [t for t in self.queue if t.status in ("RUNNING", "COMPLETED")]
            cleared = orig_len - len(self.queue)
            self.save_to_disk()
            self._broadcast_update()
            return cleared

    def set_auto_run(self, enabled: bool) -> bool:
        with self._lock:
            self.auto_run = enabled
            self.save_to_disk()
            self._broadcast_update()
            if self.auto_run:
                threading.Thread(target=self.dispatch_next, daemon=True).start()
            return self.auto_run

    def run_now(self, task_id: str) -> tuple[bool, str, dict[str, Any]]:
        with self._lock:
            target_task = None
            for t in self.queue:
                if t.task_id == task_id:
                    target_task = t
                    break
            if not target_task:
                return False, f"Task {task_id} not found in queue", {}

            if target_task.status == "RUNNING":
                return False, "Task is already running", {}

            return self._execute_task(target_task)

    def _execute_task(self, task: QueuedTrainingTask) -> tuple[bool, str, dict[str, Any]]:
        cfg = task.config
        model = cfg.get("model", "xfeat_line_matcher")
        data_dir = cfg.get("data_dir", "data/synthetic_cad")
        epochs = cfg.get("epochs", 15)
        lr = cfg.get("lr", 0.0005)
        dustbin = cfg.get("dustbin_weight", 0.1)
        motion = cfg.get("motion_weight", 0.1)
        region_weight = cfg.get("region_weight", 1.0)
        line_weight = cfg.get("line_weight", 1.0)
        sem_weight = cfg.get("sem_weight", 0.1)
        geo_weight = cfg.get("geo_weight", 0.2)
        min_area = cfg.get("min_area_ratio", 0.40)
        data_ratio = cfg.get("data_ratio", 1.0)
        pyramid = cfg.get("pyramid", True)
        amp = cfg.get("amp", True)
        label = task.label or cfg.get("label", "")
        is_semantic = str(model) in ("semantic_line_matcher", "semantic_line")

        cmd = [
            "train.py",
            "--model", str(model),
            "--data-dir", str(data_dir),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--dustbin-weight", str(dustbin),
            "--motion-weight", str(0.0 if is_semantic else motion),
            "--min-area-ratio", str(min_area),
            "--data-ratio", str(data_ratio),
        ]
        if is_semantic:
            cmd.extend([
                "--region-weight", str(region_weight),
                "--line-weight", str(line_weight),
                "--sem-weight", str(sem_weight),
                "--geo-weight", str(geo_weight),
            ])
        if pyramid:
            cmd.append("--pyramid")
        else:
            cmd.append("--no-pyramid")
        if amp:
            cmd.append("--amp")
        if label:
            cmd.extend(["--label", str(label)])

        ok, msg, task_dict = TASK_MANAGER.start_task("train", cmd)
        if ok:
            task.status = "RUNNING"
            task.current_stage = "TRAIN"
            task.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_queue_task_id = task.task_id
            self.save_to_disk()
            self._broadcast_update()
        else:
            task.status = "FAILED"
            task.current_stage = "DONE"
            task.error_msg = msg
            self.save_to_disk()
            self._broadcast_update()
        return ok, msg, task_dict

    def dispatch_next(self) -> None:
        """Finds next PENDING task in queue and executes it if no task is currently active."""
        with self._lock:
            if TASK_MANAGER.active_task and TASK_MANAGER.active_task.status == "RUNNING":
                return

            pending_task = None
            for t in self.queue:
                if t.status == "PENDING":
                    pending_task = t
                    break

            if pending_task:
                self._execute_task(pending_task)

    def on_task_completed(self, exit_code: int) -> None:
        with self._lock:
            current_task = None
            if self.current_queue_task_id:
                for t in self.queue:
                    if t.task_id == self.current_queue_task_id:
                        current_task = t
                        break

            if current_task is not None:
                # 1. If currently in TRAIN stage
                if current_task.current_stage == "TRAIN":
                    if exit_code == 0:
                        # Training succeeded -> advance to EVAL stage
                        current_task.current_stage = "EVAL"
                        self.save_to_disk()
                        self._broadcast_update()

                        # Launch automated evaluation and visualization in background thread
                        threading.Thread(
                            target=self._start_eval_for_task,
                            args=(current_task,),
                            daemon=True,
                        ).start()
                        return
                    else:
                        # Training failed or was cancelled
                        is_cancelled = exit_code in (-15, -9, 130, 143)
                        current_task.status = "CANCELLED" if is_cancelled else "FAILED"
                        current_task.current_stage = "DONE"
                        current_task.exit_code = exit_code
                        current_task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.current_queue_task_id = None
                        self.save_to_disk()
                        self._broadcast_update()

                # 2. If currently in EVAL stage
                elif current_task.current_stage == "EVAL":
                    is_cancelled = exit_code in (-15, -9, 130, 143)
                    current_task.status = "CANCELLED" if is_cancelled else "COMPLETED"
                    current_task.current_stage = "DONE"
                    current_task.exit_code = exit_code
                    current_task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.current_queue_task_id = None
                    self.save_to_disk()
                    self._broadcast_update()

                    # Trigger curves and review update
                    try:
                        from review_runs import render_run_curves
                        model_name = current_task.config.get("model", "xfeat_line_matcher")
                        latest_dir = ROOT_DIR / "checkpoints" / model_name / "latest"
                        if latest_dir.exists():
                            render_run_curves(latest_dir.resolve())
                    except Exception as e:
                        print(f"[!] Warning: Failed to render curves after eval: {e}")

            if self.auto_run:
                threading.Timer(2.0, self.dispatch_next).start()

    def _start_eval_for_task(self, task: QueuedTrainingTask) -> None:
        """Automatically triggers benchmark evaluation and visualization sample generation."""
        import time
        time.sleep(1.0)  # Sync delay

        model_name = task.config.get("model", "xfeat_line_matcher")
        data_dir = task.config.get("data_dir", "data/xfeat_line_matcher")

        # Locate checkpoint generated from the training run
        latest_dir = ROOT_DIR / "checkpoints" / model_name / "latest"
        ckpt_path = None
        if latest_dir.exists():
            resolved_dir = latest_dir.resolve()
            if (resolved_dir / "best_model.pth").is_file():
                ckpt_path = resolved_dir / "best_model.pth"
            elif (resolved_dir / "last_model.pth").is_file():
                ckpt_path = resolved_dir / "last_model.pth"
            else:
                pths = sorted(list(resolved_dir.glob("*.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
                if pths:
                    ckpt_path = pths[0]

        if ckpt_path is None:
            base_dir = ROOT_DIR / "checkpoints" / model_name
            if base_dir.exists():
                all_pths = sorted(list(base_dir.glob("**/*.pth")), key=lambda p: p.stat().st_mtime, reverse=True)
                if all_pths:
                    ckpt_path = all_pths[0]

        if ckpt_path is None:
            print(f"[!] Warning: No checkpoint found for task {task.task_id}, completing task directly.")
            with self._lock:
                task.status = "COMPLETED"
                task.current_stage = "DONE"
                task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.current_queue_task_id = None
                self.save_to_disk()
                self._broadcast_update()
            if self.auto_run:
                threading.Timer(2.0, self.dispatch_next).start()
            return

        cmd = [
            "eval.py",
            "--checkpoint", str(ckpt_path),
            "--data-dir", str(data_dir),
            "--split", "test",
            "--num-vis", "5",
            "--out-dir", "outputs/semantic_matching_vis",
        ]

        ok, msg, _ = TASK_MANAGER.start_task("eval", cmd)
        if not ok:
            print(f"[!] Warning: Failed to launch automated eval task: {msg}")
            with self._lock:
                task.status = "COMPLETED"
                task.current_stage = "DONE"
                task.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.current_queue_task_id = None
                self.save_to_disk()
                self._broadcast_update()
            if self.auto_run:
                threading.Timer(2.0, self.dispatch_next).start()

    def _broadcast_update(self) -> None:
        TASK_MANAGER.broadcast({
            "type": "queue_updated",
            "queue_data": self.get_queue_data(),
        })

    def get_queue_data(self) -> dict[str, Any]:
        with self._lock:
            pending_count = sum(1 for t in self.queue if t.status == "PENDING")
            running_task = next((t.to_dict() for t in self.queue if t.status == "RUNNING"), None)
            today_str = datetime.now().strftime("%Y-%m-%d")
            completed_today = sum(1 for t in self.queue if t.status == "COMPLETED" and t.finished_at and t.finished_at.startswith(today_str))
            return {
                "auto_run": self.auto_run,
                "current_task": running_task,
                "pending_count": pending_count,
                "completed_count": completed_today,
                "total_count": len(self.queue),
                "tasks": [t.to_dict() for t in self.queue],
            }


TASK_QUEUE_MANAGER = TaskQueueManager()
TASK_MANAGER.queue_manager = TASK_QUEUE_MANAGER

