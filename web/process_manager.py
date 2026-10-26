"""
Background Process and Task Manager for Semantic LSD Matching.
Handles training, evaluation, and dataset generation processes with real-time log streaming,
graceful cancellation (SIGINT), and status tracking.
"""

from __future__ import annotations

import asyncio
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
