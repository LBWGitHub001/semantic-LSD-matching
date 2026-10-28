"""
Tornado Web Server for Semantic LSD Matching Management System.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tornado.ioloop
import tornado.web

from web.api_handlers import (
    ActiveLogsHandler,
    CheckpointsHandler,
    CompareRunsHandler,
    DatasetGenHandler,
    DatasetsHandler,
    EvalHandler,
    GalleryHandler,
    HistoricalLogsHandler,
    LogsWebSocketHandler,
    ModelPreviewHandler,
    QueueHandler,
    RunDetailHandler,
    RunsHandler,
    StatusHandler,
    TrainHandler,
    TrainUiConfigHandler,
)
from web.process_manager import ROOT_DIR

WEB_DIR = Path(__file__).resolve().parent


class MainHandler(tornado.web.RequestHandler):
    """Serves the main single-page dashboard HTML."""

    def get(self) -> None:
        html_path = WEB_DIR / "templates" / "index.html"
        if html_path.is_file():
            self.render(str(html_path))
        else:
            self.write("<h1>Semantic LSD Matching Management System</h1><p>index.html not found.</p>")


class TrainLiveHandler(tornado.web.RequestHandler):
    """Serves the standalone live training monitor page."""

    def get(self) -> None:
        html_path = WEB_DIR / "templates" / "train_live.html"
        if html_path.is_file():
            self.render(str(html_path))
        else:
            self.write("<h1>Training Monitor</h1><p>train_live.html not found.</p>")


class ReviewHandler(tornado.web.RequestHandler):
    """Serves the standalone experiment post-mortem review page."""

    def get(self) -> None:
        html_path = WEB_DIR / "templates" / "review.html"
        if html_path.is_file():
            self.render(str(html_path))
        else:
            self.write("<h1>Experiment Review</h1><p>review.html not found.</p>")


class FaviconHandler(tornado.web.RequestHandler):
    """Serves /favicon.ico to eliminate 404 console noise."""

    def get(self) -> None:
        fav_path = WEB_DIR / "static" / "favicon.ico"
        if fav_path.is_file():
            self.set_header("Content-Type", "image/x-icon")
            with open(fav_path, "rb") as f:
                self.write(f.read())
        else:
            self.set_header("Content-Type", "image/svg+xml")
            self.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">⚡</text></svg>')


class NoCacheStaticHandler(tornado.web.StaticFileHandler):
    """Serve UI assets without a sticky browser cache."""

    def set_extra_headers(self, path: str) -> None:
        self.set_header("Cache-Control", "no-cache, must-revalidate")


def make_app() -> tornado.web.Application:
    """Configures and returns the Tornado Application."""
    settings: dict[str, Any] = {
        "template_path": str(WEB_DIR / "templates"),
        "static_path": str(WEB_DIR / "static"),
        "debug": False,
        "autoreload": False,
    }

    handlers = [
        # Main UI
        (r"/", MainHandler),
        (r"/train_live", TrainLiveHandler),
        (r"/review", ReviewHandler),
        (r"/favicon.ico", FaviconHandler),

        # REST APIs
        (r"/api/status", StatusHandler),
        (r"/api/runs", RunsHandler),
        (r"/api/runs/compare", CompareRunsHandler),
        (r"/api/runs/(?P<run_name>[^/]+)", RunDetailHandler),
        (r"/api/checkpoints", CheckpointsHandler),
        (r"/api/datasets", DatasetsHandler),
        (r"/api/gallery", GalleryHandler),
        (r"/api/model/preview", ModelPreviewHandler),
        (r"/api/queue", QueueHandler),
        (r"/api/train", TrainHandler),
        (r"/api/ui/train-config", TrainUiConfigHandler),
        (r"/api/eval", EvalHandler),
        (r"/api/dataset", DatasetGenHandler),
        (r"/api/logs/active", ActiveLogsHandler),
        (r"/api/logs/history", HistoricalLogsHandler),

        # WebSocket Live Terminal
        (r"/ws/logs", LogsWebSocketHandler),

        # Static assets. No long-lived cache: the UI script is updated in place.
        (r"/static/(.*)", NoCacheStaticHandler, {"path": str(WEB_DIR / "static")}),
        # Files (checkpoints, outputs, dataset samples)
        (r"/files/(.*)", tornado.web.StaticFileHandler, {"path": str(ROOT_DIR)}),
    ]

    return tornado.web.Application(handlers, **settings)


def start_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Starts the Tornado web server."""
    app = make_app()
    try:
        app.listen(port, address=host)
    except OSError as e:
        if getattr(e, "errno", None) == 98:  # Address already in use
            print(f"\n[!] 端口冲突错误: 端口 {port} 已被占用 (Address already in use)。")
            print(f"    • 若已在后台运行服务，请直接在浏览器中访问: http://localhost:{port}")
            print(f"    • 若需释放端口并重启，可运行: fuser -k {port}/tcp")
            print(f"    • 若需更换端口启动，可传入参数: python app.py --port <新端口> (例如 7861)\n")
            sys.exit(1)
        raise e
    print("\n" + "=" * 70)
    print("   Semantic LSD Matching: Visual Management System (管理系统)  ")
    print("=" * 70)
    print(f"  • Local Web Interface  : http://localhost:{port}")
    if host == "0.0.0.0":
        print(f"  • Network Interface    : http://0.0.0.0:{port}")
    print(f"  • Project Directory    : {ROOT_DIR}")
    print("  • Press Ctrl+C in terminal to shutdown.")
    print("=" * 70 + "\n")
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    start_server()
