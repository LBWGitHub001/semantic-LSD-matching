"""
Tornado Web Server for Semantic LSD Matching Management System.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
    RunDetailHandler,
    RunsHandler,
    StatusHandler,
    TrainHandler,
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

        # REST APIs
        (r"/api/status", StatusHandler),
        (r"/api/runs", RunsHandler),
        (r"/api/runs/compare", CompareRunsHandler),
        (r"/api/runs/(?P<run_name>[^/]+)", RunDetailHandler),
        (r"/api/checkpoints", CheckpointsHandler),
        (r"/api/datasets", DatasetsHandler),
        (r"/api/gallery", GalleryHandler),
        (r"/api/train", TrainHandler),
        (r"/api/eval", EvalHandler),
        (r"/api/dataset", DatasetGenHandler),
        (r"/api/logs/active", ActiveLogsHandler),
        (r"/api/logs/history", HistoricalLogsHandler),

        # WebSocket Live Terminal
        (r"/ws/logs", LogsWebSocketHandler),

        # Static assets
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": str(WEB_DIR / "static")}),
        # Files (checkpoints, outputs, dataset samples)
        (r"/files/(.*)", tornado.web.StaticFileHandler, {"path": str(ROOT_DIR)}),
    ]

    return tornado.web.Application(handlers, **settings)


def start_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Starts the Tornado web server."""
    app = make_app()
    app.listen(port, address=host)
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
