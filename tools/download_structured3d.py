#!/usr/bin/env python3
"""多线程断点续传下载 Structured3D 压缩包。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_URL = (
    "https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/"
    "Structured3D_perspective_full_00.zip"
)
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_THREADS = 8
MAX_RETRIES = 6


def _filename_from_url(url: str) -> str:
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    return name or "download.bin"


def _format_bytes(num: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num:.2f} B"


def _format_eta(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def probe(url: str, timeout: float) -> tuple[int, str | None, bool]:
    """返回 (文件大小, ETag, 是否支持 Range)。"""
    size: int | None = None
    etag: str | None = None
    ranged = False
    head = Request(url, method="HEAD")
    try:
        with urlopen(head, timeout=timeout) as resp:
            size = int(resp.headers["Content-Length"])
            etag = resp.headers.get("ETag")
            ranged = "bytes" in (resp.headers.get("Accept-Ranges") or "").lower()
    except (HTTPError, URLError, KeyError, ValueError, TimeoutError):
        size = None

    if size is not None and ranged:
        return size, etag, True

    req = Request(url, headers={"Range": "bytes=0-0"})
    with urlopen(req, timeout=timeout) as resp:
        content_range = resp.headers.get("Content-Range", "")
        if size is None:
            if "/" not in content_range:
                raise RuntimeError("服务器没有返回文件大小，无法分块下载")
            size = int(content_range.split("/")[-1])
        etag = etag or resp.headers.get("ETag")
        ranged = resp.status == 206 or ranged
    return size, etag, ranged


def _load_state(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(path: str, state: dict) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fetch_range(url: str, start: int, end: int, timeout: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(req, timeout=timeout) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"意外的 HTTP 状态 {resp.status}")
                data = resp.read()
            expected = end - start + 1
            if len(data) != expected:
                raise RuntimeError(f"分块长度不符：期望 {expected}，实际 {len(data)}")
            return data
        except Exception as exc:  # noqa: BLE001 - 网络错误统一重试
            last_error = exc
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"分块 {start}-{end} 下载失败：{last_error}") from last_error


def download(
    url: str,
    output: str,
    threads: int = DEFAULT_THREADS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = 120,
) -> str:
    size, etag, ranged = probe(url, timeout)
    if not ranged:
        raise RuntimeError("服务器不支持断点续传（Accept-Ranges）")
    if size <= 0:
        raise RuntimeError("文件大小无效")

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    state_path = output + ".progress.json"
    chunk_count = (size + chunk_size - 1) // chunk_size
    state = _load_state(state_path)
    same_object = (
        state is not None
        and state.get("url") == url
        and state.get("size") == size
        and state.get("etag") == etag
        and state.get("chunk_size") == chunk_size
    )
    if os.path.isfile(output) and os.path.getsize(output) == size and state is None:
        print(f"文件已存在且大小一致，跳过下载：{output}")
        return output
    if not same_object:
        state = {
            "url": url,
            "etag": etag,
            "size": size,
            "chunk_size": chunk_size,
            "done": [],
        }
        with open(output, "wb") as handle:
            handle.truncate(size)
        _save_state(state_path, state)
    elif not os.path.isfile(output) or os.path.getsize(output) != size:
        with open(output, "wb") as handle:
            handle.truncate(size)

    done = set(state["done"])
    pending = [index for index in range(chunk_count) if index not in done]
    if not pending:
        os.remove(state_path)
        print(f"已经下载完成：{output}")
        return output

    lock = threading.Lock()
    finished_bytes = len(done) * chunk_size
    if chunk_count - 1 in done:
        finished_bytes -= chunk_size - (size - (chunk_count - 1) * chunk_size)
    started = time.monotonic()
    baseline = finished_bytes
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(1):
            with lock:
                current = finished_bytes
            elapsed = max(time.monotonic() - started, 1e-6)
            speed = (current - baseline) / elapsed
            remain = (size - current) / speed if speed > 0 else float("inf")
            percent = current / size * 100
            line = (
                f"\r{percent:6.2f}%  {_format_bytes(current)} / {_format_bytes(size)}"
                f"  {_format_bytes(speed)}/s  ETA {_format_eta(remain)}"
                f"  线程 {threads}"
            )
            print(line, end="", flush=True)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()

    def worker(index: int) -> int:
        nonlocal finished_bytes
        start = index * chunk_size
        end = min(start + chunk_size, size) - 1
        data = _fetch_range(url, start, end, timeout)
        with open(output, "r+b") as handle:
            handle.seek(start)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with lock:
            done.add(index)
            state["done"] = sorted(done)
            _save_state(state_path, state)
            finished_bytes += len(data)
        return index

    failures: list[BaseException] = []
    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(worker, index) for index in pending]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:  # noqa: BLE001
                    failures.append(exc)
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    finally:
        stop.set()
        reporter.join(timeout=1)
        print()

    if failures:
        print(f"下载中断，已保存进度，可重新运行继续：{state_path}", file=sys.stderr)
        raise failures[0]

    if os.path.getsize(output) != size:
        raise RuntimeError("下载结束但文件大小不一致")
    os.remove(state_path)
    print(f"下载完成：{output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多线程断点续传下载 Structured3D 压缩包")
    parser.add_argument("--url", default=DEFAULT_URL, help="下载地址")
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="保存路径，默认 data/Structured3D/<文件名>",
    )
    parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS, help="线程数")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="每个分块的字节数，默认 8MiB",
    )
    parser.add_argument("--timeout", type=float, default=120, help="单次请求超时（秒）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        print("线程数至少为 1", file=sys.stderr)
        return 2
    if args.chunk_size < 1:
        print("分块大小至少为 1 字节", file=sys.stderr)
        return 2
    output = args.output or str(ROOT / "data" / "Structured3D" / _filename_from_url(args.url))
    try:
        download(args.url, output, args.threads, args.chunk_size, args.timeout)
    except KeyboardInterrupt:
        print("\n已中断。再次运行同一命令会从上次进度继续。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"下载失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
