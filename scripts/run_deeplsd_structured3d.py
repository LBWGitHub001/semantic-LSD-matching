"""Pipelined DeepLSD on Structured3D perspective views.

The GPU runs FP16 field inference while CPU workers turn finished fields
into line segments. Already-written fields.npz files are consumed by the
same workers and then deleted.

Outputs under vis_deeplsd/<scene>_<room>_v<id>/:
  lines.jpg    RGB with detected segments
  lines.npy    (N, 2, 2) endpoints in pixels
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEEPLSD_ROOT = ROOT / "third_party" / "DeepLSD"
if str(DEEPLSD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPLSD_ROOT))

from deeplsd.models.deeplsd_inference import DeepLSD


def view_dir(rgb_path: Path, out_root: Path) -> Path:
    scene = rgb_path.parts[-7]
    room = rgb_path.parts[-5]
    view = rgb_path.parts[-2]
    return out_root / f"{scene}_{room}_v{view}"


def output_paths(rgb_path: Path, out_root: Path) -> tuple[Path, Path, Path]:
    dest = view_dir(rgb_path, out_root)
    return dest / "fields.npz", dest / "lines.jpg", dest / "lines.npy"


def read_view(rgb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def load_model(weight: Path, device: torch.device) -> DeepLSD:
    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    net = DeepLSD({"detect_lines": False})
    net.load_state_dict(ckpt["model"])
    return net.to(device).eval()


def _forward_batch(net: DeepLSD, grays: list[np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    batch = np.stack(grays, axis=0)
    tensor = torch.from_numpy(batch)
    if device.type == "cuda":
        tensor = tensor.pin_memory()
    tensor = tensor.unsqueeze(1).to(device=device, dtype=torch.float32, non_blocking=True)
    tensor.div_(255.0)
    try:
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = net({"image": tensor})
            else:
                out = net({"image": tensor})
        df = out["df"].float().cpu().numpy().astype(np.float16)
        angle = out["line_level"].float().cpu().numpy().astype(np.float16)
        return df, angle
    finally:
        del tensor


def infer_batch(net: DeepLSD, grays: list[np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if not grays:
        empty = np.zeros((0, 0, 0), dtype=np.float16)
        return empty, empty
    try:
        return _forward_batch(net, grays, device)
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(grays) == 1:
            raise
        mid = max(1, len(grays) // 2)
        df_a, ang_a = infer_batch(net, grays[:mid], device)
        df_b, ang_b = infer_batch(net, grays[mid:], device)
        return np.concatenate([df_a, df_b], axis=0), np.concatenate([ang_a, ang_b], axis=0)


def detect_lines(gray: np.ndarray, df: np.ndarray, line_level: np.ndarray) -> np.ndarray:
    """CPU post-process matching DeepLSD.detect_afm_lines (normal filtering, no merge)."""
    if str(DEEPLSD_ROOT) not in sys.path:
        sys.path.insert(0, str(DEEPLSD_ROOT))
    from deeplsd.geometry.line_utils import filter_outlier_lines
    from deeplsd.utils.tensor import preprocess_angle
    from pytlsd import lsd

    img = gray.astype(np.float64)
    df64 = np.asarray(df, dtype=np.float64)
    level = np.asarray(line_level, dtype=np.float64)
    gradnorm = np.maximum(5.0 - df64, 0.0)
    angle = preprocess_angle(level - np.pi / 2, img, mask=True)[0]
    angle[gradnorm < 3] = -1024
    raw = lsd(img, scale=1.0, gradnorm=gradnorm, gradangle=angle, grad_nfa=True)
    if raw.size == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    lines = raw[:, :4].reshape(-1, 2, 2)
    filtered = filter_outlier_lines(
        img,
        lines[:, :, [1, 0]],
        df64,
        level - np.pi / 2,
        mode="inlier_thresh",
        use_grad=False,
        inlier_thresh=0.5,
        df_thresh=1.5,
        ang_thresh=np.pi / 9,
    )[0]
    if filtered.size == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.asarray(filtered[:, :, [1, 0]], dtype=np.float32)


def draw_lines(bgr: np.ndarray, lines: np.ndarray) -> np.ndarray:
    canvas = bgr.copy()
    for seg in lines:
        p0 = (int(round(seg[0, 0])), int(round(seg[0, 1])))
        p1 = (int(round(seg[1, 0])), int(round(seg[1, 1])))
        cv2.line(canvas, p0, p1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.line(canvas, p0, p1, (255, 180, 0), 1, cv2.LINE_AA)
    return canvas


def _write_lines(bgr: np.ndarray, lines: np.ndarray, jpg_path: str, npy_path: str) -> int:
    dest = Path(jpg_path).parent
    dest.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, lines)
    cv2.imwrite(jpg_path, draw_lines(bgr, lines), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return len(lines)


def postprocess_one(task: tuple) -> int:
    """Disk task reads fields.npz. Memory task already has the two fields."""
    kind = task[0]
    if kind == "disk":
        _, rgb_path, fields_path, jpg_path, npy_path, keep_fields = task
        bgr, gray = read_view(Path(rgb_path))
        with np.load(fields_path) as fields:
            df = fields["df"]
            angle = fields["angle"]
        count = _write_lines(bgr, detect_lines(gray, df, angle), jpg_path, npy_path)
        if not keep_fields:
            Path(fields_path).unlink(missing_ok=True)
        return count
    _, bgr, gray, df, angle, jpg_path, npy_path = task
    return _write_lines(bgr, detect_lines(gray, df, angle), jpg_path, npy_path)


def os_cpu_count() -> int:
    import os
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "Structured3D_perspective_full_00" / "Structured3D")
    parser.add_argument("--out", type=Path, default=ROOT / "Structured3D_perspective_full_00" / "vis_deeplsd")
    parser.add_argument("--weight", type=Path, default=DEEPLSD_ROOT / "weights" / "deeplsd_wireframe.tar")
    parser.add_argument("--batch-size", type=int, default=4, help="FP16 network batch. OOM splits it automatically.")
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--post-workers", type=int, default=0, help="0 uses up to 8 CPU workers.")
    parser.add_argument("--keep-fields", action="store_true", help="Keep fields.npz that were already on disk.")
    args = parser.parse_args()

    images = sorted(args.root.glob("scene_*/2D_rendering/*/perspective/full/*/rgb_rawlight.png"))
    args.out.mkdir(parents=True, exist_ok=True)
    done = 0
    disk_tasks = []
    pending = []
    for path in images:
        fields, jpg, npy = output_paths(path, args.out)
        if npy.is_file() and jpg.is_file():
            done += 1
            if fields.is_file() and not args.keep_fields:
                fields.unlink()
            continue
        if fields.is_file():
            disk_tasks.append(("disk", str(path), str(fields), str(jpg), str(npy), args.keep_fields))
        else:
            pending.append(path)

    workers = args.post_workers or os_cpu_count()
    total = len(disk_tasks) + len(pending)
    print(
        f"[*] pipeline: infer {len(pending)}, post from fields {len(disk_tasks)}, "
        f"done {done}, batch={args.batch_size}, post={workers}",
        flush=True,
    )
    if total == 0:
        print("[+] done", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args.weight, device) if pending else None
    progress = tqdm(total=total, desc="DeepLSD", unit="view", dynamic_ncols=True)
    read_pool = ThreadPoolExecutor(max_workers=args.read_workers) if pending else None
    batches = [pending[i : i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    max_inflight = max(workers * 2, 1)
    inflight: set = set()
    inferred = 0

    def harvest(block: bool) -> None:
        nonlocal inflight
        if not inflight:
            return
        if block:
            finished, inflight = wait_finished(inflight, block=True)
        else:
            finished = {item for item in inflight if item.done()}
            inflight -= finished
        for future in finished:
            count = future.result()
            progress.update(1)
            progress.set_postfix(lines=count, gpu=inferred, refresh=False)

    def wait_finished(pending_futures: set, block: bool) -> tuple[set, set]:
        from concurrent.futures import wait, FIRST_COMPLETED
        if not pending_futures:
            return set(), set()
        done, left = wait(pending_futures, timeout=0 if not block else None, return_when=FIRST_COMPLETED)
        if block and not done:
            done, left = wait(pending_futures, return_when=FIRST_COMPLETED)
        return done, left

    ctx = mp.get_context("spawn")
    disk_left = list(disk_tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        def pump_disk() -> None:
            while disk_left and len(inflight) < max_inflight:
                inflight.add(pool.submit(postprocess_one, disk_left.pop()))

        def submit_read(batch_paths: list[Path]):
            return batch_paths, [read_pool.submit(read_view, path) for path in batch_paths]

        queued = [submit_read(batch) for batch in batches[: args.prefetch_batches]]
        next_batch = args.prefetch_batches
        try:
            pump_disk()
            while queued:
                while len(inflight) >= max_inflight:
                    harvest(block=True)
                    pump_disk()
                harvest(block=False)
                pump_disk()
                batch_paths, futures = queued.pop(0)
                if next_batch < len(batches):
                    queued.append(submit_read(batches[next_batch]))
                    next_batch += 1
                loaded = [future.result() for future in futures]
                groups: dict[tuple[int, int], list[int]] = {}
                for index, (_, gray) in enumerate(loaded):
                    groups.setdefault(gray.shape, []).append(index)
                fields: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(loaded)
                for indexes in groups.values():
                    df, angle = infer_batch(net, [loaded[i][1] for i in indexes], device)
                    for slot, index in enumerate(indexes):
                        fields[index] = (df[slot], angle[slot])
                for path, (bgr, gray), pair in zip(batch_paths, loaded, fields):
                    _, jpg, npy = output_paths(path, args.out)
                    df, angle = pair
                    while len(inflight) >= max_inflight:
                        harvest(block=True)
                    inflight.add(pool.submit(postprocess_one, ("mem", bgr, gray, df, angle, str(jpg), str(npy))))
                inferred += len(batch_paths)
                progress.set_postfix(gpu=inferred, refresh=False)
                pump_disk()
            while disk_left:
                while len(inflight) >= max_inflight:
                    harvest(block=True)
                pump_disk()
                if disk_left and len(inflight) >= max_inflight:
                    harvest(block=True)
        finally:
            if read_pool is not None:
                read_pool.shutdown(wait=True)
            if net is not None:
                del net
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        while inflight:
            harvest(block=True)
    progress.close()
    print("[+] done", flush=True)


if __name__ == "__main__":
    main()
