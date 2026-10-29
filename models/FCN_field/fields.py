"""DeepLSD network outputs used as the two extra input channels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEEPLSD_ROOT = ROOT / "third_party" / "DeepLSD"
DEFAULT_WEIGHT = DEEPLSD_ROOT / "weights" / "deeplsd_wireframe.tar"


def load_deeplsd(weight: Path, device):
    import sys
    import torch

    if str(DEEPLSD_ROOT) not in sys.path:
        sys.path.insert(0, str(DEEPLSD_ROOT))
    from deeplsd.models.deeplsd_inference import DeepLSD
    checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    net = DeepLSD({"detect_lines": False})
    net.load_state_dict(checkpoint["model"])
    return net.to(device).eval()


def _clean_fields(distance: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.nan_to_num(distance, nan=5.0, posinf=5.0, neginf=0.0).astype(np.float32)
    angle = np.mod(np.nan_to_num(angle, nan=0.0), np.pi).astype(np.float32)
    return distance, angle


def infer_fields_batch(net, bgrs: list[np.ndarray], device) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run the distance and angle heads on a batch. OOM splits the batch."""
    import cv2
    import torch

    if not bgrs:
        return []
    if len({image.shape for image in bgrs}) != 1:
        return [infer_fields(net, image, device) for image in bgrs]
    gray = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in bgrs])
    tensor = torch.from_numpy(gray).to(device=device, dtype=torch.float32).unsqueeze(1)
    tensor.div_(255.0)
    try:
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = net({"image": tensor})
            else:
                output = net({"image": tensor})
        distances = output["df"].float().cpu().numpy()
        angles = output["line_level"].float().cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(bgrs) == 1:
            raise
        mid = max(1, len(bgrs) // 2)
        return infer_fields_batch(net, bgrs[:mid], device) + infer_fields_batch(net, bgrs[mid:], device)
    return [_clean_fields(distances[index], angles[index]) for index in range(len(bgrs))]


def infer_fields(net, bgr: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Return the distance field and the line-angle field, both float32."""
    return infer_fields_batch(net, [bgr], device)[0]
