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


def infer_fields(net, bgr: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Return the distance field and the line-angle field, both float32."""
    import cv2
    import torch

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray).to(device=device, dtype=torch.float32)[None, None]
    tensor.div_(255.0)
    with torch.inference_mode():
        output = net({"image": tensor})
    distance = output["df"][0].float().cpu().numpy()
    angle = output["line_level"][0].float().cpu().numpy()
    distance = np.nan_to_num(distance, nan=5.0, posinf=5.0, neginf=0.0).astype(np.float32)
    angle = np.mod(np.nan_to_num(angle, nan=0.0), np.pi).astype(np.float32)
    return distance, angle
