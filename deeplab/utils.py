"""Preprocess, predict, and visualize DeepLabv3 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .voc_labels import VOC_CLASSES, VOC_COLORMAP


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image, weights: Any) -> torch.Tensor:
    """Apply the transforms bundled with the torchvision weights."""
    if weights is None:
        raise ValueError("pretrained weights are required for default transforms")
    return weights.transforms()(image)


@torch.inference_mode()
def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    weights: Any,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Run DeepLabv3 and return a HxW int64 mask of class ids.

    The mask is resized back to the original image size with nearest-neighbor.
    """
    if device is None:
        device = next(model.parameters()).device

    original_size = image.size  # (W, H)
    batch = preprocess_image(image, weights).unsqueeze(0).to(device)
    logits = model(batch)["out"][0]  # [C, h, w]
    mask_small = logits.argmax(0).byte().cpu().numpy()

    mask_img = Image.fromarray(mask_small, mode="L").resize(
        original_size, resample=Image.NEAREST
    )
    return np.asarray(mask_img, dtype=np.int64)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    """Map class ids to a PASCAL VOC palette RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(VOC_COLORMAP):
        rgb[mask == class_id] = color
    return Image.fromarray(rgb, mode="RGB")


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Blend original RGB image with a colorized mask."""
    color = colorize_mask(mask).convert("RGBA")
    base = image.convert("RGBA")
    blended = Image.blend(base, color, alpha=alpha)
    return blended.convert("RGB")


def summarize_mask(mask: np.ndarray) -> list[tuple[str, int]]:
    """Return (class_name, pixel_count) for classes present in the mask."""
    ids, counts = np.unique(mask, return_counts=True)
    rows: list[tuple[str, int]] = []
    for class_id, count in zip(ids.tolist(), counts.tolist()):
        name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else str(class_id)
        rows.append((name, int(count)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows
