"""Load pretrained DeepLabv3 from torchvision."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)

_BACKBONES: dict[str, tuple[Any, Any]] = {
    "resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights),
    "resnet101": (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights),
    "mobilenet_v3_large": (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    ),
}


def available_backbones() -> list[str]:
    return list(_BACKBONES.keys())


def build_deeplabv3(
    backbone: str = "resnet101",
    pretrained: bool = True,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, Any]:
    """
    Build DeepLabv3 and return (model, weights_or_None).

    When pretrained=True, COCO-with-VOC-labels weights are downloaded
    automatically on first use (cached under ~/.cache/torch/hub/checkpoints/).
    """
    key = backbone.lower()
    if key not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Choose from {available_backbones()}"
        )

    builder, weights_enum = _BACKBONES[key]
    weights = weights_enum.DEFAULT if pretrained else None
    model = builder(weights=weights)
    model.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, weights
