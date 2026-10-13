"""DeepLabv3 semantic segmentation (torchvision)."""

from .model import build_deeplabv3, available_backbones

__all__ = ["build_deeplabv3", "available_backbones"]
