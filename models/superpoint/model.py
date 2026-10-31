"""SuperPoint encoder with descriptor and endpoint detector heads.

The shared encoder follows the public Magic Leap network.
- Descriptor head: predicts 256-D semi-dense metric descriptors on an 8x8 grid.
- Keypoint detector head: predicts 65-channel semi-dense logits (64 cell pixels + 1 dustbin),
  unfolded via softmax and pixel_shuffle into full-resolution probability heatmaps (B, 1, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPoint(nn.Module):
    """SuperPoint 特征描述子与直线端点联合预测网络。输入是 (N, 5, H, W)，高和宽都要能被 8 整除。

    五个通道依次是 RGB/255、归一化对数距离场/5、角度场/π。
    编码器做三次 2×2 池化，一个格子对应原图上的 8×8。
    - 描述子头: 输出每个格子 256 维特征，再双三次插值回原图大小并做 L2 归一化；
    - 端点预测头: 输出 65 通道半稠密热力图，Softmax 去除 Dustbin 后通过 pixel_shuffle 恢复 (N, 1, H, W) 概率图。
    """

    def __init__(self, descriptor_dim: int = 256, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.descriptor_dim = int(descriptor_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(5, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # 描述子分支
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0)

        # 端点/角点预测分支 (仿照 SuperPoint 65 通道: 64 像素格子 + 1 dustbin 通道)
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.shape[1] != 5:
            raise ValueError("SuperPoint 接受 5 通道：RGB、距离场、角度场")
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError("高和宽必须能被 8 整除")
        features = self._encode(image)

        # 1. 描述子分支
        desc = self.convDb(self.relu(self.convDa(features)))
        desc = F.normalize(desc, p=2, dim=1)

        # 2. 端点预测分支
        semi = self.convPb(self.relu(self.convPa(features)))
        prob = F.softmax(semi, dim=1)[:, :-1]  # 剔除第 65 维 Dustbin -> (B, 64, Hc, Wc)
        heatmap = F.pixel_shuffle(prob, 8)     # 重排为全分辨率概率图 -> (B, 1, H, W)

        return {
            "desc": desc,
            "descriptor": upsample_descriptors(desc),
            "semi": semi,
            "heatmap": heatmap,
        }

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x


def upsample_descriptors(desc: torch.Tensor) -> torch.Tensor:
    """半稠密描述子双三次插值回原图，再做一次 L2 归一化。"""
    dense = F.interpolate(desc, scale_factor=8, mode="bicubic", align_corners=True)
    return F.normalize(dense, p=2, dim=1)
