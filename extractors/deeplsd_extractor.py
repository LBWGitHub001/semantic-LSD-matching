"""DeepLSD 几何距离场与高精度线段提取器 (DeepLSD Geometric Extractor).

封装 DeepLSD (CVPR 2023) 官方推理模型:
1. 预测亚像素几何距离场 (Distance Field, DF) 与最近线角度场 (Angle Field, AF);
2. 结合 AFM (Attraction Field Map) 与 LSD 提取几何直线段 [M, 2, 2];
3. 计算线段几何置信度评分 (Line Confidence Scores)。
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEEPLSD_DIR = ROOT / "third_party" / "DeepLSD"
if str(DEEPLSD_DIR) not in sys.path:
    sys.path.insert(0, str(DEEPLSD_DIR))

try:
    from deeplsd.models.deeplsd_inference import DeepLSD
except ImportError as e:
    DeepLSD = None
    _import_err = e


class DeepLSDExtractor:
    """DeepLSD 几何线段与距离场提取引擎。"""

    def __init__(
        self,
        checkpoint_path: str | Path = "third_party/DeepLSD/weights/deeplsd_wireframe.tar",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        detect_lines: bool = True,
        min_line_len: float = 10.0,
    ) -> None:
        if DeepLSD is None:
            raise ImportError(f"无法导入 DeepLSD 模型: {_import_err}")

        self.device = torch.device(device)
        self.detect_lines = detect_lines
        self.min_line_len = min_line_len
        self.ckpt_path = Path(checkpoint_path).resolve()

        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"未能找到 DeepLSD 权重文件: {self.ckpt_path}")

        print(f"[*] 正在加载 DeepLSD 几何模型: {self.ckpt_path.name}...")
        conf = {
            "detect_lines": self.detect_lines,
            "line_neighborhood": 5,
            "multiscale": False,
            "line_detection_params": {
                "merge": True,
                "grad_nfa": True,
                "filtering": "normal",
                "grad_thresh": 3,
            },
        }
        self.net = DeepLSD(conf)
        checkpoint = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
        self.net.load_state_dict(checkpoint["model"])
        self.net.to(self.device).eval()
        print("[✓] DeepLSD 模型加载就绪！")

    @torch.no_grad()
    def detect(self, image_gray_or_rgb: np.ndarray, max_lines: int | None = 300) -> dict[str, Any]:
        """前向提取线段、距离场与角度场。

        参数:
            image_gray_or_rgb: (H, W) 或 (H, W, 3) uint8 图像
            max_lines: 保留的最大线段数 (按置信度/长度截断，默认 300)

        返回:
            dict: {
                "lines": np.ndarray (M, 2, 2) [x, y],
                "line_scores": np.ndarray (M,),
                "df": np.ndarray (H, W) float32 [0, 5.0],
                "af": np.ndarray (H, W) float32 [0, pi],
            }
        """
        if image_gray_or_rgb.ndim == 3:
            gray = cv2.cvtColor(image_gray_or_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_gray_or_rgb

        h, w = gray.shape[:2]
        tensor_img = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0

        out = self.net({"image": tensor_img})

        df_np = out["df"].squeeze(0).cpu().numpy().astype(np.float32)
        af_np = out["line_level"].squeeze(0).cpu().numpy().astype(np.float32)

        if self.detect_lines and "lines" in out and len(out["lines"]) > 0:
            lines = out["lines"][0]  # (M, 2, 2)
            if isinstance(lines, torch.Tensor):
                lines = lines.cpu().numpy()
            lines = np.asarray(lines, dtype=np.float32)

            # 计算每条线段的长度与几何评分
            diff = lines[:, 1] - lines[:, 0]
            lengths = np.linalg.norm(diff, axis=1)

            # 过滤超短噪点线段
            valid = lengths >= self.min_line_len
            lines = lines[valid]
            lengths = lengths[valid]

            # 沿线查询距离场与梯度平均强度作为评分 (值越小距真线越近)
            scores = np.sqrt(lengths)

            # 如果超出上限，截取评分最高的线段
            if max_lines and len(lines) > max_lines:
                idx = np.argsort(-scores)[:max_lines]
                lines = lines[idx]
                scores = scores[idx]
        else:
            lines = np.zeros((0, 2, 2), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)

        return {
            "lines": lines,
            "line_scores": scores.astype(np.float32),
            "df": df_np,
            "af": af_np,
        }


def main():
    """单元测试辅助函数。"""
    print("[*] 正在测试 DeepLSDExtractor...")
    img = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(img, (50, 50), (400, 50), 255, 3)
    cv2.line(img, (400, 50), (400, 300), 255, 3)

    try:
        extractor = DeepLSDExtractor(checkpoint_path="/home/lbw/Project/new-LSD-matching/third_party/DeepLSD/weights/deeplsd_wireframe.tar")
        res = extractor.detect(img)
        print(f"[✓] 检测到线段数量: {len(res['lines'])}")
        print(f"[✓] 距离场尺寸: {res['df'].shape}")
        print(f"[✓] 角度场尺寸: {res['af'].shape}")
        print("[✓] DeepLSDExtractor 单元测试成功！")
    except Exception as e:
        print(f"[!] 测试失败: {e}")


if __name__ == "__main__":
    main()
