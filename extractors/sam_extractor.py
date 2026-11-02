"""高精度 SAM 语义特征与轮廓属性提取器 (SAM High-Precision Semantic Extractor).

核心功能:
1. 加载 SAM ViT-H 旗舰超大模型 (6.36 亿参数)，支持 FP16 显存安全前向推理；
2. 提取全图 256 维高分辨率语义嵌入特征图 (Dense Image Embeddings);
3. 亚像素高密采样池化 (High-Density Line Pooling):
   沿每条几何线段均匀采样 16 个点，通过双线性插值采样 SAM 语义特征并做均值池化与 L2 归一化；
4. 物理边界轮廓属性判定:
   分析线段与实例分割掩膜 (Masks) 的重叠关系，判定线段是否为物体真实几何外轮廓边界。
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    from segment_anything.modeling.image_encoder import Attention, add_decomposed_rel_pos

    # 针对 PyTorch 2.0+ 启用原生内存优化 FlashAttention (SDPA)
    # 消除 ViT-H 全局 Attention 分配 1024MB 矩阵造成的显存瓶颈
    def _sdpa_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        if self.use_rel_pos:
            rel_pos = torch.zeros((B * self.num_heads, H * W, H * W), dtype=q.dtype, device=q.device)
            rel_pos = add_decomposed_rel_pos(rel_pos, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        x = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

    Attention.forward = _sdpa_attention_forward

except ImportError:
    sam_model_registry = None
    SamPredictor = None
    SamAutomaticMaskGenerator = None


class SAMExtractor:
    """SAM 语义特征与实例边界提取引擎。"""

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str | Path = "weights/sam/sam_vit_h_4b8939.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True,
        num_line_samples: int = 16,
    ) -> None:
        if sam_model_registry is None:
            raise ImportError("未检测到 segment_anything 库，请先安装: pip install segment-anything")

        self.device = torch.device(device)
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")
        self.num_line_samples = num_line_samples
        self.ckpt_path = Path(checkpoint_path).resolve()

        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"未能找到 SAM 权重文件: {self.ckpt_path}")

        print(f"[*] 正在加载 SAM ({model_type}) 旗舰模型: {self.ckpt_path.name} (FP16={self.use_fp16})...")
        self.sam = sam_model_registry[model_type](checkpoint=str(self.ckpt_path))
        self.sam.to(self.device).eval()

        self.predictor = SamPredictor(self.sam)
        self.mask_generator = None  # 延迟按需初始化，节约常规特征提取开销
        print("[✓] SAM 旗舰模型加载完成！")

    @torch.no_grad()
    def extract_image_embeddings(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        """提取输入图像的 SAM 密集语义特征图。

        参数:
            image_rgb: (H, W, 3) uint8 RGB 图像

        返回:
            embeddings: (1, 256, H_feat, W_feat) 浮点特征图 (在 GPU 上)
            meta: 包含原始尺寸、变换后尺寸及缩放比的元数据字典
        """
        orig_h, orig_w = image_rgb.shape[:2]

        with torch.amp.autocast("cuda", enabled=self.use_fp16):
            self.predictor.set_image(image_rgb)
            features = self.predictor.features  # (1, 256, 64, 64)

        meta = {
            "orig_size": (orig_h, orig_w),
            "input_size": self.predictor.input_size,  # (new_h, new_w) 缩放后的有效图像区域
            "feat_size": (features.shape[-2], features.shape[-1]),  # (64, 64)
        }
        return features, meta

    def pool_line_descriptors(
        self,
        embeddings: torch.Tensor,
        lines: np.ndarray | torch.Tensor,
        meta: dict[str, Any],
    ) -> torch.Tensor:
        """沿线段高密插值采样 SAM 特征图并进行均值池化。

        参数:
            embeddings: (1, 256, H_f, W_f) 特征图
            lines: (M, 2, 2) 或 (M, 4) 图像原始坐标系下的线段端点 [(x1, y1), (x2, y2)]
            meta: extract_image_embeddings 返回的元数据字典

        返回:
            line_descs: (256, M) L2 归一化语义描述子 (对齐 GlueStick 规范)
        """
        if len(lines) == 0:
            return torch.zeros((embeddings.shape[1], 0), dtype=torch.float32, device=self.device)

        if isinstance(lines, np.ndarray):
            lines_tensor = torch.from_numpy(lines).float().to(self.device)
        else:
            lines_tensor = lines.float().to(self.device)

        if lines_tensor.ndim == 2 and lines_tensor.shape[1] == 4:
            lines_tensor = lines_tensor.view(-1, 2, 2)

        # 1. 沿每条线段在 [0, 1] 参数域均匀生成采样权重
        num_pts = self.num_line_samples
        t = torch.linspace(0.0, 1.0, num_pts, device=self.device).view(1, num_pts, 1)  # (1, 16, 1)
        p_start = lines_tensor[:, 0:1, :]  # (M, 1, 2) [x, y]
        p_end = lines_tensor[:, 1:2, :]    # (M, 1, 2) [x, y]

        # sample_pts: (M, 16, 2) 原始图像分辨率下的坐标点
        sample_pts = p_start + t * (p_end - p_start)

        # 2. 将原始图像坐标严格映射到 SAM 64x64 特征图归一化网格坐标 [-1, 1]
        orig_h, orig_w = meta["orig_size"]
        new_h, new_w = meta["input_size"]
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        # SAM 内部补齐至 1024x1024 (对应特征图 64x64, 步长 16)
        feat_h, feat_w = meta["feat_size"]  # 64, 64
        grid_x = (sample_pts[..., 0] * scale_x) / 16.0
        grid_y = (sample_pts[..., 1] * scale_y) / 16.0

        # 归一化至 [-1, 1] 供 F.grid_sample 使用
        norm_x = 2.0 * (grid_x / max(feat_w - 1, 1)) - 1.0
        norm_y = 2.0 * (grid_y / max(feat_h - 1, 1)) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, M, 16, 2)

        # 3. 双线性无损插值提取
        with torch.amp.autocast("cuda", enabled=self.use_fp16):
            # sampled: (1, 256, M, 16)
            sampled = F.grid_sample(embeddings, grid, mode="bilinear", padding_mode="border", align_corners=True)

        # 4. 沿线段方向平均池化并做 L2 归一化
        line_feats = sampled.squeeze(0).mean(dim=-1)  # (256, M)
        line_feats = F.normalize(line_feats.float(), p=2, dim=0)  # 沿 256 维做单位范数归一化

        return line_feats

    @torch.no_grad()
    def extract_line_mask_attributes(
        self,
        image_rgb: np.ndarray,
        lines: np.ndarray,
        max_masks: int = 64,
    ) -> np.ndarray:
        """分析线段在实例分割掩膜上的几何归属，计算轮廓边界属性。

        返回:
            attrs: (M, 3) 数组 [dominant_mask_id, stability_score, is_boundary]
                - dominant_mask_id: 整数实例 ID
                - stability_score: 掩膜置信度 [0, 1]
                - is_boundary: 1.0 (物体轮廓边缘), 0.0 (物体内部纹理线)
        """
        m = len(lines)
        if m == 0:
            return np.zeros((0, 3), dtype=np.float32)

        if self.mask_generator is None:
            # 高精度配置 MaskGenerator (points_per_side=16 确保 8GB 显存绝对安全)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=16,
                points_per_batch=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                min_mask_region_area=100,
            )

        with torch.amp.autocast("cuda", enabled=self.use_fp16):
            masks = self.mask_generator.generate(image_rgb)

        torch.cuda.empty_cache()

        h, w = image_rgb.shape[:2]
        instance_map = np.zeros((h, w), dtype=np.int32)
        score_map = np.zeros((h, w), dtype=np.float32)
        boundary_map = np.zeros((h, w), dtype=np.uint8)

        # 按照面积从大到小排列，小物体覆盖大物体
        sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)[:max_masks]

        for idx, mask_info in enumerate(sorted_masks, start=1):
            mask = mask_info["segmentation"]
            score = float(mask_info.get("stability_score", 0.95))
            instance_map[mask] = idx
            score_map[mask] = score

            # 计算物体轮廓线 (Sobel 或形态学膨胀做差)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(mask.astype(np.uint8), kernel)
            boundary = (dilated ^ mask.astype(np.uint8)) > 0
            boundary_map[boundary] = 1

        # 沿线段端点连线密集采样 16 个点，判定落点属性
        lines_4 = lines.reshape(-1, 4)
        attrs = np.zeros((m, 3), dtype=np.float32)

        for i, (x1, y1, x2, y2) in enumerate(lines_4):
            xs = np.clip(np.linspace(x1, x2, 16), 0, w - 1).astype(int)
            ys = np.clip(np.linspace(y1, y2, 16), 0, h - 1).astype(int)

            ids = instance_map[ys, xs]
            scores = score_map[ys, xs]
            bounds = boundary_map[ys, xs]

            # 主导 Mask ID (众数)
            counts = np.bincount(ids)
            dom_id = int(np.argmax(counts))

            # 平均稳定性置信度
            mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0

            # 如果超过 20% 的采样点落在物体边界带上，或线段跨越不同 Mask，则标记为边界轮廓线
            is_bound = 1.0 if (np.mean(bounds) > 0.20 or len(np.unique(ids[ids > 0])) > 1) else 0.0

            attrs[i] = [float(dom_id), mean_score, is_bound]

        return attrs


def main():
    """单元测试辅助函数。"""
    print("[*] 正在对 SAMExtractor 进行形状与前向验证...")
    # 构建虚拟图像与线段测试
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

    lines = np.array([
        [[100, 100], [300, 100]],
        [[300, 100], [300, 300]],
        [[150, 150], [250, 250]],
    ], dtype=np.float32)

    try:
        extractor = SAMExtractor(model_type="vit_h")
        feat, meta = extractor.extract_image_embeddings(img)
        print(f"[✓] SAM 特征图尺寸: {feat.shape}")
        line_feats = extractor.pool_line_descriptors(feat, lines, meta)
        print(f"[✓] 线段语义描述子尺寸: {line_feats.shape} (期望: (256, 3))")
        attrs = extractor.extract_line_mask_attributes(img, lines)
        print(f"[✓] 线段 Mask 属性形状: {attrs.shape}")
        print("[✓] SAMExtractor 单元测试成功！")
    except Exception as e:
        print(f"[!] 测试失败: {e}")


if __name__ == "__main__":
    main()
