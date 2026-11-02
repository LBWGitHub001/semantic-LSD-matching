"""线框拓扑结构构建与 SuperPoint 特征引擎 (Wireframe & SuperPoint Builder).

实现 GlueStick 标准的端点合流、Junction 聚类与图连接性拓扑矩阵生成:
1. 加载 SuperPoint 模型，提取图像密集/稀疏关键点与其 256 维视觉描述子;
2. 将 DeepLSD 线段端点与 SuperPoint 关键点通过 DBSCAN 进行空间合流聚类，生成唯一 Junctions;
3. 构建拓扑邻接连接矩阵 (Connectivity, [J, J]);
4. 建立线段到 Junctions 的端点索引映射 (lines_junc_idx, [M, 2])。
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
gluestick_cands = [ROOT / "third_party" / "GlueStick", ROOT / "GlueStick"]
GLUESTICK_DIR = next((p for p in gluestick_cands if p.is_dir()), gluestick_cands[0])
if str(GLUESTICK_DIR) not in sys.path:
    sys.path.insert(0, str(GLUESTICK_DIR))

try:
    from gluestick.models.superpoint import SuperPoint, sample_descriptors
except ImportError as e:
    SuperPoint = None
    sample_descriptors = None
    _sp_err = e


class WireframeBuilder:
    """线框图拓扑构建器。"""

    def __init__(
        self,
        sp_weights_path: str | Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        nms_radius: float = 3.0,
        max_num_keypoints: int = 1000,
    ) -> None:
        self.device = torch.device(device)
        self.nms_radius = nms_radius

        if sp_weights_path and Path(sp_weights_path).is_file():
            self.sp_path = Path(sp_weights_path).resolve()
        else:
            cand_paths = [
                ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "superpoint_v1.pth",
                ROOT / "GlueStick" / "resources" / "weights" / "superpoint_v1.pth",
                ROOT / "weights" / "superpoint" / "superpoint_v1.pth",
            ]
            self.sp_path = next((p.resolve() for p in cand_paths if p.is_file()), cand_paths[0].resolve())

        if SuperPoint is None:
            raise ImportError(f"无法导入 SuperPoint 模型: {_sp_err}")

        sp_conf = {
            "has_detector": True,
            "has_descriptor": True,
            "descriptor_dim": 256,
            "trainable": False,
            "return_all": True,
            "sparse_outputs": True,
            "nms_radius": 4,
            "detection_threshold": 0.005,
            "max_num_keypoints": max_num_keypoints,
            "force_num_keypoints": False,
            "remove_borders": 4,
        }
        print(f"[*] 正在初始化 SuperPoint 特征抽取器...")
        self.sp = SuperPoint(sp_conf).to(self.device).eval()
        if self.sp_path.is_file():
            ckpt = torch.load(self.sp_path, map_location=self.device)
            # 处理可能的 state_dict 键嵌套
            state = ckpt.get("model", ckpt)
            self.sp.load_state_dict(state, strict=False)
            print(f"[✓] SuperPoint 权重加载成功: {self.sp_path.name}")
        else:
            print(f"[!] 警告: 未在 {self.sp_path} 找到权重，使用默认初始化")

    @torch.no_grad()
    def extract_superpoint(self, image_gray: np.ndarray) -> dict[str, torch.Tensor]:
        """提取 SuperPoint 关键点与描述子。

        参数:
            image_gray: (H, W) uint8 灰度图

        返回:
            keypoints: (N, 2)
            scores: (N,)
            descriptors: (256, N)
            dense_descriptors: (1, 256, H/8, W/8)
        """
        tensor_img = torch.from_numpy(image_gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        pred = self.sp({"image": tensor_img})
        return {
            "keypoints": pred["keypoints"][0],        # (N, 2)
            "scores": pred["keypoint_scores"][0],     # (N,)
            "descriptors": pred["descriptors"][0],   # (256, N)
            "dense_descriptors": pred["all_descriptors"],  # (1, 256, H/8, W/8)
        }

    def build_wireframe(
        self,
        lines: np.ndarray | torch.Tensor,
        line_scores: np.ndarray | torch.Tensor,
        dense_descs: torch.Tensor,
        extra_keypoints: torch.Tensor | None = None,
        extra_scores: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """将线段端点聚类为 Junctions，并构建邻接矩阵。

        返回:
            junctions: (J, 2) 浮点坐标
            junc_scores: (J,) 置信度
            junc_descs: (256, J) 描述子
            connectivity: (J, J) bool 邻接矩阵
            snapped_lines: (M, 2, 2) 贴合到交点的几何线段
            lines_junc_idx: (M, 2) 线段端点对应的 Junction 索引
        """
        if isinstance(lines, np.ndarray):
            lines = torch.from_numpy(lines).float().to(self.device)
        if isinstance(line_scores, np.ndarray):
            line_scores = torch.from_numpy(line_scores).float().to(self.device)

        m = len(lines)
        if m == 0:
            # 空线段兜底
            return {
                "junctions": torch.zeros((0, 2), dtype=torch.float32, device=self.device),
                "junc_scores": torch.zeros((0,), dtype=torch.float32, device=self.device),
                "junc_descs": torch.zeros((256, 0), dtype=torch.float32, device=self.device),
                "connectivity": torch.zeros((0, 0), dtype=torch.bool, device=self.device),
                "snapped_lines": torch.zeros((0, 2, 2), dtype=torch.float32, device=self.device),
                "lines_junc_idx": torch.zeros((0, 2), dtype=torch.long, device=self.device),
            }

        endpoints = lines.reshape(-1, 2)  # (2*M, 2)
        ep_scores = torch.repeat_interleave(line_scores, 2)

        # 如果提供了关键点，一并加入聚类池
        if extra_keypoints is not None and len(extra_keypoints) > 0:
            all_points = torch.cat([endpoints, extra_keypoints], dim=0)
            all_scores = torch.cat([ep_scores, extra_scores], dim=0)
        else:
            all_points = endpoints
            all_scores = ep_scores

        # 使用 DBSCAN 聚类合流邻近交点
        np_pts = all_points.cpu().numpy()
        db = DBSCAN(eps=self.nms_radius, min_samples=1).fit(np_pts)
        clusters = torch.tensor(db.labels_, dtype=torch.long, device=self.device)
        n_clusters = len(set(db.labels_))

        # 计算每个聚类簇的平均中心与最大评分
        junctions = torch.zeros((n_clusters, 2), dtype=torch.float32, device=self.device)
        junctions.scatter_reduce_(0, clusters[:, None].repeat(1, 2), all_points, reduce="mean", include_self=False)

        junc_scores = torch.zeros((n_clusters,), dtype=torch.float32, device=self.device)
        junc_scores.scatter_reduce_(0, clusters, all_scores, reduce="amax", include_self=False)

        # 重新映射线段端点与索引
        lines_clusters = clusters[: 2 * m].reshape(m, 2)
        snapped_lines = junctions[lines_clusters].reshape(m, 2, 2)

        # 构建 Junction 邻接连接矩阵
        connectivity = torch.eye(n_clusters, dtype=torch.bool, device=self.device)
        p0 = lines_clusters[:, 0]
        p1 = lines_clusters[:, 1]
        connectivity[p0, p1] = True
        connectivity[p1, p0] = True

        # 从 SuperPoint dense_descriptors 稠密图中双线性采样 Junctions 描述子
        junc_descs = sample_descriptors(junctions.unsqueeze(0), dense_descs, 8)[0]  # (256, J)

        return {
            "junctions": junctions,
            "junc_scores": junc_scores,
            "junc_descs": junc_descs,
            "connectivity": connectivity,
            "snapped_lines": snapped_lines,
            "lines_junc_idx": lines_clusters,
        }


def main():
    print("[*] 正在测试 WireframeBuilder...")
    img = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), 255, 2)
    builder = WireframeBuilder()
    sp_out = builder.extract_superpoint(img)
    lines = np.array([
        [[100, 100], [300, 100]],
        [[300, 100], [300, 300]],
        [[300, 300], [100, 300]],
        [[100, 300], [100, 100]],
    ], dtype=np.float32)
    scores = np.ones(4, dtype=np.float32)
    wf = builder.build_wireframe(lines, scores, sp_out["dense_descriptors"], sp_out["keypoints"], sp_out["scores"])
    print(f"[✓] 合流 Junctions 数量: {len(wf['junctions'])}")
    print(f"[✓] 拓扑连接矩阵尺寸: {wf['connectivity'].shape}")
    print(f"[✓] 描述子维度: {wf['junc_descs'].shape}")
    print("[✓] WireframeBuilder 单元测试成功！")


if __name__ == "__main__":
    main()

