"""PyTorch 语义增强 GlueStick 数据集管道 (Semantic GlueStick PyTorch Dataset & DataLoader).

无缝对接 GlueStick 与后续语义图神经网络 (GNN) 训练:
1. 读取预生成的 .npz 多模态样本文件;
2. 转换为标准 PyTorch 张量，自动组装关键点、几何线段、拓扑连接图、SAM 256 维语义线特征与真值标签;
3. 提供智能动态 Padding 的 collate_fn，支持变长点、线批次训练与推理。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SemanticGlueStickDataset(Dataset):
    """语义增强 GlueStick 样本加载器 (支持 pure npz 与 image+npz 混合架构, 支持 train/val 划分)."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "all",
        val_ratio: float = 0.15,
        val_scenes: Sequence[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.npz_dir = self.data_dir / "npz" if (self.data_dir / "npz").is_dir() else self.data_dir
        self.images_dir = self.data_dir / "images"
        self.split = split

        index_file = self.data_dir / "dataset_index.json"
        all_metadata = []
        if index_file.is_file():
            with open(index_file, "r", encoding="utf-8") as f:
                all_metadata = json.load(f)
            raw_sample_names = [m["sample_name"] for m in all_metadata]
            self.meta_map = {m["sample_name"]: m for m in all_metadata}
        else:
            npz_files = sorted(list(self.npz_dir.glob("*.npz")))
            raw_sample_names = [p.stem for p in npz_files]
            self.meta_map = {}

        if len(raw_sample_names) == 0:
            raise FileNotFoundError(f"目录 {self.data_dir} 中未找到任何 .npz 样本文件！")

        # 场景划分 (按 Scene 隔离，杜绝数据泄露)
        scene_to_samples: dict[str, list[str]] = {}
        for s in raw_sample_names:
            scene = self.meta_map.get(s, {}).get("scene", s.split("_pair")[0] if "_pair" in s else s)
            if scene not in scene_to_samples:
                scene_to_samples[scene] = []
            scene_to_samples[scene].append(s)

        all_scenes = sorted(list(scene_to_samples.keys()))
        rng = np.random.RandomState(seed)
        shuffled_scenes = list(all_scenes)
        rng.shuffle(shuffled_scenes)

        if val_scenes is not None:
            chosen_val_scenes = set(val_scenes)
        else:
            num_val = max(1, int(round(len(all_scenes) * val_ratio)))
            chosen_val_scenes = set(shuffled_scenes[:num_val])

        if split == "train":
            active_scenes = [sc for sc in all_scenes if sc not in chosen_val_scenes]
        elif split == "val":
            active_scenes = [sc for sc in all_scenes if sc in chosen_val_scenes]
        else:
            active_scenes = all_scenes

        self.sample_names = []
        for sc in active_scenes:
            self.sample_names.extend(scene_to_samples[sc])
        self.metadata = [self.meta_map[s] for s in self.sample_names if s in self.meta_map]

    def __len__(self) -> int:
        return len(self.sample_names)

    def get_numpy_data(self, idx: int) -> dict[str, Any]:
        """获取单个样本的原始 NumPy 字典 (自动解耦处理 image+npz 与 pure npz 格式)."""
        name = self.sample_names[idx]
        file_path = self.npz_dir / f"{name}.npz"
        if not file_path.is_file():
            file_path = self.data_dir / f"{name}.npz"
        if not file_path.is_file():
            raise FileNotFoundError(f"未找到样本特征文件: {file_path}")

        raw_data = np.load(str(file_path), allow_pickle=True)
        data = {k: raw_data[k] for k in raw_data.files}

        # 检查图像是否存在于 npz 中，若不存在则从独立图像文件读取
        if "image0" not in data or "image1" not in data:
            meta = self.meta_map.get(name, {})
            # 候选路径寻找
            img0_cand = [
                self.data_dir / str(data.get("img0_path", "")),
                self.data_dir / meta.get("img0_file", ""),
                self.images_dir / f"{name}_0.png",
                self.images_dir / f"{name}_0.jpg",
            ]
            img1_cand = [
                self.data_dir / str(data.get("img1_path", "")),
                self.data_dir / meta.get("img1_file", ""),
                self.images_dir / f"{name}_1.png",
                self.images_dir / f"{name}_1.jpg",
            ]

            p0 = next((p for p in img0_cand if p.is_file()), None)
            p1 = next((p for p in img1_cand if p.is_file()), None)

            if p0 is None or p1 is None:
                raise FileNotFoundError(f"样本 {name} 缺失物理图像文件 (查找于: {self.images_dir})")

            img0_gray = cv2.imread(str(p0), cv2.IMREAD_GRAYSCALE)
            img1_gray = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
            data["image0"] = img0_gray
            data["image1"] = img1_gray

        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        name = self.sample_names[idx]
        data = self.get_numpy_data(idx)

        # 图像处理: (H, W) -> (1, H, W) float [0, 1]
        img0 = torch.from_numpy(data["image0"]).float().unsqueeze(0) / 255.0
        img1 = torch.from_numpy(data["image1"]).float().unsqueeze(0) / 255.0

        # 几何关键点与视觉描述子
        kpts0 = torch.from_numpy(data["keypoints0"]).float()
        kpts1 = torch.from_numpy(data["keypoints1"]).float()
        descs0 = torch.from_numpy(data["descriptors0"]).float()  # (256, N0)
        descs1 = torch.from_numpy(data["descriptors1"]).float()  # (256, N1)
        kpt_scores0 = torch.from_numpy(data["keypoint_scores0"]).float() if "keypoint_scores0" in data else torch.ones(len(kpts0), dtype=torch.float32)
        kpt_scores1 = torch.from_numpy(data["keypoint_scores1"]).float() if "keypoint_scores1" in data else torch.ones(len(kpts1), dtype=torch.float32)

        # 几何线段与拓扑
        lines0 = torch.from_numpy(data["lines0"]).float()  # (M0, 2, 2)
        lines1 = torch.from_numpy(data["lines1"]).float()  # (M1, 2, 2)
        line_scores0 = torch.from_numpy(data["line_scores0"]).float() if "line_scores0" in data else torch.ones(len(lines0), dtype=torch.float32)
        line_scores1 = torch.from_numpy(data["line_scores1"]).float() if "line_scores1" in data else torch.ones(len(lines1), dtype=torch.float32)
        lines_junc_idx0 = torch.from_numpy(data["lines_junc_idx0"]).long()
        lines_junc_idx1 = torch.from_numpy(data["lines_junc_idx1"]).long()
        conn0 = torch.from_numpy(data["connectivity0"]).bool()
        conn1 = torch.from_numpy(data["connectivity1"]).bool()

        # SAM 语义增强描述子与 Mask 属性
        sem_descs0 = torch.from_numpy(data["line_semantic_descs0"]).float()  # (256, M0)
        sem_descs1 = torch.from_numpy(data["line_semantic_descs1"]).float()  # (256, M1)
        mask_attrs0 = torch.from_numpy(data["line_mask_attrs0"]).float()     # (M0, 3)
        mask_attrs1 = torch.from_numpy(data["line_mask_attrs1"]).float()     # (M1, 3)

        # 真实标签
        gt_m0 = torch.from_numpy(data["gt_matches0"]).long()
        gt_lm0 = torch.from_numpy(data["gt_line_matches0"]).long()

        return {
            "name": name,
            "image0": img0,
            "image1": img1,
            # 点
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "keypoint_scores0": kpt_scores0,
            "keypoint_scores1": kpt_scores1,
            "descriptors0": descs0,
            "descriptors1": descs1,
            # 线
            "lines0": lines0,
            "lines1": lines1,
            "line_scores0": line_scores0,
            "line_scores1": line_scores1,
            "lines_junc_idx0": lines_junc_idx0,
            "lines_junc_idx1": lines_junc_idx1,
            "connectivity0": conn0,
            "connectivity1": conn1,
            # 语义
            "line_semantic_descs0": sem_descs0,
            "line_semantic_descs1": sem_descs1,
            "line_mask_attrs0": mask_attrs0,
            "line_mask_attrs1": mask_attrs1,
            # 监督
            "gt_matches0": gt_m0,
            "gt_line_matches0": gt_lm0,
            "K0": torch.from_numpy(data["K0"]).float(),
            "K1": torch.from_numpy(data["K1"]).float(),
            "T_0to1": torch.from_numpy(data["T_0to1"]).float(),
        }


def semantic_gluestick_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """智能批次对齐装配函数 (支持变长线段与交点)."""
    # 针对 batch_size == 1 快速直通 (GlueStick 典型训练/评测方式)
    if len(batch) == 1:
        elem = batch[0]
        out = {}
        for k, v in elem.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0)  # 扩展 Batch 维度 (1, ...)
            else:
                out[k] = [v]
        return out

    # 针对 batch_size > 1，对齐最大长度并生成有效掩码
    b_size = len(batch)
    max_kpts0 = max(b["keypoints0"].shape[0] for b in batch)
    max_kpts1 = max(b["keypoints1"].shape[0] for b in batch)
    max_lines0 = max(b["lines0"].shape[0] for b in batch)
    max_lines1 = max(b["lines1"].shape[0] for b in batch)

    out = {
        "name": [b["name"] for b in batch],
        "image0": torch.stack([b["image0"] for b in batch]),
        "image1": torch.stack([b["image1"] for b in batch]),
        "K0": torch.stack([b["K0"] for b in batch]),
        "K1": torch.stack([b["K1"] for b in batch]),
        "T_0to1": torch.stack([b["T_0to1"] for b in batch]),
    }

    # 填充函数
    def pad_tensor_2d(tensors, max_len, pad_value=0):
        # (N, D)
        d = tensors[0].shape[1] if tensors[0].ndim > 1 else 1
        res = torch.full((b_size, max_len, d), pad_value, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            if len(t) > 0:
                res[i, :len(t)] = t.view(len(t), -1)
        return res if tensors[0].ndim > 1 else res.squeeze(-1)

    out["keypoints0"] = pad_tensor_2d([b["keypoints0"] for b in batch], max_kpts0)
    out["keypoints1"] = pad_tensor_2d([b["keypoints1"] for b in batch], max_kpts1)
    out["keypoint_scores0"] = pad_tensor_2d([b["keypoint_scores0"] for b in batch], max_kpts0, pad_value=0.0)
    out["keypoint_scores1"] = pad_tensor_2d([b["keypoint_scores1"] for b in batch], max_kpts1, pad_value=0.0)

    # 描述子 (256, N) -> (B, 256, max_N)
    desc_dim = batch[0]["descriptors0"].shape[0]
    padded_desc0 = torch.zeros((b_size, desc_dim, max_kpts0), dtype=torch.float32)
    padded_desc1 = torch.zeros((b_size, desc_dim, max_kpts1), dtype=torch.float32)
    for i, b in enumerate(batch):
        padded_desc0[i, :, :b["descriptors0"].shape[1]] = b["descriptors0"]
        padded_desc1[i, :, :b["descriptors1"].shape[1]] = b["descriptors1"]
    out["descriptors0"] = padded_desc0
    out["descriptors1"] = padded_desc1

    # 线段 (M, 2, 2) -> (B, max_M, 2, 2)
    padded_lines0 = torch.zeros((b_size, max_lines0, 2, 2), dtype=torch.float32)
    padded_lines1 = torch.zeros((b_size, max_lines1, 2, 2), dtype=torch.float32)
    for i, b in enumerate(batch):
        padded_lines0[i, :len(b["lines0"])] = b["lines0"]
        padded_lines1[i, :len(b["lines1"])] = b["lines1"]
    out["lines0"] = padded_lines0
    out["lines1"] = padded_lines1

    out["line_scores0"] = pad_tensor_2d([b["line_scores0"] for b in batch], max_lines0, pad_value=0.0)
    out["line_scores1"] = pad_tensor_2d([b["line_scores1"] for b in batch], max_lines1, pad_value=0.0)
    out["lines_junc_idx0"] = pad_tensor_2d([b["lines_junc_idx0"] for b in batch], max_lines0, pad_value=0)
    out["lines_junc_idx1"] = pad_tensor_2d([b["lines_junc_idx1"] for b in batch], max_lines1, pad_value=0)

    # 语义线段描述子 (256, M) -> (B, 256, max_M)
    padded_sem0 = torch.zeros((b_size, desc_dim, max_lines0), dtype=torch.float32)
    padded_sem1 = torch.zeros((b_size, desc_dim, max_lines1), dtype=torch.float32)
    for i, b in enumerate(batch):
        padded_sem0[i, :, :b["line_semantic_descs0"].shape[1]] = b["line_semantic_descs0"]
        padded_sem1[i, :, :b["line_semantic_descs1"].shape[1]] = b["line_semantic_descs1"]
    out["line_semantic_descs0"] = padded_sem0
    out["line_semantic_descs1"] = padded_sem1

    # 实例属性 (M, 3) -> (B, max_M, 3)
    padded_attrs0 = torch.zeros((b_size, max_lines0, 3), dtype=torch.float32)
    padded_attrs1 = torch.zeros((b_size, max_lines1, 3), dtype=torch.float32)
    for i, b in enumerate(batch):
        padded_attrs0[i, :len(b["line_mask_attrs0"])] = b["line_mask_attrs0"]
        padded_attrs1[i, :len(b["line_mask_attrs1"])] = b["line_mask_attrs1"]
    out["line_mask_attrs0"] = padded_attrs0
    out["line_mask_attrs1"] = padded_attrs1

    # 真值匹配
    out["gt_matches0"] = pad_tensor_2d([b["gt_matches0"] for b in batch], max_kpts0, pad_value=-1)
    out["gt_line_matches0"] = pad_tensor_2d([b["gt_line_matches0"] for b in batch], max_lines0, pad_value=-1)

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="测试 SemanticGlueStickDataset 加载")
    parser.add_argument("--data-dir", default="data/test_semantic_gluestick")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    ds = SemanticGlueStickDataset(args.data_dir)
    print(f"[✓] 成功初始化数据集，样本总数: {len(ds)}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=semantic_gluestick_collate_fn)
    for b in loader:
        print("[✓] 成功加载批次样本:")
        print(f"    image0: {b['image0'].shape}")
        print(f"    lines0: {b['lines0'].shape}")
        print(f"    line_semantic_descs0: {b['line_semantic_descs0'].shape}")
        print(f"    gt_matches0: {b['gt_matches0'].shape}")
        break


if __name__ == "__main__":
    main()
