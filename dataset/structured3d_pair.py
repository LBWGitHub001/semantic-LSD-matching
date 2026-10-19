"""
Structured3D PyTorch Datasets for Model Training.

Provides:
  - Structured3DPairTrainDataset: Two-view pair dataset for training feature / line matching networks
    (supporting real viewpoint pairs, synthetic homography warps, or hybrid mode).
  - Structured3DSingleTrainDataset: Single-image dataset for training semantic segmentation
    or monocular line/depth networks.
  - warp_points_depth: Fast vectorized utility to compute pixel-level ground-truth correspondences
    between two views using depth and relative camera pose.
"""

from __future__ import annotations

import glob
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Project root path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _find_dataset_root(custom_path: str | Path | None = None) -> Path:
    if custom_path is not None:
        p = Path(custom_path)
        if (p / "Structured3D").is_dir():
            return p / "Structured3D"
        return p

    candidates = [
        ROOT / "Structured3D_perspective_full_00" / "Structured3D",
        ROOT / "Structured3D_perspective_full_00",
        ROOT / "data" / "Structured3D",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("scene_*")):
            return c
    raise FileNotFoundError(
        "Could not auto-locate Structured3D dataset directory. "
        "Please provide dataset_dir explicitly."
    )


def _load_camera_params(
    pose_path: str | Path,
    target_w: int,
    target_h: int,
    orig_w: int = 1280,
    orig_h: int = 720,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parse camera_pose.txt and compute scaled K, c2w, and w2c tensors.

    Returns:
        K: (3, 3) float32 intrinsics scaled to (target_w, target_h)
        c2w: (4, 4) float32 camera-to-world matrix (translation in meters)
        w2c: (4, 4) float32 world-to-camera matrix
    """
    with open(pose_path, "r", encoding="utf-8") as f:
        vals = [float(x) for x in f.read().strip().split()]

    # Position in millimeters -> convert to meters for standard SLAM/training convention
    pos_m = np.array(vals[0:3], dtype=np.float32) / 1000.0
    look = np.array(vals[3:6], dtype=np.float32)
    up = np.array(vals[6:9], dtype=np.float32)
    xfov = vals[9]
    yfov = vals[10]

    look = look / np.maximum(np.linalg.norm(look), 1e-8)
    up = up / np.maximum(np.linalg.norm(up), 1e-8)
    right = np.cross(look, up)
    right = right / np.maximum(np.linalg.norm(right), 1e-8)

    # Intrinsics at target resolution
    fx = (target_w / 2.0) / math.tan(xfov)
    fy = (target_h / 2.0) / math.tan(yfov)
    cx = target_w / 2.0
    cy = target_h / 2.0

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # OpenCV convention: X right, Y down (-up), Z look
    x_axis = right
    y_axis = -up
    z_axis = look

    R_c2w = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos_m

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = pos_m

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c

    return torch.from_numpy(K), torch.from_numpy(c2w), torch.from_numpy(w2c)


def warp_points_depth(
    coords0: torch.Tensor,
    depth0: torch.Tensor,
    depth1: torch.Tensor,
    K0: torch.Tensor,
    K1: torch.Tensor,
    T_0to1: torch.Tensor,
    dist_thresh: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Re-project 2D pixel coordinates from view 0 to view 1 using depth and relative pose.

    Args:
        coords0: (N, 2) pixel coordinates [x, y] in view 0
        depth0: (H, W) depth map of view 0 in meters
        depth1: (H, W) depth map of view 1 in meters
        K0: (3, 3) intrinsics of view 0
        K1: (3, 3) intrinsics of view 1
        T_0to1: (4, 4) relative pose from view 0 to view 1
        dist_thresh: relative depth consistency threshold (e.g. 8% depth difference)

    Returns:
        coords1: (N, 2) projected coordinates in view 1
        valid_mask: (N,) boolean mask indicating mutual visibility without occlusion
    """
    H, W = depth0.shape
    device = coords0.device

    x0 = coords0[:, 0]
    y0 = coords0[:, 1]
    ix0 = torch.clamp(x0.round().long(), 0, W - 1)
    iy0 = torch.clamp(y0.round().long(), 0, H - 1)

    z0 = depth0[iy0, ix0]
    valid_z0 = z0 > 0.05

    # Back-project to 3D camera 0: P0 = z0 * K0_inv @ [x, y, 1]
    K0_inv = torch.inverse(K0.to(device))
    homo0 = torch.stack([x0, y0, torch.ones_like(x0)], dim=1)  # (N, 3)
    rays0 = (K0_inv @ homo0.T).T  # (N, 3)
    pts3d_0 = rays0 * z0.unsqueeze(1)  # (N, 3)

    # Transform to camera 1: P1 = R @ P0 + t
    R = T_0to1[:3, :3].to(device)
    t = T_0to1[:3, 3].to(device)
    pts3d_1 = (R @ pts3d_0.T).T + t.unsqueeze(0)  # (N, 3)
    z1_proj = pts3d_1[:, 2]
    valid_z1 = z1_proj > 0.05

    # Project to image plane 1
    K1_dev = K1.to(device)
    homo1 = (K1_dev @ pts3d_1.T).T
    u1 = homo1[:, 0] / torch.clamp(z1_proj, min=1e-4)
    v1 = homo1[:, 1] / torch.clamp(z1_proj, min=1e-4)
    coords1 = torch.stack([u1, v1], dim=1)

    in_bounds = (u1 >= 0) & (u1 <= W - 1) & (v1 >= 0) & (v1 <= H - 1)

    # Occlusion check against depth1
    iu1 = torch.clamp(u1.round().long(), 0, W - 1)
    iv1 = torch.clamp(v1.round().long(), 0, H - 1)
    z1_actual = depth1[iv1, iu1]

    # Mutual depth consistency check
    rel_diff = torch.abs(z1_proj - z1_actual) / torch.clamp(z1_actual, min=0.1)
    depth_consistent = (z1_actual > 0.05) & (rel_diff < dist_thresh)

    valid_mask = valid_z0 & valid_z1 & in_bounds & depth_consistent
    return coords1, valid_mask


class Structured3DPairTrainDataset(Dataset):
    """
    PyTorch Two-View Dataset for Feature / Line Matching Training.

    Supports:
      - Real viewpoint pairs from the same room (filtered by rotation and baseline limits)
      - Synthetic homography transformations with photometric jitter
      - Hybrid mode (60% real pairs + 40% synthetic warps, proven optimal in XFeat)
      - Dense ground-truth correspondence coordinate generation
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        image_size: tuple[int, int] = (360, 640),
        pair_mode: str = "hybrid",
        hybrid_ratio: float = 0.6,
        max_rotation_deg: float = 85.0,
        min_baseline_m: float = 0.2,
        max_baseline_m: float = 4.0,
        load_normals: bool = False,
        photometric_augment: bool = True,
        num_dense_matches: int = 2048,
    ) -> None:
        super().__init__()
        self.root_dir = _find_dataset_root(dataset_dir)
        self.image_size = image_size  # (H, W)
        self.pair_mode = pair_mode.lower()
        self.hybrid_ratio = hybrid_ratio
        self.max_rotation_deg = max_rotation_deg
        self.min_baseline_m = min_baseline_m
        self.max_baseline_m = max_baseline_m
        self.load_normals = load_normals
        self.photometric_augment = photometric_augment
        self.num_dense_matches = num_dense_matches

        self.views: list[Path] = []
        self.real_pairs: list[tuple[Path, Path, float, float]] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Discover scenes, viewpoints, and build candidate matching pairs."""
        pattern = self.root_dir / "scene_*" / "2D_rendering" / "*" / "perspective" / "full" / "*"
        all_views = [Path(p) for p in sorted(glob.glob(str(pattern))) if (Path(p) / "camera_pose.txt").is_file()]
        self.views = all_views

        # Group by room
        room_groups: dict[tuple[str, str], list[Path]] = {}
        for vp in all_views:
            parts = vp.parts
            scene_id = parts[-6] if parts[-5] == "2D_rendering" else parts[-5]
            room_id = parts[-4]
            room_groups.setdefault((scene_id, room_id), []).append(vp)

        # Build valid two-view pairs within same room
        target_h, target_w = self.image_size
        for group in room_groups.values():
            n = len(group)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = group[i], group[j]
                    _, _, w2c1 = _load_camera_params(p1 / "camera_pose.txt", target_w, target_h)
                    _, _, w2c2 = _load_camera_params(p2 / "camera_pose.txt", target_w, target_h)

                    R1 = w2c1[:3, :3].numpy()
                    t1 = -R1.T @ w2c1[:3, 3].numpy()
                    R2 = w2c2[:3, :3].numpy()
                    t2 = -R2.T @ w2c2[:3, 3].numpy()

                    # Baseline
                    base_m = float(np.linalg.norm(t1 - t2))
                    if not (self.min_baseline_m <= base_m <= self.max_baseline_m):
                        continue

                    # Rotation angle
                    R_rel = R2 @ R1.T
                    tr = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
                    rot_deg = float(np.degrees(np.arccos(tr)))
                    if rot_deg > self.max_rotation_deg:
                        continue

                    self.real_pairs.append((p1, p2, base_m, rot_deg))

    def __len__(self) -> int:
        if self.pair_mode == "real":
            return len(self.real_pairs)
        # In hybrid or synthetic mode, total pairs equal views length or real_pairs
        return max(len(self.real_pairs), len(self.views))

    def _apply_photometric(self, img: Image.Image) -> Image.Image:
        """Apply random brightness, contrast, and saturation jitter."""
        if not self.photometric_augment:
            return img
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            img = ImageEnhance.Color(img).enhance(factor)
        return img

    def _load_view_tensors(
        self, view_dir: Path
    ) -> dict[str, torch.Tensor]:
        target_h, target_w = self.image_size

        # 1. RGB
        rgb_img = Image.open(view_dir / "rgb_rawlight.png").convert("RGB")
        rgb_resized = rgb_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        rgb_aug = self._apply_photometric(rgb_resized)
        rgb_t = torch.from_numpy(np.array(rgb_aug, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # 2. Depth (uint16 in mm -> float32 in meters)
        depth_pil = Image.open(view_dir / "depth.png")
        depth_arr = np.array(depth_pil.resize((target_w, target_h), Image.Resampling.NEAREST), dtype=np.float32)
        depth_t = torch.from_numpy(depth_arr / 1000.0)

        # 3. Semantic (uint8 class IDs)
        sem_pil = Image.open(view_dir / "semantic.png")
        sem_arr = np.array(sem_pil.resize((target_w, target_h), Image.Resampling.NEAREST), dtype=np.int64)
        sem_t = torch.from_numpy(sem_arr)

        # 4. Camera Intrinsics & Extrinsics
        K, c2w, w2c = _load_camera_params(view_dir / "camera_pose.txt", target_w, target_h)

        res: dict[str, torch.Tensor] = {
            "image": rgb_t,
            "depth": depth_t,
            "semantic": sem_t,
            "K": K,
            "c2w": c2w,
            "w2c": w2c,
        }

        # 5. Normal map (optional)
        if self.load_normals:
            norm_path = view_dir / "normal.png"
            norm_img = Image.open(norm_path).convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
            norm_arr = np.array(norm_img, dtype=np.float32) / 127.5 - 1.0
            norm_len = np.maximum(np.linalg.norm(norm_arr, axis=-1, keepdims=True), 1e-6)
            norm_arr = norm_arr / norm_len
            res["normal"] = torch.from_numpy(norm_arr.transpose(2, 0, 1))

        return res

    def _sample_synthetic_pair(self, idx: int) -> dict[str, Any]:
        """Generate a synthetic homography warped pair from a single view."""
        view_dir = self.views[idx % len(self.views)]
        data0 = self._load_view_tensors(view_dir)
        H_size, W_size = self.image_size

        # Create random homography (rotation, scale, perspective)
        pts1 = np.float32([[0, 0], [W_size, 0], [W_size, H_size], [0, H_size]])
        # Random offsets up to 18% of dimension
        max_dx, max_dy = W_size * 0.18, H_size * 0.18
        pts2 = pts1 + np.random.uniform(-max_dx, max_dx, pts1.shape).astype(np.float32)
        # Perspective transform matrix
        M = cv2.getPerspectiveTransform(pts1, pts2) if "cv2" in sys.modules else None

        if M is None:
            # Fallback affine via numpy if cv2 is absent
            theta = random.uniform(-0.25, 0.25)
            scale = random.uniform(0.85, 1.15)
            cos_t, sin_t = math.cos(theta) * scale, math.sin(theta) * scale
            M = np.array([
                [cos_t, -sin_t, random.uniform(-max_dx, max_dx)],
                [sin_t, cos_t, random.uniform(-max_dy, max_dy)],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)

        # Warp image0 to image1 using grid_sample
        M_tensor = torch.from_numpy(M).float()
        M_inv = torch.inverse(M_tensor)

        # Create normalized grid for torch grid_sample
        y, x = torch.meshgrid(torch.arange(H_size), torch.arange(W_size), indexing="ij")
        homo = torch.stack([x.flatten().float(), y.flatten().float(), torch.ones(H_size * W_size)], dim=0)
        src_coords = M_inv @ homo
        src_x = src_coords[0] / torch.clamp(src_coords[2], min=1e-6)
        src_y = src_coords[1] / torch.clamp(src_coords[2], min=1e-6)

        # Normalize to [-1, 1]
        grid_x = (src_x / (W_size - 1.0)) * 2.0 - 1.0
        grid_y = (src_y / (H_size - 1.0)) * 2.0 - 1.0
        grid = torch.stack([grid_x.view(H_size, W_size), grid_y.view(H_size, W_size)], dim=-1).unsqueeze(0)

        # Warp RGB
        img0_batch = data0["image"].unsqueeze(0)
        img1 = F.grid_sample(img0_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        if self.photometric_augment:
            img1 = torch.clamp(img1 * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0.0, 1.0)

        # Warp Depth
        depth0_batch = data0["depth"].unsqueeze(0).unsqueeze(0)
        depth1 = F.grid_sample(depth0_batch, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0]

        # Warp Semantic
        sem0_batch = data0["semantic"].float().unsqueeze(0).unsqueeze(0)
        sem1 = F.grid_sample(sem0_batch, grid, mode="nearest", padding_mode="zeros", align_corners=True)[0, 0].long()

        # Build relative identity pose with homography
        T_rel = torch.eye(4, dtype=torch.float32)

        out = {
            "image0": data0["image"],
            "image1": img1,
            "depth0": data0["depth"],
            "depth1": depth1,
            "semantic0": data0["semantic"],
            "semantic1": sem1,
            "K0": data0["K"],
            "K1": data0["K"],
            "T_0to1": T_rel,
            "T_1to0": T_rel,
            "H_matrix": M_tensor,
            "pair_type": "synthetic",
        }

        if self.num_dense_matches > 0:
            out["matches"] = self._sample_dense_homography_matches(M_tensor, H_size, W_size)

        return out

    def _sample_dense_homography_matches(
        self, H_mat: torch.Tensor, H_size: int, W_size: int
    ) -> torch.Tensor:
        """Sample valid pixel correspondences from homography H."""
        n = self.num_dense_matches
        x0 = torch.rand(n * 2) * (W_size - 1)
        y0 = torch.rand(n * 2) * (H_size - 1)
        homo0 = torch.stack([x0, y0, torch.ones_like(x0)], dim=0)
        homo1 = H_mat @ homo0
        x1 = homo1[0] / torch.clamp(homo1[2], min=1e-6)
        y1 = homo1[1] / torch.clamp(homo1[2], min=1e-6)

        valid = (x1 >= 0) & (x1 <= W_size - 1) & (y1 >= 0) & (y1 <= H_size - 1) & (homo1[2] > 0)
        x0_v, y0_v = x0[valid][:n], y0[valid][:n]
        x1_v, y1_v = x1[valid][:n], y1[valid][:n]

        matches = torch.stack([x0_v, y0_v, x1_v, y1_v], dim=1)
        return matches

    def _sample_real_pair(self, idx: int) -> dict[str, Any]:
        """Load a real pair of viewpoints from the same room with ground truth pose."""
        p1, p2, base_m, rot_deg = self.real_pairs[idx % len(self.real_pairs)]

        # Randomize order (A->B or B->A)
        if random.random() < 0.5:
            p1, p2 = p2, p1

        data0 = self._load_view_tensors(p1)
        data1 = self._load_view_tensors(p2)

        # Relative pose from camera 0 to camera 1: T_0to1 = w2c1 @ c2w0
        T_0to1 = data1["w2c"] @ data0["c2w"]
        T_1to0 = torch.inverse(T_0to1)

        out = {
            "image0": data0["image"],
            "image1": data1["image"],
            "depth0": data0["depth"],
            "depth1": data1["depth"],
            "semantic0": data0["semantic"],
            "semantic1": data1["semantic"],
            "K0": data0["K"],
            "K1": data1["K"],
            "T_0to1": T_0to1,
            "T_1to0": T_1to0,
            "baseline_m": torch.tensor(base_m, dtype=torch.float32),
            "rotation_deg": torch.tensor(rot_deg, dtype=torch.float32),
            "pair_type": "real",
        }

        if self.load_normals and "normal" in data0 and "normal" in data1:
            out["normal0"] = data0["normal"]
            out["normal1"] = data1["normal"]

        if self.num_dense_matches > 0:
            out["matches"] = self._sample_dense_real_matches(
                data0["depth"], data1["depth"], data0["K"], data1["K"], T_0to1
            )

        return out

    def _sample_dense_real_matches(
        self,
        depth0: torch.Tensor,
        depth1: torch.Tensor,
        K0: torch.Tensor,
        K1: torch.Tensor,
        T_0to1: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ground-truth pixel matches between two real camera frames."""
        H_size, W_size = self.image_size
        n = self.num_dense_matches
        # Sample candidate points in view 0 where depth > 0
        valid_y, valid_x = torch.where(depth0 > 0.1)
        if len(valid_y) < n:
            return torch.zeros((0, 4), dtype=torch.float32)

        perm = torch.randperm(len(valid_y))[: n * 3]
        cand_x = valid_x[perm].float()
        cand_y = valid_y[perm].float()
        coords0 = torch.stack([cand_x, cand_y], dim=1)

        coords1, valid_mask = warp_points_depth(
            coords0, depth0, depth1, K0, K1, T_0to1, dist_thresh=0.08
        )
        c0_sel = coords0[valid_mask][:n]
        c1_sel = coords1[valid_mask][:n]
        return torch.cat([c0_sel, c1_sel], dim=1)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.pair_mode == "real":
            return self._sample_real_pair(idx)
        elif self.pair_mode == "synthetic":
            return self._sample_synthetic_pair(idx)
        else:  # hybrid
            if random.random() < self.hybrid_ratio and len(self.real_pairs) > 0:
                return self._sample_real_pair(idx)
            return self._sample_synthetic_pair(idx)


class Structured3DSingleTrainDataset(Dataset):
    """
    PyTorch Single-View Dataset for Semantic Segmentation or Depth/Line Training.
    """

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        image_size: tuple[int, int] = (360, 640),
        load_normals: bool = True,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = _find_dataset_root(dataset_dir)
        self.image_size = image_size
        self.load_normals = load_normals
        self.augment = augment

        pattern = self.root_dir / "scene_*" / "2D_rendering" / "*" / "perspective" / "full" / "*"
        self.views = [Path(p) for p in sorted(glob.glob(str(pattern))) if (Path(p) / "camera_pose.txt").is_file()]

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vdir = self.views[idx]
        target_h, target_w = self.image_size

        # Image
        img = Image.open(vdir / "rgb_rawlight.png").convert("RGB")
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # Random horizontal flip
        flip = self.augment and random.random() < 0.5
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        img_t = torch.from_numpy(np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        # Depth
        depth_pil = Image.open(vdir / "depth.png").resize((target_w, target_h), Image.Resampling.NEAREST)
        if flip:
            depth_pil = depth_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        depth_t = torch.from_numpy(np.array(depth_pil, dtype=np.float32) / 1000.0)

        # Semantic
        sem_pil = Image.open(vdir / "semantic.png").resize((target_w, target_h), Image.Resampling.NEAREST)
        if flip:
            sem_pil = sem_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        sem_t = torch.from_numpy(np.array(sem_pil, dtype=np.int64))

        K, c2w, w2c = _load_camera_params(vdir / "camera_pose.txt", target_w, target_h)

        sample: dict[str, torch.Tensor] = {
            "image": img_t,
            "depth": depth_t,
            "semantic": sem_t,
            "K": K,
            "c2w": c2w,
            "w2c": w2c,
        }

        if self.load_normals:
            norm_pil = Image.open(vdir / "normal.png").convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
            if flip:
                norm_pil = norm_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            norm_arr = np.array(norm_pil, dtype=np.float32) / 127.5 - 1.0
            norm_len = np.maximum(np.linalg.norm(norm_arr, axis=-1, keepdims=True), 1e-6)
            sample["normal"] = torch.from_numpy((norm_arr / norm_len).transpose(2, 0, 1))

        return sample
