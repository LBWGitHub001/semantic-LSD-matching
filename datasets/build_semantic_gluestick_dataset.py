"""ScanNet-1500 语义增强 GlueStick 数据集主生成器 (Semantic GlueStick Dataset Builder).

融合流程:
1. 读取 ScanNet-1500 校准配对表 (pairs_calibrated.txt);
2. 图像自适应统一分辨率 (640x480) 并适配内参 K;
3. 几何分支: DeepLSD 提取亚像素线段与距离场，SuperPoint 提取特征点与视觉描述子;
4. 拓扑分支: 线框交点 DBSCAN 聚类合流，构建图邻接矩阵 (connectivity);
5. 语义分支 (SAM ViT-H 旗舰超大模型):
   - 提取 256 维高分辨率语义特征图;
   - 16 点沿线高密双线性采样与 L2 归一化平均池化 -> 256-d 线段语义描述子;
   - 关键点语义特征采样 -> 256-d 关键点语义描述子;
   - 实例 Mask 重合度与物理轮廓边界属性判定 [M, 3];
6. 监督真值: 依据真实位姿 T_0to1 与真实深度图严格反投影，生成几何端点匹配真值与线段匹配真值;
7. 紧凑序列化导出为 .npz 格式与索引表 dataset_index.json。
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractors.deeplsd_extractor import DeepLSDExtractor
from extractors.sam_extractor import SAMExtractor
from extractors.wireframe_builder import WireframeBuilder


def parse_pairs_line(line: str) -> dict[str, Any]:
    """解析 pairs_calibrated.txt 单行内容。"""
    tokens = line.strip().split()
    if len(tokens) < 36:
        raise ValueError(f"格式错误，token 数量不足 36: {len(tokens)}")

    img0_rel = tokens[0]
    img1_rel = tokens[1]
    scene = img0_rel.split("/")[0]
    k0 = np.array([float(x) for x in tokens[2:11]], dtype=np.float32).reshape(3, 3)
    k1 = np.array([float(x) for x in tokens[11:20]], dtype=np.float32).reshape(3, 3)
    t_0to1 = np.array([float(x) for x in tokens[20:36]], dtype=np.float32).reshape(4, 4)

    return {
        "scene": scene,
        "img0_rel": img0_rel,
        "img1_rel": img1_rel,
        "K0": k0,
        "K1": k1,
        "T_0to1": t_0to1,
    }


def filter_pairs_by_scene(
    lines: list[str],
    pairs_per_scene: int | None = None,
    max_scenes: int | None = None,
    target_scenes: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """按场景筛选样本对。

    返回:
        list of (line, scene, pair_idx_in_scene)
    """
    scene_groups: dict[str, list[str]] = {}
    for line in lines:
        s = line.strip().split()[0].split("/")[0]
        if s not in scene_groups:
            scene_groups[s] = []
        scene_groups[s].append(line)

    scene_keys = list(scene_groups.keys())
    if target_scenes:
        scene_keys = [s for s in scene_keys if s in target_scenes]
    if max_scenes:
        scene_keys = scene_keys[:max_scenes]

    filtered = []
    for s in scene_keys:
        pair_lines = scene_groups[s]
        if pairs_per_scene is not None:
            pair_lines = pair_lines[:pairs_per_scene]
        for p_idx, l in enumerate(pair_lines):
            filtered.append((l, s, p_idx))

    return filtered


def compute_ground_truth_matches(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    lines0: np.ndarray,
    lines1: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray,
    dist_thresh: float = 3.0,
    line_dist_thresh: float = 5.0,
    line_overlap_thresh: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """基于深度图与相对相机位姿严格反投影计算跨视角点/线匹配真值。"""
    n0 = len(kpts0)
    m0 = len(lines0)
    gt_matches0 = np.full(n0, -1, dtype=np.int64)
    gt_line_matches0 = np.full(m0, -1, dtype=np.int64)

    h0, w0 = depth0.shape[:2]
    h1, w1 = depth1.shape[:2]

    # ================= 1. 关键点匹配真值 =================
    if n0 > 0 and len(kpts1) > 0:
        xs0 = np.clip(np.round(kpts0[:, 0]).astype(int), 0, w0 - 1)
        ys0 = np.clip(np.round(kpts0[:, 1]).astype(int), 0, h0 - 1)
        z0 = depth0[ys0, xs0] / 1000.0  # 毫米转米

        valid0 = (z0 > 0.1) & (z0 < 10.0)
        valid_idx = np.where(valid0)[0]

        if len(valid_idx) > 0:
            uv1_homo = np.stack([kpts0[valid_idx, 0], kpts0[valid_idx, 1], np.ones(len(valid_idx))], axis=1)  # (V, 3)
            # 反投影至相机 0 坐标系: P0 = z * K0_inv * [u, v, 1]^T
            p0 = (np.linalg.inv(K0) @ uv1_homo.T * z0[valid_idx]).T  # (V, 3)
            p0_homo = np.concatenate([p0, np.ones((len(valid_idx), 1))], axis=1)  # (V, 4)

            # 变换至相机 1 坐标系: P1 = T_0to1 @ P0
            p1 = (T_0to1 @ p0_homo.T).T[:, :3]  # (V, 3)
            z1_proj = p1[:, 2]

            valid_in_front = z1_proj > 0.1
            sub_valid = valid_idx[valid_in_front]
            p1_valid = p1[valid_in_front]

            if len(sub_valid) > 0:
                # 投影到图像 1 平面: uv1_proj = K1 @ P1 / z1
                uv1_proj = (K1 @ p1_valid.T).T
                uv1_proj = uv1_proj[:, :2] / uv1_proj[:, 2:3]

                # 距离匹配
                dists = np.linalg.norm(uv1_proj[:, None, :] - kpts1[None, :, :], axis=2)  # (V_sub, N1)
                min_idx = np.argmin(dists, axis=1)
                min_dist = np.min(dists, axis=1)

                for orig_i, j, d in zip(sub_valid, min_idx, min_dist):
                    if d < dist_thresh:
                        gt_matches0[orig_i] = j

    # ================= 2. 线段匹配真值 =================
    if m0 > 0 and len(lines1) > 0:
        # 对每条线段在深度图上采样 10 个点
        t_steps = np.linspace(0.0, 1.0, 10)
        p_start0 = lines0[:, 0, :]  # (M0, 2)
        p_end0 = lines0[:, 1, :]    # (M0, 2)

        for i in range(m0):
            pts0 = p_start0[i:i+1] + t_steps[:, None] * (p_end0[i:i+1] - p_start0[i:i+1])  # (10, 2)
            xs = np.clip(np.round(pts0[:, 0]).astype(int), 0, w0 - 1)
            ys = np.clip(np.round(pts0[:, 1]).astype(int), 0, h0 - 1)
            zs = depth0[ys, xs] / 1000.0

            valid_pts = (zs > 0.1) & (zs < 10.0)
            if np.sum(valid_pts) < 6:
                continue

            pts0_v = pts0[valid_pts]
            zs_v = zs[valid_pts]
            uv_homo = np.stack([pts0_v[:, 0], pts0_v[:, 1], np.ones(len(pts0_v))], axis=1)
            p0 = (np.linalg.inv(K0) @ uv_homo.T * zs_v).T
            p0_homo = np.concatenate([p0, np.ones((len(p0), 1))], axis=1)
            p1 = (T_0to1 @ p0_homo.T).T[:, :3]

            if np.any(p1[:, 2] <= 0.1):
                continue

            proj_pts1 = (K1 @ p1.T).T
            proj_pts1 = proj_pts1[:, :2] / proj_pts1[:, 2:3]  # (V, 2)

            # 计算与 lines1 中所有线段的平均垂直几何距离
            best_j = -1
            min_mean_dist = line_dist_thresh

            for j, line1 in enumerate(lines1):
                lp1 = line1[0]
                lp2 = line1[1]
                v = lp2 - lp1
                v_len = np.linalg.norm(v)
                if v_len < 1e-4:
                    continue
                v_norm = v / v_len
                # 点到线段距离
                diffs = proj_pts1 - lp1
                proj_t = np.dot(diffs, v_norm)
                clamped_t = np.clip(proj_t, 0, v_len)
                closest = lp1 + clamped_t[:, None] * v_norm
                dists = np.linalg.norm(proj_pts1 - closest, axis=1)

                mean_d = np.mean(dists)
                overlap = np.sum((proj_t >= 0) & (proj_t <= v_len)) / len(proj_t)

                if mean_d < min_mean_dist and overlap >= line_overlap_thresh:
                    min_mean_dist = mean_d
                    best_j = j

            if best_j != -1:
                gt_line_matches0[i] = best_j

    return gt_matches0, gt_line_matches0


def main():
    parser = argparse.ArgumentParser(description="批量构建 ScanNet 语义增强 GlueStick 数据集")
    parser.add_argument("--pairs-file", default="data/scannet/scannet1500/pairs_calibrated.txt", help="配对索引路径")
    parser.add_argument("--data-dir", default="data/scannet/scannet1500", help="ScanNet 数据根目录")
    parser.add_argument("--out-dir", default="data/scannet_semantic_gluestick", help="生成数据集导出目录")
    parser.add_argument("--sam-checkpoint", default="weights/sam/sam_vit_h_4b8939.pth", help="SAM ViT-H 权重路径")
    parser.add_argument("--deeplsd-checkpoint", default="third_party/DeepLSD/weights/deeplsd_wireframe.tar", help="DeepLSD 权重")
    parser.add_argument("--sp-checkpoint", default="GlueStick/resources/weights/superpoint_v1.pth", help="SuperPoint 权重")
    parser.add_argument("--max-pairs", type=int, default=None, help="最大处理图像对数 (测试时使用)")
    parser.add_argument("--scenes", nargs="+", default=None, help="指定要处理的场景列表 (如 scene0707_00 scene0708_00)")
    parser.add_argument("--max-scenes", type=int, default=None, help="最多处理多少个场景")
    parser.add_argument("--pairs-per-scene", type=int, default=None, help="每个场景最多处理几对图")
    parser.add_argument("--image-npz-scenes", nargs="+", default=None, help="指定使用 image+npz 格式的场景列表")
    parser.add_argument("--num-image-npz-scenes", type=int, default=3, help="前几个场景使用 image+npz 格式 (默认: 3)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    args = parser.parse_args()

    pairs_path = Path(args.pairs_file).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_path.is_file():
        raise FileNotFoundError(f"未找到配对文件: {pairs_path}")

    with open(pairs_path, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip()]

    # 场景筛选
    if args.scenes or args.max_scenes or args.pairs_per_scene:
        selected_pairs = filter_pairs_by_scene(
            all_lines,
            pairs_per_scene=args.pairs_per_scene,
            max_scenes=args.max_scenes,
            target_scenes=args.scenes,
        )
    else:
        selected_pairs = [(l, l.split()[0].split("/")[0], idx) for idx, l in enumerate(all_lines)]

    if args.max_pairs:
        selected_pairs = selected_pairs[: args.max_pairs]

    # 统计所有参与构建的场景并确定使用 image+npz 格式的场景
    unique_scenes = []
    for _, s, _ in selected_pairs:
        if s not in unique_scenes:
            unique_scenes.append(s)

    if args.image_npz_scenes:
        image_npz_scenes = set(args.image_npz_scenes)
    else:
        num_img_scenes = max(0, min(args.num_image_npz_scenes, len(unique_scenes)))
        image_npz_scenes = set(unique_scenes[:num_img_scenes])

    images_dir = out_dir / "images"
    npz_dir = out_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    if len(image_npz_scenes) > 0:
        images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[*] 启动高精度语义 GlueStick 数据集生成: 共 {len(selected_pairs)} 对样本 ({len(unique_scenes)} 个场景)")
    print(f"[*] 输出目录: {out_dir}")
    print(f"[*] 存储模式配置 (Hybrid Storage Mode):")
    print(f"    - 独立图像格式 (image+npz): {len(image_npz_scenes)} 个场景 -> {sorted(list(image_npz_scenes))}")
    print(f"    - 纯NPZ内嵌格式 (npz): 剩余 {len(unique_scenes) - len(image_npz_scenes)} 个场景")
    print("=" * 70)

    # 1. 初始化三大核心引擎
    sam_extractor = SAMExtractor(
        model_type="vit_h",
        checkpoint_path=args.sam_checkpoint,
        device=args.device,
        use_fp16=True,
        num_line_samples=16,
    )
    deeplsd_extractor = DeepLSDExtractor(
        checkpoint_path=args.deeplsd_checkpoint,
        device=args.device,
        detect_lines=True,
    )
    wireframe_builder = WireframeBuilder(
        sp_weights_path=args.sp_checkpoint,
        device=args.device,
        nms_radius=3.0,
    )

    metadata_list = []
    target_size = (640, 480)  # (W, H)

    for idx, (line, scene, pair_idx) in enumerate(tqdm(selected_pairs, desc="构建进度")):
        info = parse_pairs_line(line)
        img0_rel = info["img0_rel"]
        img1_rel = info["img1_rel"]

        img0_full = data_dir / img0_rel
        img1_full = data_dir / img1_rel

        # 深度图路径推导: sceneXXXX_00/color/15.jpg -> sceneXXXX_00/depth/15.png
        depth0_full = data_dir / img0_rel.replace("/color/", "/depth/").replace(".jpg", ".png")
        depth1_full = data_dir / img1_rel.replace("/color/", "/depth/").replace(".jpg", ".png")

        if not (img0_full.is_file() and img1_full.is_file()):
            print(f"[!] 警告: 跳过缺失图像对 {img0_rel} - {img1_rel}")
            continue

        # 读取图像 (RGB) 与深度
        img0_raw = cv2.imread(str(img0_full))
        img1_raw = cv2.imread(str(img1_full))
        if img0_raw is None or img1_raw is None:
            continue

        img0_rgb = cv2.cvtColor(img0_raw, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_raw, cv2.COLOR_BGR2RGB)

        orig_h0, orig_w0 = img0_rgb.shape[:2]
        orig_h1, orig_w1 = img1_rgb.shape[:2]

        # 统一缩放至 640x480
        img0 = cv2.resize(img0_rgb, target_size, interpolation=cv2.INTER_AREA)
        img1 = cv2.resize(img1_rgb, target_size, interpolation=cv2.INTER_AREA)

        scale0_x = target_size[0] / orig_w0
        scale0_y = target_size[1] / orig_h0
        scale1_x = target_size[0] / orig_w1
        scale1_y = target_size[1] / orig_h1

        # 调整内参矩阵
        K0 = info["K0"].copy()
        K0[0, :] *= scale0_x
        K0[1, :] *= scale0_y
        K1 = info["K1"].copy()
        K1[0, :] *= scale1_x
        K1[1, :] *= scale1_y

        # 读取并缩放深度图
        depth0 = cv2.imread(str(depth0_full), cv2.IMREAD_UNCHANGED) if depth0_full.is_file() else None
        depth1 = cv2.imread(str(depth1_full), cv2.IMREAD_UNCHANGED) if depth1_full.is_file() else None
        if depth0 is None:
            depth0 = np.zeros((target_size[1], target_size[0]), dtype=np.uint16)
        elif depth0.shape != (target_size[1], target_size[0]):
            depth0 = cv2.resize(depth0, target_size, interpolation=cv2.INTER_NEAREST)

        if depth1 is None:
            depth1 = np.zeros((target_size[1], target_size[0]), dtype=np.uint16)
        elif depth1.shape != (target_size[1], target_size[0]):
            depth1 = cv2.resize(depth1, target_size, interpolation=cv2.INTER_NEAREST)

        # ---------------- 视图 0 特征提取 ----------------
        gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        dlsd0 = deeplsd_extractor.detect(gray0)
        sp0 = wireframe_builder.extract_superpoint(gray0)
        wf0 = wireframe_builder.build_wireframe(
            lines=dlsd0["lines"],
            line_scores=dlsd0["line_scores"],
            dense_descs=sp0["dense_descriptors"],
            extra_keypoints=sp0["keypoints"],
            extra_scores=sp0["scores"],
        )
        sam_emb0, meta0 = sam_extractor.extract_image_embeddings(img0)
        line_sem_descs0 = sam_extractor.pool_line_descriptors(sam_emb0, wf0["snapped_lines"], meta0)
        line_mask_attrs0 = sam_extractor.extract_line_mask_attributes(img0, wf0["snapped_lines"].cpu().numpy())

        # ---------------- 视图 1 特征提取 ----------------
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        dlsd1 = deeplsd_extractor.detect(gray1)
        sp1 = wireframe_builder.extract_superpoint(gray1)
        wf1 = wireframe_builder.build_wireframe(
            lines=dlsd1["lines"],
            line_scores=dlsd1["line_scores"],
            dense_descs=sp1["dense_descriptors"],
            extra_keypoints=sp1["keypoints"],
            extra_scores=sp1["scores"],
        )
        sam_emb1, meta1 = sam_extractor.extract_image_embeddings(img1)
        line_sem_descs1 = sam_extractor.pool_line_descriptors(sam_emb1, wf1["snapped_lines"], meta1)
        line_mask_attrs1 = sam_extractor.extract_line_mask_attributes(img1, wf1["snapped_lines"].cpu().numpy())

        # ---------------- 跨视角几何投影真值匹配 ----------------
        gt_m0, gt_lm0 = compute_ground_truth_matches(
            kpts0=wf0["junctions"].cpu().numpy(),
            kpts1=wf1["junctions"].cpu().numpy(),
            lines0=wf0["snapped_lines"].cpu().numpy(),
            lines1=wf1["snapped_lines"].cpu().numpy(),
            depth0=depth0,
            depth1=depth1,
            K0=K0,
            K1=K1,
            T_0to1=info["T_0to1"],
        )

        # ---------------- 序列化导出 ----------------
        sample_name = f"{scene}_pair{pair_idx:02d}"
        npz_filename = f"{sample_name}.npz"
        npz_rel_path = f"npz/{npz_filename}"
        sample_npz_path = npz_dir / npz_filename

        is_image_npz = (scene in image_npz_scenes)

        save_dict = {
            # 几何关键点与视觉描述子
            "keypoints0": wf0["junctions"].cpu().numpy(),
            "keypoints1": wf1["junctions"].cpu().numpy(),
            "keypoint_scores0": wf0["junc_scores"].cpu().numpy(),
            "keypoint_scores1": wf1["junc_scores"].cpu().numpy(),
            "descriptors0": wf0["junc_descs"].cpu().numpy(),
            "descriptors1": wf1["junc_descs"].cpu().numpy(),
            # 几何线段与拓扑
            "lines0": wf0["snapped_lines"].cpu().numpy(),
            "lines1": wf1["snapped_lines"].cpu().numpy(),
            "line_scores0": dlsd0["line_scores"],
            "line_scores1": dlsd1["line_scores"],
            "lines_junc_idx0": wf0["lines_junc_idx"].cpu().numpy(),
            "lines_junc_idx1": wf1["lines_junc_idx"].cpu().numpy(),
            "connectivity0": wf0["connectivity"].cpu().numpy(),
            "connectivity1": wf1["connectivity"].cpu().numpy(),
            "df0": dlsd0["df"],
            "df1": dlsd1["df"],
            # SAM 语义增强特征
            "line_semantic_descs0": line_sem_descs0.cpu().numpy(),
            "line_semantic_descs1": line_sem_descs1.cpu().numpy(),
            "line_mask_attrs0": line_mask_attrs0,
            "line_mask_attrs1": line_mask_attrs1,
            # 真实位姿与真值匹配
            "gt_matches0": gt_m0,
            "gt_line_matches0": gt_lm0,
            "K0": K0,
            "K1": K1,
            "T_0to1": info["T_0to1"],
        }

        meta_entry = {
            "sample_name": sample_name,
            "scene": scene,
            "pair_idx": pair_idx,
            "storage_format": "image+npz" if is_image_npz else "npz",
            "npz_file": npz_rel_path,
            "num_junc0": len(wf0["junctions"]),
            "num_junc1": len(wf1["junctions"]),
            "num_lines0": len(wf0["snapped_lines"]),
            "num_lines1": len(wf1["snapped_lines"]),
            "num_gt_matches": int(np.sum(gt_m0 >= 0)),
            "num_gt_line_matches": int(np.sum(gt_lm0 >= 0)),
        }

        if is_image_npz:
            img0_filename = f"{sample_name}_0.png"
            img1_filename = f"{sample_name}_1.png"
            img0_rel_path = f"images/{img0_filename}"
            img1_rel_path = f"images/{img1_filename}"

            # 导出独立清晰图像 (RGB)
            cv2.imwrite(str(images_dir / img0_filename), cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(images_dir / img1_filename), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))

            save_dict["img0_path"] = img0_rel_path
            save_dict["img1_path"] = img1_rel_path
            save_dict["storage_format"] = "image+npz"

            meta_entry["img0_file"] = img0_rel_path
            meta_entry["img1_file"] = img1_rel_path
        else:
            # 纯 npz 格式: 内嵌灰度图像
            save_dict["image0"] = gray0
            save_dict["image1"] = gray1
            save_dict["storage_format"] = "npz"

        np.savez_compressed(str(sample_npz_path), **save_dict)

        torch.cuda.empty_cache()

        metadata_list.append(meta_entry)

    # 保存元数据索引
    index_file = out_dir / "dataset_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] 数据集构建完成！共导出 {len(metadata_list)} 对样本至 {out_dir}")
    print(f"[✓] 索引表已保存至: {index_file}")


if __name__ == "__main__":
    main()
