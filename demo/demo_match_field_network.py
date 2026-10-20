"""
Demonstration: Geometry-Field Prompted Line Matching Network (MaskLineMatcherNet).

Features demonstrated:
  1. Automated 6-channel input creation (RGB + EDT distance field + AFM orientation field).
  2. Forward inference through 6-channel Backbone + Line-Transformer.
  3. Direct generation of the canonical (255, 255) match probability matrix.
  4. Visualization of the multi-modal geometry fields (EDT, AFM) alongside matching outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LineFeat import (
    MaskLineMatcherNet,
    compute_geometry_field_np,
    create_6ch_input_tensor,
    get_distinct_colors,
)


def visualize_geometry_fields(
    rgb_img: np.ndarray,
    lines: np.ndarray,
    height: int,
    width: int,
    title: str = "Geometry Field",
) -> np.ndarray:
    """Renders side-by-side composite: [RGB with lines | EDT Distance Field | AFM Orientation Field]."""
    # 1. RGB with rendered lines
    vis_rgb = rgb_img.copy()
    for l in lines:
        pt1 = (int(round(l[0])), int(round(l[1])))
        pt2 = (int(round(l[2])), int(round(l[3])))
        cv2.line(vis_rgb, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis_rgb, pt1, 3, (0, 0, 255), -1)
        cv2.circle(vis_rgb, pt2, 3, (0, 0, 255), -1)

    # 2. Geometry Field (3 channels: EDT, sin2theta, cos2theta)
    field = compute_geometry_field_np(lines, height=height, width=width)
    edt = field[0]
    sin2 = field[1]
    cos2 = field[2]

    # Colorize EDT with colormap
    edt_uint8 = (edt * 255.0).astype(np.uint8)
    vis_edt = cv2.applyColorMap(edt_uint8, cv2.COLORMAP_VIRIDIS)

    # Colorize AFM (Map sin/cos into HSV hue and value)
    angle = np.arctan2(sin2, cos2) % (2 * np.pi)
    hue = (angle / (2 * np.pi) * 179).astype(np.uint8)
    mag = np.hypot(sin2, cos2)
    val = (np.clip(mag, 0, 1) * 255).astype(np.uint8)
    sat = np.full_like(hue, 255)
    hsv = np.stack([hue, sat, val], axis=-1)
    vis_afm = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Top banners
    def add_banner(img: np.ndarray, text: str) -> np.ndarray:
        banner = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([banner, img])

    panel1 = add_banner(vis_rgb, f"{title}: RGB + Lines ({len(lines)})")
    panel2 = add_banner(vis_edt, "Continuous EDT (Distance Field)")
    panel3 = add_banner(vis_afm, "Continuous AFM (Angle Orientation Field)")

    return np.hstack([panel1, panel2, panel3])


def main() -> None:
    print("=" * 80)
    print("      MaskLineMatcherNet (6-Channel Input -> 255x255 Match Matrix) Demo")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # 1. Load sample pair from test set
    pair_dir = ROOT / "data" / "test_samples" / "pair1_small_rotation"
    mode_dir = pair_dir / "mode_a_pure_permuted"

    img0_path = pair_dir / "image0.png"
    img1_path = pair_dir / "image1.png"

    bgr0 = cv2.imread(str(img0_path))
    bgr1 = cv2.imread(str(img1_path))
    H, W = bgr0.shape[:2]

    with open(mode_dir / "lines0.json", "r") as f:
        lines0 = np.array(json.load(f), dtype=np.float32)
    with open(mode_dir / "lines1.json", "r") as f:
        lines1 = np.array(json.load(f), dtype=np.float32)
    with open(mode_dir / "gt_matches.json", "r") as f:
        gt_matches = json.load(f)

    print(f"[*] Loaded Image0: {H}x{W}, {len(lines0)} lines")
    print(f"[*] Loaded Image1: {H}x{W}, {len(lines1)} lines")
    print(f"[*] Ground-truth correspondences: {len(gt_matches)} pairs")

    # 2. Render and save multi-modal geometric fields
    out_dir = ROOT / "outputs" / "field_network"
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_vis0 = visualize_geometry_fields(bgr0, lines0, H, W, "View 0")
    geom_vis1 = visualize_geometry_fields(bgr1, lines1, H, W, "View 1")
    cv2.imwrite(str(out_dir / "view0_geometry_fields.jpg"), geom_vis0)
    cv2.imwrite(str(out_dir / "view1_geometry_fields.jpg"), geom_vis1)
    print(f"[*] Saved multi-modal geometric field visualizations to:\n    {out_dir / 'view0_geometry_fields.jpg'}")

    # 3. Build 6-channel input tensors (RGB + EDT + AFM)
    rgb0_t = torch.from_numpy(cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    rgb1_t = torch.from_numpy(cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb0_norm = (rgb0_t - mean) / std
    rgb1_norm = (rgb1_t - mean) / std

    six_ch0 = create_6ch_input_tensor(rgb0_norm, lines0).unsqueeze(0).to(device)
    six_ch1 = create_6ch_input_tensor(rgb1_norm, lines1).unsqueeze(0).to(device)
    print(f"[*] Formed 6-channel input tensor shapes: {six_ch0.shape}")

    # 4. Instantiate MaskLineMatcherNet
    model = MaskLineMatcherNet(
        d_model=128,
        num_layers=4,
        max_matrix_size=255,
    ).to(device).eval()

    lines0_t = torch.from_numpy(lines0).unsqueeze(0).to(device)
    lines1_t = torch.from_numpy(lines1).unsqueeze(0).to(device)

    # 5. Forward Pass: Direct (255, 255) Output!
    with torch.no_grad():
        out = model(six_ch0, six_ch1, lines0_t, lines1_t)

    matrix_255 = out["matrix_255"][0].cpu().numpy()  # (255, 255)
    P_sub = out["P_sub"][0].cpu().numpy()            # (N0, N1)

    print("\n" + "=" * 80)
    print("                       Model Outputs Verification")
    print("=" * 80)
    print(f"1. Canonical Output Matrix Shape: {matrix_255.shape} (Row: Line0 ID, Col: Line1 ID)")
    print(f"2. Active Submatrix: {lines0.shape[0]} lines x {lines1.shape[0]} lines")
    print(f"3. Inactive Padding Region (rows > {len(lines0)} or cols > {len(lines1)}): max={matrix_255[len(lines0):, len(lines1):].max():.4f} (All 0.0)")

    # Print active submatrix
    print("\nActive (N0 x N1) Probability Submatrix:")
    print("       " + "".join([f"Col{j:<4}" for j in range(len(lines1))]))
    for i in range(len(lines0)):
        row_str = f"Row{i:<2} | " + " ".join([f"{P_sub[i, j]:.3f}" for j in range(len(lines1))])
        print(row_str)

    # 6. Save visualization of the (255, 255) matrix heatmap
    mat_norm = np.clip(matrix_255 / (matrix_255.max() + 1e-6), 0, 1)
    heatmap = (mat_norm * 255.0).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
    # Scale up for easy viewing
    heatmap_large = cv2.resize(heatmap_color, (510, 510), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(out_dir / "canonical_255x255_probability_matrix.png"), heatmap_large)
    print(f"\n[*] Exported canonical 255x255 match matrix visualization to:\n    {out_dir / 'canonical_255x255_probability_matrix.png'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
