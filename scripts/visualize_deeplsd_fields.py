"""Draw DeepLSD network outputs: distance field and line-angle field.

The network does not emit segments. Each view becomes one panel:
  original | distance field
  angle field | angle painted back onto the photo

Default writes three Structured3D examples under vis_deeplsd_model/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEEPLSD_ROOT = ROOT / "third_party" / "DeepLSD"
if str(DEEPLSD_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPLSD_ROOT))

DATA = ROOT / "Structured3D_perspective_full_00" / "Structured3D"
DEFAULT_VIEWS = (
    DATA / "scene_00000/2D_rendering/485145/perspective/full/0/rgb_rawlight.png",
    DATA / "scene_00000/2D_rendering/490854/perspective/full/0/rgb_rawlight.png",
    DATA / "scene_00175/2D_rendering/406304/perspective/full/0/rgb_rawlight.png",
)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
DF_MAX = 5.0


def load_model(weight: Path, device):
    import torch
    from deeplsd.models.deeplsd_inference import DeepLSD

    ckpt = torch.load(weight, map_location="cpu", weights_only=False)
    net = DeepLSD({"detect_lines": False})
    net.load_state_dict(ckpt["model"])
    return net.to(device).eval()


def infer_fields(net, gray: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    tensor = torch.from_numpy(gray).to(device=device, dtype=torch.float32)[None, None]
    tensor.div_(255.0)
    with torch.inference_mode():
        out = net({"image": tensor})
    df = out["df"][0].float().cpu().numpy()
    angle = out["line_level"][0].float().cpu().numpy()
    return df, angle


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.truetype(FONT_PATH, size, index=2)


def colorize_df(df: np.ndarray) -> np.ndarray:
    distance = np.clip(np.nan_to_num(df, nan=DF_MAX) / DF_MAX, 0.0, 1.0)
    level = np.round((1.0 - distance) * 255).astype(np.uint8)
    return cv2.applyColorMap(level, cv2.COLORMAP_INFERNO)


def colorize_angle(angle: np.ndarray, df: np.ndarray, cutoff: float = 2.5) -> np.ndarray:
    distance = np.nan_to_num(df, nan=DF_MAX)
    direction = np.mod(np.nan_to_num(angle, nan=0.0), np.pi)
    hue = np.round(np.clip(direction / np.pi, 0.0, 1.0) * 179).astype(np.uint8)
    near = np.clip((cutoff - distance) / cutoff, 0.0, 1.0)
    saturation = np.round(255 * np.clip(near * 1.6, 0.0, 1.0)).astype(np.uint8)
    value = np.round(28 + 210 * near).astype(np.uint8)
    hsv = np.dstack([hue, saturation, value])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_angle(bgr: np.ndarray, angle: np.ndarray, df: np.ndarray, cutoff: float = 1.6) -> np.ndarray:
    color = colorize_angle(angle, df, cutoff=cutoff)
    fade = np.clip((cutoff - np.nan_to_num(df, nan=DF_MAX)) / 0.7, 0.0, 1.0)[..., None]
    base = bgr.astype(np.float32) * 0.45
    mixed = base * (1.0 - fade) + color.astype(np.float32) * fade
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _legend(width: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    font = _font(22)
    canvas = Image.new("RGB", (width, 118), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), "距离场：亮 = 在直线上（0 px），暗 = 远离（5 px）", font=font, fill=(235, 235, 235))
    bar_h = 28
    y0 = 40
    bar_w = width // 2 - 40
    ramp = np.repeat(np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :], bar_h, axis=0)
    distance_bar = cv2.cvtColor(cv2.applyColorMap(ramp, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(distance_bar), (16, y0))
    draw.text((16, y0 + bar_h + 2), "0 px", font=font, fill=(220, 220, 220))
    draw.text((16 + bar_w - 52, y0 + bar_h + 2), "5 px", font=font, fill=(220, 220, 220))

    draw.text((width // 2 + 8, 8), "角度场：颜色 = 无向直线方向，只在靠近直线处显色", font=font, fill=(235, 235, 235))
    ax0 = width // 2 + 16
    hue = np.linspace(0, 179, bar_w, dtype=np.uint8)
    hsv = np.dstack([
        np.repeat(hue[None, :], bar_h, axis=0),
        np.full((bar_h, bar_w), 220, np.uint8),
        np.full((bar_h, bar_w), 230, np.uint8),
    ])
    canvas.paste(Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)), (ax0, y0))
    for degree in (0, 45, 90, 135, 180):
        label = {0: "0° 水平", 90: "90° 竖直"}.get(degree, f"{degree}°")
        tx = ax0 + int(degree / 180 * (bar_w - 1))
        draw.line([(tx, y0 - 4), (tx, y0)], fill=(255, 255, 255))
        draw.text((tx - 8, y0 + bar_h + 2), label, font=font, fill=(220, 220, 220))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _titled(bgr: np.ndarray, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    height, width = bgr.shape[:2]
    bar = Image.new("RGB", (width, 52), (24, 24, 24))
    ImageDraw.Draw(bar).text((16, 10), title, font=_font(32), fill=(245, 245, 245))
    header = cv2.cvtColor(np.asarray(bar), cv2.COLOR_RGB2BGR)
    return np.vstack([header, bgr])


def render_panel(bgr: np.ndarray, df: np.ndarray, angle: np.ndarray) -> np.ndarray:
    panels = [
        _titled(bgr, "原图"),
        _titled(colorize_df(df), "距离场  到最近直线的距离"),
        _titled(colorize_angle(angle, df), "角度场  颜色是直线方向"),
        _titled(overlay_angle(bgr, angle, df), "叠在原图上  只画距离 < 1.6 px"),
    ]
    gap = np.full((panels[0].shape[0], 8, 3), 24, np.uint8)
    top = np.hstack([panels[0], gap, panels[1]])
    bottom = np.hstack([panels[2], gap, panels[3]])
    seam = np.full((8, top.shape[1], 3), 24, np.uint8)
    body = np.vstack([top, seam, bottom])
    legend = _legend(body.shape[1])
    spacer = np.full((8, body.shape[1], 3), 24, np.uint8)
    return np.vstack([body, spacer, legend])


def view_name(rgb_path: Path) -> str:
    parts = rgb_path.parts
    if len(parts) >= 7 and parts[-3] == "full":
        return f"{parts[-7]}_{parts[-5]}_v{parts[-2]}"
    return rgb_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw DeepLSD distance and angle fields.")
    parser.add_argument("images", nargs="*", type=Path, help="rgb images; default is three Structured3D examples")
    parser.add_argument("--out", type=Path, default=ROOT / "Structured3D_perspective_full_00" / "vis_deeplsd_model")
    parser.add_argument("--weight", type=Path, default=DEEPLSD_ROOT / "weights" / "deeplsd_wireframe.tar")
    args = parser.parse_args()

    images = args.images or [path for path in DEFAULT_VIEWS if path.is_file()]
    if not images:
        raise SystemExit("no input images")
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(args.weight, device)
    for rgb_path in images:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rgb_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        df, angle = infer_fields(net, gray, device)
        name = view_name(rgb_path)
        dest = args.out / f"{name}.jpg"
        cv2.imwrite(str(dest), render_panel(bgr, df, angle), [cv2.IMWRITE_JPEG_QUALITY, 92])
        finite = np.isfinite(df)
        print(
            f"[+] {dest}  df {float(np.nanmin(df)):.2f}..{float(np.nanmax(df)):.2f}"
            f"  nan {int((~finite).sum())}  df<1.5 {(df < 1.5).mean() * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
