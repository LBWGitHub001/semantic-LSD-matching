"""
Modular Architecture Visualizer Engine for Semantic LSD Matching Networks.
Renders clean, publication-quality 4K topology diagrams with explicit input/output tensor shapes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"


class BaseArchitectureVisualizer:
    """Base class providing drawing primitives for modern deep neural network diagrams."""

    def __init__(self, width: float = 24.0, height: float = 13.5, dpi: int = 180) -> None:
        self.width = width
        self.height = height
        self.dpi = dpi
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def setup_canvas(self, bg_color: str = "#0B0F19") -> None:
        """Initializes normalized canvas (0 to 1 on both axes)."""
        self.fig = plt.figure(figsize=(self.width, self.height), facecolor=bg_color, dpi=self.dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(bg_color)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")

    def draw_badge(
        self,
        x: float,
        y: float,
        text: str,
        bg_color: str,
        text_color: str = "#FFFFFF",
        fontsize: float = 8.5,
        width: Optional[float] = None,
        height: float = 0.024,
    ) -> None:
        """Draws a pill-shaped badge for tensor shapes or category tags."""
        if width is None:
            width = len(text) * 0.0072 + 0.020
        rect = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            facecolor=bg_color,
            edgecolor="none",
            zorder=10,
        )
        self.ax.add_patch(rect)
        self.ax.text(
            x,
            y,
            text,
            color=text_color,
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
            fontfamily="sans-serif",
        )

    def draw_module_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str = "",
        bg_color: str = "#1E293B",
        border_color: str = "#3B82F6",
        in_shape: Optional[str] = None,
        out_shape: Optional[str] = None,
        radius: float = 0.015,
        title_color: str = "#FFFFFF",
    ) -> None:
        """Draws a modern rounded module box with header, body and tensor shape badges."""
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=1.8,
            zorder=3,
        )
        self.ax.add_patch(rect)

        self.ax.text(
            x + w / 2,
            y + h - 0.025,
            title,
            color=title_color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=5,
            fontfamily="sans-serif",
        )

        if subtitle:
            self.ax.text(
                x + w / 2,
                y + h - 0.045,
                subtitle,
                color="#94A3B8",
                fontsize=8.5,
                ha="center",
                va="center",
                zorder=5,
                fontfamily="sans-serif",
            )

        if in_shape:
            self.draw_badge(
                x + 0.045,
                y + h - 0.005,
                f"IN: {in_shape}",
                bg_color="#0F766E",
                text_color="#CCFBF1",
                fontsize=8,
            )

        if out_shape:
            self.draw_badge(
                x + w - 0.055,
                y + 0.005,
                f"OUT: {out_shape}",
                bg_color="#1D4ED8",
                text_color="#DBEAFE",
                fontsize=8,
            )

    def draw_arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#64748B",
        label: Optional[str] = None,
        label_pos: float = 0.5,
        lw: float = 2.0,
        style: str = "->",
    ) -> None:
        """Draws a directed connection arrow with an optional tensor label pill."""
        self.ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style,
                color=color,
                lw=lw,
                mutation_scale=14,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=4,
        )
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos
            self.draw_badge(lx, ly, label, bg_color="#0F172A", text_color="#E2E8F0", fontsize=8.2, height=0.022)

    def draw_top_banner(self, title: str, subtitle: str, tag_left: str, tag_right: str) -> None:
        """Renders unified top header banner."""
        banner_rect = FancyBboxPatch(
            (0.015, 0.915),
            0.97,
            0.072,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="#111827",
            edgecolor="#1F2937",
            linewidth=1.2,
            zorder=2,
        )
        self.ax.add_patch(banner_rect)

        self.ax.text(0.5, 0.965, title, color="#F8FAFC", fontsize=18, fontweight="bold", ha="center", va="center", zorder=5)
        self.ax.text(0.5, 0.935, subtitle, color="#94A3B8", fontsize=11, ha="center", va="center", zorder=5)

        self.draw_badge(0.10, 0.95, tag_left, bg_color="#065F46", text_color="#6EE7B7", fontsize=9.5)
        self.draw_badge(0.90, 0.95, tag_right, bg_color="#1E3A8A", text_color="#93C5FD", fontsize=9.5)

    def draw_bottom_footer(self, formula: str, specs: str) -> None:
        """Renders unified bottom footer banner with loss function and specs."""
        foot_rect = FancyBboxPatch(
            (0.015, 0.012),
            0.97,
            0.052,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            facecolor="#0F172A",
            edgecolor="#334155",
            linewidth=1.2,
            zorder=2,
        )
        self.ax.add_patch(foot_rect)

        foot_text = f"Objective:  {formula}    |    {specs}"
        self.ax.text(0.5, 0.038, foot_text, color="#94A3B8", fontsize=10.5, ha="center", va="center", zorder=5)

    def save(self, out_path: Path | str) -> Path:
        """Saves canvas to PNG image and closes matplotlib figure."""
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), dpi=self.dpi, facecolor="#0B0F19", edgecolor="none")
        plt.close(self.fig)
        return out_p


class XFeatLineMatcherVisualizer(BaseArchitectureVisualizer):
    """Visualizes XFeatLineMatcher (Visual Patches + LSD Lines + BBox + Transformer + Sinkhorn)."""

    def render(self, out_path: Path | str) -> Path:
        self.setup_canvas()

        self.draw_top_banner(
            title="XFeatLineMatcher: Multi-Modal Straight Line & Region Semantic Alignment Network",
            subtitle="Detailed Module Topology, Forward Pipeline Data Flow & Explicit Input/Output Tensor Dimensionality Specifications",
            tag_left="SOTA 79.2% F1",
            tag_right="Params: 1,722,081 (1.72M)",
        )

        # Stage 1: Dual-View Inputs (Col 1)
        col1_x, col1_w = 0.02, 0.15
        g1_bg = FancyBboxPatch((col1_x - 0.006, 0.08), col1_w + 0.012, 0.81,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g1_bg)
        self.ax.text(col1_x + col1_w / 2, 0.87, "STAGE 1: DUAL-VIEW INPUTS", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col1_x, 0.61, col1_w, 0.23,
                             title="1. Candidate Region Patches",
                             subtitle="Adaptive-margin Crop & Resize",
                             bg_color="#064E3B", border_color="#10B981",
                             in_shape="Image (B, 3, H, W)",
                             out_shape="(B, M, 3, 112, 112)",
                             title_color="#A7F3D0")
        self.ax.text(col1_x + col1_w / 2, 0.725, "• Normalized RGB 3-Channel\n• Bilinear Interpolation to 112x112\n• Isolates Object from Cluttered Scene\n• M: Number of Patches per View",
                     color="#D1FAE5", fontsize=8.5, ha="center", va="center", zorder=5, linespacing=1.5)

        self.draw_module_box(col1_x, 0.35, col1_w, 0.23,
                             title="2. Associated LSD Lines",
                             subtitle="Sub-pixel Straight Lines",
                             bg_color="#451A03", border_color="#F59E0B",
                             in_shape="LSD Lines (B, N, 4)",
                             out_shape="(B, M, K=8, 5)",
                             title_color="#FDE68A")
        self.ax.text(col1_x + col1_w / 2, 0.465, "• K=8 Top Straight Lines per Patch\n• 5D Vector: [cu, cv, length, sin2θ, cos2θ]\n• Orientation Invariance: θ ∈ [0, π)\n• In-patch Localized Coordinates",
                     color="#FEF3C7", fontsize=8.5, ha="center", va="center", zorder=5, linespacing=1.5)

        self.draw_module_box(col1_x, 0.10, col1_w, 0.21,
                             title="3. Bounding Boxes Layout",
                             subtitle="2D Global Spatial Frame",
                             bg_color="#1E3A8A", border_color="#3B82F6",
                             in_shape="BBoxes [x1, y1, x2, y2]",
                             out_shape="(B, M, 4)",
                             title_color="#BFDBFE")
        self.ax.text(col1_x + col1_w / 2, 0.20, "• Global Normalized Coordinates:\n  [cx/W, cy/H, w/W, h/H] ∈ [0, 1]\n• Encodes Absolute Context Layout\n• Resolves Geometric Ambiguity",
                     color="#DBEAFE", fontsize=8.5, ha="center", va="center", zorder=5, linespacing=1.5)

        # Stage 2: Tri-Modal Feature Encoders (Col 2)
        col2_x, col2_w = 0.21, 0.19
        g2_bg = FancyBboxPatch((col2_x - 0.006, 0.08), col2_w + 0.012, 0.81,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g2_bg)
        self.ax.text(col2_x + col2_w / 2, 0.87, "STAGE 2: TRI-MODAL FEATURE ENCODING", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col2_x, 0.58, col2_w, 0.27,
                             title="XFeat Feature Pyramid (FPN)",
                             subtitle="Multi-Scale Pyramidal Visual Backbone",
                             bg_color="#062D24", border_color="#059669",
                             in_shape="(B*M, 3, 112, 112)",
                             out_shape="t_vis: (B, M, 192)",
                             title_color="#6EE7B7")

        fpn_details = [
            ("Level P1 (Stride 2)", "(B*M, 64, 56, 56) ➔ AvgPool ➔ 64D", "Fine-grain Edge Details"),
            ("Level P2 (Stride 4)", "(B*M, 64, 28, 28) ➔ AvgPool ➔ 64D", "Part & Shape Structure"),
            ("Level P3 (Stride 8)", "(B*M, 64, 14, 14) ➔ AvgPool ➔ 64D", "Semantic Object Class")
        ]
        for i, (lvl, shp, desc) in enumerate(fpn_details):
            by = 0.73 - i * 0.046
            rect_sub = FancyBboxPatch((col2_x + 0.01, by), col2_w - 0.02, 0.038,
                                     boxstyle="round,pad=0.002,rounding_size=0.006",
                                     facecolor="#041F19", edgecolor="#10B981", linewidth=1, zorder=4)
            self.ax.add_patch(rect_sub)
            self.ax.text(col2_x + 0.02, by + 0.019, lvl, color="#34D399", fontsize=8, fontweight="bold", va="center", zorder=5)
            self.ax.text(col2_x + col2_w - 0.02, by + 0.024, shp, color="#A7F3D0", fontsize=7.5, ha="right", va="center", zorder=5)
            self.ax.text(col2_x + col2_w - 0.02, by + 0.010, desc, color="#6EE7B7", fontsize=7, ha="right", va="center", zorder=5)

        self.ax.text(col2_x + col2_w / 2, 0.595, "Fusion: torch.cat([P1, P2, P3], dim=-1) ➔ 192D Visual Token",
                     color="#A7F3D0", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5)

        self.draw_module_box(col2_x, 0.34, col2_w, 0.21,
                             title="PatchLineGeometryEncoder",
                             subtitle="5D Geometry Projection + K-Line Pooling",
                             bg_color="#2C1204", border_color="#D97706",
                             in_shape="(B*M, K=8, 5)",
                             out_shape="t_line: (B, M, 64)",
                             title_color="#FCD34D")
        self.ax.text(col2_x + col2_w / 2, 0.435,
                     "• Line MLP: Linear(5➔64) ➔ LayerNorm ➔ ReLU\n  ➔ Linear(64➔64) ➔ LayerNorm ➔ ReLU\n• FP16-Safe Masked MaxPool1d over K lines\n• Masked AveragePool1d over K lines\n• Learnable fallback null_line_token (64D) for K=0",
                     color="#FEF3C7", fontsize=8, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_module_box(col2_x, 0.10, col2_w, 0.20,
                             title="ObjectGeometryEncoder",
                             subtitle="2D Spatial Coordinate Mapping",
                             bg_color="#0F224A", border_color="#2563EB",
                             in_shape="(B*M, 4)",
                             out_shape="t_bbox: (B, M, 64)",
                             title_color="#93C5FD")
        self.ax.text(col2_x + col2_w / 2, 0.19,
                     "• Spatial MLP: Linear(4➔64) ➔ LayerNorm ➔ ReLU\n  ➔ Linear(64➔64) ➔ LayerNorm ➔ ReLU\n• Maps [cx, cy, w, h] to 64D position embedding\n• Provides spatial inductive bias for cross-view reasoning",
                     color="#DBEAFE", fontsize=8, ha="center", va="center", zorder=5, linespacing=1.4)

        # Arrows Col 1 -> Col 2
        self.draw_arrow(col1_x + col1_w, 0.725, col2_x, 0.725, color="#10B981", label="(B,M,3,112,112)")
        self.draw_arrow(col1_x + col1_w, 0.465, col2_x, 0.445, color="#F59E0B", label="(B,M,8,5)")
        self.draw_arrow(col1_x + col1_w, 0.200, col2_x, 0.200, color="#3B82F6", label="(B,M,4)")

        # Stage 3: Multi-Modal Token Fusion (Col 3)
        col3_x, col3_w = 0.43, 0.12
        g3_bg = FancyBboxPatch((col3_x - 0.006, 0.08), col3_w + 0.012, 0.81,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g3_bg)
        self.ax.text(col3_x + col3_w / 2, 0.87, "STAGE 3: TOKEN FUSION", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col3_x, 0.28, col3_w, 0.48,
                             title="Multi-Modal Fusion Layer",
                             subtitle="Concatenation & Linear Projection",
                             bg_color="#2E1065", border_color="#7C3AED",
                             in_shape="Concat 320D",
                             out_shape="t_fused: (B, M, 128)",
                             title_color="#DDD6FE")
        self.ax.text(col3_x + col3_w / 2, 0.56,
                     "1. Modal Concatenation:\n   t_raw = [t_vis (192D) ||\n            t_line (64D) ||\n            t_bbox (64D)]\n   Shape: (B, M, 320)\n\n"
                     "2. Projection MLP:\n   • Linear(320 ➔ 128)\n   • LayerNorm(128)\n   • ReLU(inplace=True)\n\n"
                     "3. Unified Representation:\n   d_model = 128 Token\n   per Object Candidate",
                     color="#EDE9FE", fontsize=8.5, ha="center", va="center", zorder=5, linespacing=1.3)

        self.draw_arrow(col2_x + col2_w, 0.70, col3_x, 0.58, color="#10B981", label="t_vis (192D)")
        self.draw_arrow(col2_x + col2_w, 0.44, col3_x, 0.50, color="#F59E0B", label="t_line (64D)")
        self.draw_arrow(col2_x + col2_w, 0.20, col3_x, 0.42, color="#3B82F6", label="t_bbox (64D)")

        # Stage 4: Object Graph Transformer (Col 4)
        col4_x, col4_w = 0.58, 0.16
        g4_bg = FancyBboxPatch((col4_x - 0.006, 0.08), col4_w + 0.012, 0.81,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g4_bg)
        self.ax.text(col4_x + col4_w / 2, 0.87, "STAGE 4: GRAPH TRANSFORMER", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col4_x, 0.22, col4_w, 0.60,
                             title="ObjectGraphTransformer",
                             subtitle="2 Layers × 4 Heads Cross-View Reasoning",
                             bg_color="#083344", border_color="#0891B2",
                             in_shape="X0, X1: (B, M, 128)",
                             out_shape="X_ctx: (B, M, 128)",
                             title_color="#A5F3FC")

        tr_blocks = [
            ("Layer 1: Self & Cross Attention", 0.64, "#0E4A59", [
                "• Intra-View Self-Attn: MHA(128, 4 heads)",
                "  + Residual Add & LayerNorm(128)",
                "• Inter-View Cross-Attn: MHA(128, 4 heads)",
                "  + Residual Add & LayerNorm(128)",
                "• FFN: Linear(128➔256) ➔ ReLU ➔ Linear(256➔128)",
                "  + Residual Add & LayerNorm(128)"
            ]),
            ("Layer 2: Iterative Deep Context", 0.38, "#0E4A59", [
                "• Intra-View Self-Attn: MHA(128, 4 heads)",
                "  + Residual Add & LayerNorm(128)",
                "• Inter-View Cross-Attn: MHA(128, 4 heads)",
                "  + Residual Add & LayerNorm(128)",
                "• FFN: Linear(128➔256) ➔ ReLU ➔ Linear(256➔128)",
                "  + Residual Add & LayerNorm(128)"
            ])
        ]
        for title_tr, by, bg_tr, items in tr_blocks:
            rect_tr = FancyBboxPatch((col4_x + 0.008, by), col4_w - 0.016, 0.165,
                                    boxstyle="round,pad=0.002,rounding_size=0.008",
                                    facecolor=bg_tr, edgecolor="#06B6D4", linewidth=1, zorder=4)
            self.ax.add_patch(rect_tr)
            self.ax.text(col4_x + col4_w / 2, by + 0.15, title_tr, color="#67E8F9", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=5)
            self.ax.text(col4_x + 0.015, by + 0.075, "\n".join(items), color="#CFFAFE", fontsize=7.2, va="center", zorder=5, linespacing=1.35)

        self.ax.text(col4_x + col4_w / 2, 0.24, "Bidirectional Token Interaction (View 0 ⇄ View 1)",
                     color="#A5F3FC", fontsize=8, fontweight="bold", ha="center", va="center", zorder=5)

        self.draw_arrow(col3_x + col3_w, 0.52, col4_x, 0.52, color="#8B5CF6", label="t_fused (128D)")

        # Stage 5: Log-Sinkhorn & Decision (Col 5)
        col5_x, col5_w = 0.77, 0.21
        g5_bg = FancyBboxPatch((col5_x - 0.006, 0.08), col5_w + 0.012, 0.81,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g5_bg)
        self.ax.text(col5_x + col5_w / 2, 0.87, "STAGE 5: OPTIMAL TRANSPORT & MATCHER", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col5_x, 0.44, col5_w, 0.38,
                             title="Log-Sinkhorn Optimal Transport",
                             subtitle="Doubly Stochastic Matrix Solver with Dustbin",
                             bg_color="#450A0A", border_color="#EF4444",
                             in_shape="X0, X1: (B, M, 128)",
                             out_shape="P: (B, M0+1, M1+1)",
                             title_color="#FCA5A5")

        self.ax.text(col5_x + col5_w / 2, 0.61,
                     "1. Affinity Matrix S ∈ R^{M0 × M1}:\n"
                     "   S_ij = (1 / √128) · <X0_i_ctx, X1_j_ctx>\n\n"
                     "2. Dustbin Augmentation:\n"
                     "   Learnable parameter α (absorbs unmatched lines/objects)\n"
                     "   Augmented Matrix S_bar ∈ R^{(M0+1) × (M1+1)}\n\n"
                     "3. Sinkhorn Dual Iterations (20 iters in Log-Space):\n"
                     "   u^(t+1) = log r - logsumexp(S_bar + v^(t))\n"
                     "   v^(t+1) = log c - logsumexp(S_bar^T + u^(t+1))\n\n"
                     "4. Soft Assignment Probability: P = exp(log_P)",
                     color="#FEE2E2", fontsize=8, ha="center", va="center", zorder=5, linespacing=1.3)

        self.draw_module_box(col5_x, 0.10, col5_w, 0.31,
                             title="Dual Mutual Filter & Verification",
                             subtitle="Hard Assignment & Rejection Gate",
                             bg_color="#1E1B4B", border_color="#6366F1",
                             in_shape="Prob Matrix P",
                             out_shape="Final Match Pairs {(i, j)}",
                             title_color="#C7D2FE")

        self.ax.text(col5_x + col5_w / 2, 0.24,
                     "• Mutual Nearest Neighbor (MNN) Criterion:\n"
                     "    j = argmax_k P_ik   AND   i = argmax_k P_kj\n"
                     "• Probability Confidence Threshold: P_ij ≥ 0.20\n"
                     "• Dustbin Isolation: If i=M0+1 or j=M1+1 ➔ Rejected Outlier\n"
                     "• Rigid Motion Consistency Margin: d_disp ≤ 0.15\n\n"
                     "★ SOTA Performance: 79.2% Semantic F1 | 74.3% Raw Visual F1",
                     color="#E0E7FF", fontsize=8, ha="center", va="center", zorder=5, linespacing=1.35)

        self.draw_arrow(col4_x + col4_w, 0.63, col5_x, 0.63, color="#06B6D4", label="X0_ctx, X1_ctx")
        self.draw_arrow(col5_x + col5_w / 2, 0.44, col5_x + col5_w / 2, 0.41, color="#EF4444", label="Soft Prob P")

        self.draw_bottom_footer(
            formula=r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NLL}}(\bar{P}, Y) + \lambda_{\mathrm{dustbin}} \cdot \mathcal{L}_{\mathrm{dustbin}} + \lambda_{\mathrm{motion}} \cdot \mathcal{L}_{\mathrm{motion}}$",
            specs="AMP FP16 Hardware Accelerated     |     Inference: 4.8 ms/pair     |     Batch: 32",
        )

        return self.save(out_path)


class XFeatObjectMatcherVisualizer(BaseArchitectureVisualizer):
    """Visualizes XFeatObjectMatcher (Pure Object-Level Visual & Layout Alignment)."""

    def render(self, out_path: Path | str) -> Path:
        self.setup_canvas()

        self.draw_top_banner(
            title="XFeatObjectMatcher: Object Region Appearance & Spatial Layout Alignment Network",
            subtitle="Pyramidal Visual Features + 2D BBox Geometry + Cross-Attention Transformer + Sinkhorn",
            tag_left="Baseline F1: 71.3%",
            tag_right="Params: 1,680,449 (1.68M)",
        )

        # Stage 1: Inputs
        col1_x, col1_w = 0.03, 0.18
        g1_bg = FancyBboxPatch((col1_x - 0.006, 0.10), col1_w + 0.012, 0.78,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g1_bg)
        self.ax.text(col1_x + col1_w / 2, 0.85, "STAGE 1: OBJECT INPUTS", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col1_x, 0.52, col1_w, 0.30,
                             title="Region Visual Patches",
                             subtitle="Cropped from View 0 & View 1",
                             bg_color="#064E3B", border_color="#10B981",
                             in_shape="Image (B, 3, H, W)",
                             out_shape="(B, M, 3, 112, 112)",
                             title_color="#A7F3D0")
        self.ax.text(col1_x + col1_w / 2, 0.65, "• RGB Patches resized to 112x112\n• Isolates object appearances\n• Dual view inputs: View0 & View1",
                     color="#D1FAE5", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.5)

        self.draw_module_box(col1_x, 0.15, col1_w, 0.30,
                             title="Bounding Box Layout",
                             subtitle="2D Normalized Coordinates",
                             bg_color="#1E3A8A", border_color="#3B82F6",
                             in_shape="BBoxes [x1, y1, x2, y2]",
                             out_shape="(B, M, 4)",
                             title_color="#BFDBFE")
        self.ax.text(col1_x + col1_w / 2, 0.28, "• Coordinates: [cx/W, cy/H, w/W, h/H]\n• Absolute 2D spatial arrangement",
                     color="#DBEAFE", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.5)

        # Stage 2: Encoders & Fusion
        col2_x, col2_w = 0.26, 0.22
        g2_bg = FancyBboxPatch((col2_x - 0.006, 0.10), col2_w + 0.012, 0.78,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g2_bg)
        self.ax.text(col2_x + col2_w / 2, 0.85, "STAGE 2: VISUAL & LAYOUT ENCODING", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col2_x, 0.52, col2_w, 0.30,
                             title="XFeat FPN Backbone",
                             subtitle="Pyramid Levels P1, P2, P3",
                             bg_color="#062D24", border_color="#059669",
                             in_shape="(B*M, 3, 112, 112)",
                             out_shape="t_vis: (B, M, 192)",
                             title_color="#6EE7B7")
        self.ax.text(col2_x + col2_w / 2, 0.65, "• P1 (56x56) ➔ AvgPool ➔ 64D\n• P2 (28x28) ➔ AvgPool ➔ 64D\n• P3 (14x14) ➔ AvgPool ➔ 64D\n• Concatenated: 192D Feature",
                     color="#A7F3D0", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.5)

        self.draw_module_box(col2_x, 0.15, col2_w, 0.30,
                             title="ObjectGeometryEncoder",
                             subtitle="Layout Mapping MLP",
                             bg_color="#0F224A", border_color="#2563EB",
                             in_shape="(B*M, 4)",
                             out_shape="t_bbox: (B, M, 64)",
                             title_color="#93C5FD")
        self.ax.text(col2_x + col2_w / 2, 0.28, "• Linear(4➔64) ➔ LN ➔ ReLU\n• Linear(64➔64) ➔ LN ➔ ReLU\n• 64D Spatial Layout Embedding",
                     color="#DBEAFE", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.5)

        self.draw_arrow(col1_x + col1_w, 0.67, col2_x, 0.67, color="#10B981", label="(B,M,3,112,112)")
        self.draw_arrow(col1_x + col1_w, 0.30, col2_x, 0.30, color="#3B82F6", label="(B,M,4)")

        # Stage 3: Fusion & Transformer
        col3_x, col3_w = 0.53, 0.20
        g3_bg = FancyBboxPatch((col3_x - 0.006, 0.10), col3_w + 0.012, 0.78,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g3_bg)
        self.ax.text(col3_x + col3_w / 2, 0.85, "STAGE 3: FUSION & GRAPH REASONING", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col3_x, 0.15, col3_w, 0.67,
                             title="Graph Transformer & Fusion",
                             subtitle="Linear Projection + 2 Attention Layers",
                             bg_color="#083344", border_color="#0891B2",
                             in_shape="Concat 256D",
                             out_shape="X_ctx: (B, M, 128)",
                             title_color="#A5F3FC")
        self.ax.text(col3_x + col3_w / 2, 0.48,
                     "1. Fusion Projection:\n   • Concat: [t_vis (192D) || t_bbox (64D)]\n   • Linear(256 ➔ 128) ➔ LayerNorm ➔ ReLU\n   • Unified Object Token (128D)\n\n"
                     "2. Cross-View Graph Transformer:\n   • Intra-View Self-Attention (4 Heads)\n   • Inter-View Cross-Attention (4 Heads)\n   • FFN: 128 ➔ 256 ➔ 128\n   • Residual Add & LayerNorm\n\n"
                     "3. Output Contextual Tokens: (B, M, 128)",
                     color="#CFFAFE", fontsize=8.5, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_arrow(col2_x + col2_w, 0.67, col3_x, 0.55, color="#10B981", label="192D")
        self.draw_arrow(col2_x + col2_w, 0.30, col3_x, 0.42, color="#3B82F6", label="64D")

        # Stage 4: Sinkhorn Matcher
        col4_x, col4_w = 0.78, 0.19
        g4_bg = FancyBboxPatch((col4_x - 0.006, 0.10), col4_w + 0.012, 0.78,
                               boxstyle="round,pad=0.005,rounding_size=0.015",
                               facecolor="#0F172A", edgecolor="#1E293B", linewidth=1.2, zorder=1)
        self.ax.add_patch(g4_bg)
        self.ax.text(col4_x + col4_w / 2, 0.85, "STAGE 4: OPTIMAL TRANSPORT", color="#38BDF8", fontsize=10, fontweight="bold", ha="center", va="center")

        self.draw_module_box(col4_x, 0.15, col4_w, 0.67,
                             title="ObjectSinkhornMatcher",
                             subtitle="Log-Sinkhorn + Dustbin Absorption",
                             bg_color="#450A0A", border_color="#EF4444",
                             in_shape="X0, X1: (B, M, 128)",
                             out_shape="Matches {(i, j)}",
                             title_color="#FCA5A5")
        self.ax.text(col4_x + col4_w / 2, 0.48,
                     "1. Affinity Matrix S_ij = <X0_i, X1_j> / √128\n\n"
                     "2. Dustbin Column/Row Augmentation:\n   Absorbs unmatched/occluded objects\n\n"
                     "3. Sinkhorn Dual Iterations (20 iters):\n   Log-space row & col normalization\n\n"
                     "4. Mutual Nearest Neighbor Filtering:\n   P_ij ≥ 0.20 & MNN mutual check\n\n"
                     "★ Test Benchmark: 71.3% Semantic F1",
                     color="#FEE2E2", fontsize=8.5, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_arrow(col3_x + col3_w, 0.48, col4_x, 0.48, color="#06B6D4", label="128D Tokens")

        self.draw_bottom_footer(
            formula=r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{NLL}}(\bar{P}, Y) + \lambda_{\mathrm{dustbin}} \cdot \mathcal{L}_{\mathrm{dustbin}}$",
            specs="Pure Visual Baseline     |     Inference: 3.6 ms/pair     |     Batch: 32",
        )

        return self.save(out_path)


class MaskLineMatcherVisualizer(BaseArchitectureVisualizer):
    """Visualizes MaskLineFeatMatchNet (6-Channel Continuous Geometry Field)."""

    def render(self, out_path: Path | str) -> Path:
        self.setup_canvas()

        self.draw_top_banner(
            title="MaskLineFeatMatchNet: 6-Channel Geometry Field Line Matching Network",
            subtitle="RGB + Euclidean Distance Transform (EDT) + Angle Field Map (AFM) + ResNet Backbone",
            tag_left="Continuous Geometry Field",
            tag_right="Params: 3,245,184 (3.25M)",
        )

        col1_x, col1_w = 0.04, 0.25
        self.draw_module_box(col1_x, 0.18, col1_w, 0.65,
                             title="6-Channel Continuous Field Input",
                             subtitle="Photometric & Geometric Fusion Map",
                             bg_color="#1E293B", border_color="#3B82F6",
                             in_shape="RGB + Lines",
                             out_shape="Field: (B, 6, H, W)",
                             title_color="#93C5FD")
        self.ax.text(col1_x + col1_w / 2, 0.50,
                     "• Channels 1-3: RGB Color Image (3, H, W)\n\n"
                     "• Channel 4: Euclidean Distance Field (EDT)\n  Distance transform to nearest LSD straight line\n\n"
                     "• Channels 5-6: Angle Field Map (AFM)\n  [sin(2θ), cos(2θ)] orientation vectors\n\n"
                     "• Smooth Continuous Geometric Prior",
                     color="#E2E8F0", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.4)

        col2_x, col2_w = 0.36, 0.26
        self.draw_module_box(col2_x, 0.18, col2_w, 0.65,
                             title="Conv6ch ResNet Backbone",
                             subtitle="Dense Feature Extraction & Line Sampling",
                             bg_color="#062D24", border_color="#059669",
                             in_shape="(B, 6, H, W)",
                             out_shape="F: (B, 128, H/4, W/4)",
                             title_color="#6EE7B7")
        self.ax.text(col2_x + col2_w / 2, 0.50,
                     "1. 6-Channel First Convolution:\n   Conv2d(6 ➔ 64, kernel=7, stride=2)\n\n"
                     "2. ResNet Layer Blocks (Residual 128D)\n\n"
                     "3. Sub-Pixel Line Feature Sampling:\n   Bilinear interpolation along LSD endpoints\n\n"
                     "4. Line Descriptor Extraction: (B, N, 128)",
                     color="#A7F3D0", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_arrow(col1_x + col1_w, 0.50, col2_x, 0.50, color="#3B82F6", label="(B, 6, H, W)")

        col3_x, col3_w = 0.69, 0.26
        self.draw_module_box(col3_x, 0.18, col3_w, 0.65,
                             title="Dual-Softmax MNN Matcher",
                             subtitle="Optimal Mutual Correspondence",
                             bg_color="#450A0A", border_color="#EF4444",
                             in_shape="Line Descriptors (128D)",
                             out_shape="Matches {(i, j)}",
                             title_color="#FCA5A5")
        self.ax.text(col3_x + col3_w / 2, 0.50,
                     "1. Similarity Matrix: S = Descs0 · Descs1^T / τ\n\n"
                     "2. Dual-Softmax Normalization:\n   P = Softmax(S, dim=1) * Softmax(S, dim=0)\n\n"
                     "3. Mutual Nearest Neighbor Check:\n   Mutual maximum probability pairs\n\n"
                     "4. Robustness against low-texture & illumination",
                     color="#FEE2E2", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_arrow(col2_x + col2_w, 0.50, col3_x, 0.50, color="#10B981", label="(B, N, 128)")

        self.draw_bottom_footer(
            formula=r"$\mathcal{L}_{\mathrm{match}} = -\sum \log P_{i, j} + \lambda_{\mathrm{unmatched}} \mathcal{L}_{\mathrm{bg}}$",
            specs="6-Channel Geometric Field     |     Inference: 6.2 ms/pair     |     Resolution: 640x480",
        )

        return self.save(out_path)


class LineTransformerVisualizer(BaseArchitectureVisualizer):
    """Visualizes LineMatcherNet (Baseline Line Transformer)."""

    def render(self, out_path: Path | str) -> Path:
        self.setup_canvas()

        self.draw_top_banner(
            title="LineMatcherNet: Baseline Line Transformer Matching Network",
            subtitle="Endpoints 2D Coordinate MLP + Multi-Head Self/Cross Attention + Dual-Softmax",
            tag_left="Baseline Line Matcher",
            tag_right="Params: 1,412,096 (1.41M)",
        )

        col1_x, col1_w = 0.04, 0.25
        self.draw_module_box(col1_x, 0.18, col1_w, 0.65,
                             title="Sub-Pixel LSD Endpoints",
                             subtitle="Geometric 2D Line Segments",
                             bg_color="#451A03", border_color="#F59E0B",
                             in_shape="LSD Lines (B, N, 4)",
                             out_shape="(B, N, 4)",
                             title_color="#FDE68A")
        self.ax.text(col1_x + col1_w / 2, 0.50,
                     "• Input: Line segments [x1, y1, x2, y2]\n\n"
                     "• Sub-pixel line segment detector (LSD)\n\n"
                     "• Normalized coordinate frame [0, 1]\n\n"
                     "• N: Detected lines in view",
                     color="#FEF3C7", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.4)

        col2_x, col2_w = 0.36, 0.26
        self.draw_module_box(col2_x, 0.18, col2_w, 0.65,
                             title="LineTransformer Backbone",
                             subtitle="MLP Encoder + Self/Cross Attention",
                             bg_color="#083344", border_color="#0891B2",
                             in_shape="(B, N, 4)",
                             out_shape="Tokens: (B, N, 128)",
                             title_color="#A5F3FC")
        self.ax.text(col2_x + col2_w / 2, 0.50,
                     "1. Coordinate MLP:\n   Linear(4 ➔ 128) ➔ LayerNorm ➔ ReLU\n\n"
                     "2. Self-Attention (4 Layers, 4 Heads):\n   Captures intra-image geometric co-linearity\n\n"
                     "3. Cross-Attention (4 Layers, 4 Heads):\n   Transfers matching cues across views\n\n"
                     "4. Feed-Forward Network: 128 ➔ 256 ➔ 128",
                     color="#CFFAFE", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_arrow(col1_x + col1_w, 0.50, col2_x, 0.50, color="#F59E0B", label="(B, N, 4)")

        col3_x, col3_w = 0.69, 0.26
        self.draw_module_box(col3_x, 0.18, col3_w, 0.65,
                             title="Dual-Softmax Matcher",
                             subtitle="Optimal Mutual Assignment",
                             bg_color="#450A0A", border_color="#EF4444",
                             in_shape="Tokens (128D)",
                             out_shape="Matches {(i, j)}",
                             title_color="#FCA5A5")
        self.ax.text(col3_x + col3_w / 2, 0.50,
                     "1. Score Matrix: S_ij = <T0_i, T1_j> / √128\n\n"
                     "2. Bidirectional Softmax Normalization\n\n"
                     "3. Mutual Nearest Neighbor Filter (MNN)\n\n"
                     "4. Thresholding: P_ij ≥ 0.20",
                     color="#FEE2E2", fontsize=9, ha="center", va="center", zorder=5, linespacing=1.4)

        self.draw_arrow(col2_x + col2_w, 0.50, col3_x, 0.50, color="#06B6D4", label="(B, N, 128)")

        self.draw_bottom_footer(
            formula=r"$\mathcal{L}_{\mathrm{match}} = -\sum \log P_{i, j}$",
            specs="Pure Geometric Transformer     |     Inference: 3.2 ms/pair     |     Batch: 32",
        )

        return self.save(out_path)


# Registry of Architecture Visualizers
VISUALIZER_REGISTRY = {
    "xfeat_line_matcher": XFeatLineMatcherVisualizer,
    "xfeat_line": XFeatLineMatcherVisualizer,
    "line_object": XFeatLineMatcherVisualizer,
    "xfeat_object": XFeatObjectMatcherVisualizer,
    "xfeat": XFeatObjectMatcherVisualizer,
    "object": XFeatObjectMatcherVisualizer,
    "mask_line": MaskLineMatcherVisualizer,
    "mask": MaskLineMatcherVisualizer,
    "line_transformer": LineTransformerVisualizer,
    "linefeat": LineTransformerVisualizer,
}


def render_model_architecture(model_name: str, out_path: Optional[Path | str] = None, dpi: int = 180) -> Path:
    """
    Renders and saves a publication-quality architecture diagram for any supported model.
    If out_path is omitted, automatically saves to 'web/static/img/<model_name>_architecture.png'.
    """
    name = model_name.lower()
    if name not in VISUALIZER_REGISTRY:
        name = "xfeat_line_matcher"

    visualizer_cls = VISUALIZER_REGISTRY[name]
    vis = visualizer_cls(dpi=dpi)

    if out_path is None:
        static_dir = Path(__file__).resolve().parent.parent / "web" / "static" / "img"
        canonical_name = "xfeat_line_matcher" if "line_matcher" in name or "xfeat_line" in name else (
            "xfeat_object" if "object" in name else ("mask_line" if "mask" in name else "line_transformer")
        )
        out_path = static_dir / f"{canonical_name}_architecture.png"

    return vis.render(out_path)


def render_all_models(output_dir: Optional[Path | str] = None) -> dict[str, Path]:
    """Renders all model architecture diagrams into the specified directory."""
    models = ["xfeat_line_matcher", "xfeat_object", "mask_line", "line_transformer"]
    results = {}
    for m in models:
        path = render_model_architecture(m, out_path=None if output_dir is None else Path(output_dir) / f"{m}_architecture.png")
        results[m] = path
    return results


if __name__ == "__main__":
    rendered = render_all_models()
    for k, p in rendered.items():
        print(f"[✓] Successfully generated architecture diagram: {k} -> {p}")
