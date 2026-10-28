"""
Training and Experiment Evaluation Curve Visualizer.
Generates polished metric plots:
1. Training trajectory plots (Loss, Val F1, Precision, Recall, Dustbin accuracy).
2. Cross-run comparison curves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"


def plot_training_trajectory(
    history: list[dict[str, Any]],
    out_path: Path | str,
    title: str = "Training Trajectory",
    dpi: int = 150,
) -> Path:
    """Renders dual-axis or multi-panel training trajectory curves."""
    if not history:
        raise ValueError("Cannot plot empty training history")

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    train_loss = [h.get("train_loss", h.get("loss", 0.0)) for h in history]
    val_f1 = [h.get("val_f1", 0.0) for h in history]
    val_prec = [h.get("val_prec", 0.0) for h in history]
    val_rec = [h.get("val_rec", 0.0) for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5.5), facecolor="#0B0F19", dpi=dpi)
    ax1.set_facecolor("#111827")

    # Plot Loss on Left Axis
    color_loss = "#EF4444"
    ax1.set_xlabel("Epoch", color="#9CA3AF", fontsize=11, labelpad=8)
    ax1.set_ylabel("Training Loss", color=color_loss, fontsize=11, labelpad=8)
    line1 = ax1.plot(epochs, train_loss, color=color_loss, marker="o", markersize=4, lw=2.2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.tick_params(axis="x", labelcolor="#9CA3AF")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#4B5563")

    # Plot F1, Precision, Recall on Right Axis
    ax2 = ax1.twinx()
    color_f1 = "#3B82F6"
    color_prec = "#10B981"
    color_rec = "#F59E0B"
    ax2.set_ylabel("Validation Metrics (%)", color=color_f1, fontsize=11, labelpad=8)
    line2 = ax2.plot(epochs, val_f1, color=color_f1, marker="s", markersize=4.5, lw=2.5, label="Val F1 (%)")
    line3 = ax2.plot(epochs, val_prec, color=color_prec, linestyle=":", lw=1.5, alpha=0.8, label="Precision (%)")
    line4 = ax2.plot(epochs, val_rec, color=color_rec, linestyle=":", lw=1.5, alpha=0.8, label="Recall (%)")
    ax2.tick_params(axis="y", labelcolor=color_f1)
    ax2.set_ylim(0, 105)

    # Highlight Best Epoch
    best_idx = int(max(range(len(val_f1)), key=lambda i: val_f1[i]))
    best_epoch = epochs[best_idx]
    best_score = val_f1[best_idx]
    ax2.scatter([best_epoch], [best_score], color="#FBBF24", s=120, zorder=6, edgecolors="#FFF", linewidth=1.5)
    ax2.annotate(
        f"Best F1: {best_score:.1f}%\n(Ep {best_epoch})",
        xy=(best_epoch, best_score),
        xytext=(best_epoch + 0.3, best_score - 12),
        arrowprops=dict(facecolor="#FBBF24", shrink=0.08, width=1.5, headwidth=6),
        color="#FDE68A",
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1F2937", edgecolor="#FBBF24", alpha=0.9),
    )

    # Unified Legend
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4,
                     facecolor="#1F2937", edgecolor="#374151", labelcolor="#E5E7EB", fontsize=9.5)

    plt.title(title, color="#F9FAFB", fontsize=13, fontweight="bold", pad=28)
    fig.tight_layout()

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=dpi, facecolor="#0B0F19", edgecolor="none")
    plt.close(fig)
    return out_p
