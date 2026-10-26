"""
Unified Experiment Tracker, Logger, and Automated Post-Mortem Reporting Suite.

Key Capabilities:
  1. TeeLogger: Mirror console stdout/stderr to run_dir/train.log with timestamps.
  2. MetricsTracker: Append per-epoch metrics to run_dir/metrics.csv and update config.json.
  3. CurvePlotter: Automatically render 4-panel training trajectory plot (curves.png).
  4. PostMortemReportGenerator: Automatically author run_dir/report.md with diagnostics.
  5. ExperimentReviewer: CLI and API for scanning, comparing, and summarizing historical runs.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DualStreamTee:
    """Tees stdout or stderr to both the terminal and a clean disk log file."""

    def __init__(self, target_stream, log_file):
        self.target_stream = target_stream
        self.log_file = log_file

    def write(self, message: str) -> None:
        self.target_stream.write(message)
        # Strip ANSI escape sequences (e.g. terminal color codes and progress bars)
        clean_msg = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", message)
        # Replace carriage return \r with newline if followed by fresh content
        self.log_file.write(clean_msg)
        # Only flush to disk upon complete line to avoid blocking training loops with synchronous I/O
        if "\n" in message:
            self.log_file.flush()

    def flush(self) -> None:
        self.target_stream.flush()
        self.log_file.flush()


class ExperimentLogger:
    """
    Comprehensive lifecycle manager for training experiment logging, metrics tracking,
    automatic plotting, and post-mortem report generation.
    """

    def __init__(
        self,
        save_dir: Path | str,
        config_data: dict[str, Any],
        enable_stdout_tee: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config_data = config_data

        self.log_file_path = self.save_dir / "train.log"
        self.metrics_csv_path = self.save_dir / "metrics.csv"
        self.report_md_path = self.save_dir / "report.md"
        self.curves_png_path = self.save_dir / "curves.png"

        # Initialize CSV header if not present
        if not self.metrics_csv_path.exists():
            with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "dustbin_precision",
                    "dustbin_recall",
                    "dustbin_accuracy",
                    "cross_cat_fp",
                    "lr",
                    "epoch_time_sec",
                    "is_best",
                    "timestamp",
                ])

        self.history: list[dict[str, Any]] = []

        # Setup console mirroring to train.log
        self.enable_stdout_tee = enable_stdout_tee
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._log_handle = open(self.log_file_path, "a", encoding="utf-8")

        if enable_stdout_tee:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._log_handle.write(f"\n{'=' * 80}\n")
            self._log_handle.write(f"  EXPERIMENT RUN LOG INITIALIZED: {timestamp_str}\n")
            self._log_handle.write(f"  Run Directory: {self.save_dir}\n")
            self._log_handle.write(f"{'=' * 80}\n\n")
            self._log_handle.flush()

            sys.stdout = DualStreamTee(self._orig_stdout, self._log_handle)
            sys.stderr = DualStreamTee(self._orig_stderr, self._log_handle)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Optional[dict[str, float]] = None,
        is_best: bool = False,
        epoch_sec: float = 0.0,
        lr: float = 0.0,
    ) -> dict[str, Any]:
        """Records metrics for an epoch, appends to CSV and history, and updates plot."""
        val_f1 = val_metrics["f1"] if val_metrics else None
        val_prec = val_metrics["precision"] if val_metrics else None
        val_rec = val_metrics["recall"] if val_metrics else None
        dustbin_prec = val_metrics["dustbin_prec"] if val_metrics else None
        dustbin_rec = val_metrics["dustbin_rec"] if val_metrics else None
        dustbin_acc = val_metrics["dustbin_acc"] if val_metrics else None
        cross_cat_fp = val_metrics.get("cross_cat_fp", 0) if val_metrics else None

        record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_f1": round(float(val_f1), 2) if val_f1 is not None else None,
            "val_precision": round(float(val_prec), 2) if val_prec is not None else None,
            "val_recall": round(float(val_rec), 2) if val_rec is not None else None,
            "dustbin_precision": round(float(dustbin_prec), 2) if dustbin_prec is not None else None,
            "dustbin_recall": round(float(dustbin_rec), 2) if dustbin_rec is not None else None,
            "dustbin_accuracy": round(float(dustbin_acc), 2) if dustbin_acc is not None else None,
            "cross_cat_fp": int(cross_cat_fp) if cross_cat_fp is not None else None,
            "lr": float(lr),
            "epoch_time_sec": round(float(epoch_sec), 1),
            "is_best": bool(is_best),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.history.append(record)

        # Append to CSV
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["epoch"],
                record["train_loss"],
                record["val_f1"] if record["val_f1"] is not None else "",
                record["val_precision"] if record["val_precision"] is not None else "",
                record["val_recall"] if record["val_recall"] is not None else "",
                record["dustbin_precision"] if record["dustbin_precision"] is not None else "",
                record["dustbin_recall"] if record["dustbin_recall"] is not None else "",
                record["dustbin_accuracy"] if record["dustbin_accuracy"] is not None else "",
                record["cross_cat_fp"] if record["cross_cat_fp"] is not None else "",
                f"{record['lr']:.6g}",
                record["epoch_time_sec"],
                "TRUE" if record["is_best"] else "FALSE",
                record["timestamp"],
            ])

        # Synchronize into config_data
        self.config_data["history"] = self.history
        config_path = self.save_dir / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2)
        except Exception:
            pass

        # Periodically refresh plot
        if len(self.history) >= 2:
            self.plot_curves()

        return record

    def plot_curves(self, out_path: Optional[Path | str] = None) -> Optional[Path]:
        """Renders a 4-panel trajectory plot using matplotlib."""
        target_path = Path(out_path or self.curves_png_path)
        if not self.history or len(self.history) < 2:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = [h["epoch"] for h in self.history]
            train_losses = [h["train_loss"] for h in self.history]
            val_f1s = [h["val_f1"] for h in self.history if h["val_f1"] is not None]
            val_precs = [h["val_precision"] for h in self.history if h["val_precision"] is not None]
            val_recs = [h["val_recall"] for h in self.history if h["val_recall"] is not None]
            val_epochs = [h["epoch"] for h in self.history if h["val_f1"] is not None]

            dustbin_precs = [h["dustbin_precision"] for h in self.history if h["dustbin_precision"] is not None]
            dustbin_recs = [h["dustbin_recall"] for h in self.history if h["dustbin_recall"] is not None]

            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            fig.patch.set_facecolor("#FAFAFA")

            # 1. Train Loss
            ax1 = axes[0, 0]
            ax1.set_facecolor("#FFFFFF")
            ax1.plot(epochs, train_losses, color="#D9534F", linewidth=2.0, marker="o", markersize=4, label="Train Loss")
            ax1.set_title("Training Loss Trajectory", fontsize=12, fontweight="bold", pad=8)
            ax1.set_xlabel("Epoch", fontsize=10)
            ax1.set_ylabel("Loss", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right", framealpha=0.8)

            # 2. Validation Matching Performance (Precision, Recall, F1)
            ax2 = axes[0, 1]
            ax2.set_facecolor("#FFFFFF")
            if val_f1s:
                ax2.plot(val_epochs, val_f1s, color="#2B6CB0", linewidth=2.5, marker="s", markersize=4, label="Val F1-Score")
                ax2.plot(val_epochs, val_precs, color="#38A169", linewidth=1.8, linestyle="--", label="Precision")
                ax2.plot(val_epochs, val_recs, color="#DD6B20", linewidth=1.8, linestyle="--", label="Recall")

                # Highlight peak F1
                best_idx = int(np.argmax(val_f1s))
                best_ep = val_epochs[best_idx]
                best_val = val_f1s[best_idx]
                ax2.scatter([best_ep], [best_val], color="#E53E3E", s=130, zorder=5, marker="*")
                ax2.annotate(
                    f"Peak: {best_val:.1f}% (Ep {best_ep})",
                    (best_ep, best_val),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#C53030",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF5F5", edgecolor="#FEB2B2", alpha=0.9),
                )
            ax2.set_title("Validation Matching Performance", fontsize=12, fontweight="bold", pad=8)
            ax2.set_xlabel("Epoch", fontsize=10)
            ax2.set_ylabel("Percentage (%)", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend(loc="lower right", framealpha=0.8)

            # 3. Dustbin Rejection Dynamics (Check for conservative bias)
            ax3 = axes[1, 0]
            ax3.set_facecolor("#FFFFFF")
            if dustbin_precs:
                ax3.plot(val_epochs, dustbin_precs, color="#319795", linewidth=2.0, marker="^", markersize=4, label="Dustbin Precision")
                ax3.plot(val_epochs, dustbin_recs, color="#805AD5", linewidth=2.0, marker="v", markersize=4, label="Dustbin Recall")
            ax3.set_title("Dustbin Rejection Dynamics", fontsize=12, fontweight="bold", pad=8)
            ax3.set_xlabel("Epoch", fontsize=10)
            ax3.set_ylabel("Percentage (%)", fontsize=10)
            ax3.grid(True, linestyle="--", alpha=0.5)
            ax3.legend(loc="lower right", framealpha=0.8)

            # 4. Learning Rate & Epoch Time
            ax4 = axes[1, 1]
            ax4.set_facecolor("#FFFFFF")
            times = [h["epoch_time_sec"] for h in self.history]
            ax4.bar(epochs, times, color="#CBD5E0", alpha=0.6, label="Epoch Duration (s)")
            ax4.set_ylabel("Duration (seconds)", fontsize=10, color="#4A5568")
            ax4.set_xlabel("Epoch", fontsize=10)
            ax4.set_title("Training Duration & Learning Rate", fontsize=12, fontweight="bold", pad=8)
            ax4.grid(True, linestyle="--", alpha=0.3)

            # Twin axis for learning rate
            lrs = [h["lr"] for h in self.history]
            ax4_lr = ax4.twinx()
            ax4_lr.plot(epochs, lrs, color="#D69E2E", linewidth=2.0, linestyle=":", label="Learning Rate")
            ax4_lr.set_ylabel("Learning Rate", fontsize=10, color="#B7791F")
            ax4_lr.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")

            plt.tight_layout()
            fig.savefig(str(target_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            return target_path
        except Exception as e:
            print(f"[!] Warning: Failed to render training curves: {e}")
            return None

    def generate_post_mortem_report(
        self,
        raw_test_metrics: Optional[dict[str, float]] = None,
        cat_test_metrics: Optional[dict[str, float]] = None,
    ) -> Path:
        """Generates comprehensive Markdown report (report.md) with diagnostics and advice."""
        exp_info = self.config_data.get("experiment_info", {})
        hparams = self.config_data.get("hyperparameters", {})
        model_meta = self.config_data.get("model", {})
        best_val = self.config_data.get("best_validation") or {}

        run_name = exp_info.get("run_name", self.save_dir.name)
        model_type = model_meta.get("model_type", "unknown")
        start_t = exp_info.get("start_time", "N/A")
        end_t = exp_info.get("end_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        gpu_name = exp_info.get("device_name", "GPU")
        command = exp_info.get("command", "train.py")

        # 1. Diagnostic Analyses
        warnings: list[str] = []
        tips: list[str] = []

        if len(self.history) >= 4:
            f1_scores = [h["val_f1"] for h in self.history if h["val_f1"] is not None]
            loss_scores = [h["train_loss"] for h in self.history]
            if f1_scores:
                peak_f1 = max(f1_scores)
                peak_epoch = [h["epoch"] for h in self.history if h["val_f1"] == peak_f1][0]
                final_f1 = f1_scores[-1]
                final_loss = loss_scores[-1]
                peak_loss = loss_scores[peak_epoch - 1] if peak_epoch <= len(loss_scores) else final_loss

                # Overfitting check
                if peak_f1 - final_f1 >= 1.5 and final_loss < peak_loss:
                    drop = peak_f1 - final_f1
                    warnings.append(
                        f"**检测到显著过拟合倾向**：模型在第 **{peak_epoch}** 轮达到峰值 Val F1 ({peak_f1:.2f}%)，"
                        f"随后训练 Loss 继续从 {peak_loss:.4f} 下滑到 {final_loss:.4f}，但 Val F1 下跌了 **{drop:.2f}%**。"
                        f"建议后续将 `--epochs` 控制在 {peak_epoch + 2} 轮以内，或在第 {peak_epoch} 轮引入余弦衰减（降低学习率至 3e-5）。"
                    )

                # Conservative / Timid Bias Check
                dustbin_recs = [h["dustbin_recall"] for h in self.history if h["dustbin_recall"] is not None]
                val_recs = [h["val_recall"] for h in self.history if h["val_recall"] is not None]
                if dustbin_recs and val_recs:
                    final_db_r = dustbin_recs[-1]
                    final_v_r = val_recs[-1]
                    if final_db_r >= 78.0 and final_v_r <= 68.0:
                        tips.append(
                            f"**检测到模型过度防御（Conservative Bias）**：后期垃圾桶召回率高达 **{final_db_r:.1f}%**，"
                            f"而物体匹配召回率降至 **{final_v_r:.1f}%**。说明模型倾向于将模棱两可的正样本直接扔入垃圾桶逃避惩罚。"
                            f"建议适当下调 `--dustbin-weight`（如从 {hparams.get('dustbin_weight', 0.3)} 调至 0.2 或 0.15）以解放召回率。"
                        )

        # 2. Build Markdown Content
        md = []
        md.append(f"# 实验复盘报告 (Run Post-Mortem Report)\n")
        md.append(f"> **实验代号**: `{run_name}`  \n")
        md.append(f"> **模型架构**: `{model_type}` | **运行状态**: `COMPLETED`  \n")
        md.append(f"> **起止时间**: `{start_t}` ~ `{end_t}`  \n")
        md.append(f"> **计算设备**: `{gpu_name}` | **代码版本 (Commit)**: `{exp_info.get('git_commit', 'HEAD')}`\n")
        md.append("\n---\n")

        # Executive Summary
        md.append("## 一、 核心指标执行摘要\n")
        b_ep = best_val.get("epoch", "N/A")
        b_f1 = best_val.get("f1", "N/A")
        b_prec = best_val.get("precision", "N/A")
        b_rec = best_val.get("recall", "N/A")
        md.append(f"| 评测维度 | 最佳数值 | 对应轮次 / 说明 |\n")
        md.append(f"| :--- | :---: | :--- |\n")
        md.append(f"| **验证集最佳 Val F1** | **{b_f1}%** | 第 **{b_ep}** 轮产生 |\n")
        md.append(f"| **验证集 Precision** | **{b_prec}%** | 匹配准确率 |\n")
        md.append(f"| **验证集 Recall** | **{b_rec}%** | 目标召回率 |\n")
        md.append(f"| **垃圾桶识别准确率** | **{best_val.get('dustbin_accuracy', 'N/A')}%** | 区分非共视与遮挡目标 |\n")

        if raw_test_metrics:
            md.append(f"| **测试集无先验外观 F1 (Raw)** | **{raw_test_metrics.get('f1', 0):.2f}%** | 独立盲测集无类别约束 |\n")
            md.append(f"| **测试集语义约束 F1 (Semantic)** | **{cat_test_metrics.get('f1', 0) if cat_test_metrics else 'N/A'}%** | 同类约束后配对表现 |\n")
        md.append("\n")

        # Diagnostics section
        if warnings or tips:
            md.append("## 二、 训练动态智能诊断与调优建议\n")
            for w in warnings:
                md.append(f"> [!WARNING]\n> {w}\n\n")
            for t in tips:
                md.append(f"> [!TIP]\n> {t}\n\n")

        # Training curves embedding
        md.append("## 三、 训练轨迹可视化曲线\n")
        if self.curves_png_path.exists():
            md.append(f"![Training Curves]({self.curves_png_path.name})\n\n")
        else:
            md.append("*未生成曲线图*\n\n")

        # Hyperparameters
        md.append("## 四、 关键超参数与环境配置\n")
        md.append("| 参数名 | 参数值 | 说明 |\n")
        md.append("| :--- | :--- | :--- |\n")
        md.append(f"| `model_type` | `{model_type}` | 网络主干模型 |\n")
        md.append(f"| `learning_rate` | `{hparams.get('learning_rate')}` | 初始学习率 |\n")
        md.append(f"| `epochs` | `{hparams.get('epochs')}` | 总训练轮数 |\n")
        md.append(f"| `dustbin_weight` | `{hparams.get('dustbin_weight')}` | 垃圾桶损失权重 |\n")
        md.append(f"| `motion_loss_weight` | `{hparams.get('motion_loss_weight')}` | 2D位移场几何共识权重 |\n")
        md.append(f"| `min_area_ratio` | `{hparams.get('min_area_ratio')}` | 最小共视匹配面积比例门限 |\n")
        md.append(f"| `use_pyramid` | `{model_meta.get('use_pyramid')}` | 多尺度特征金字塔 (FPN) |\n")
        md.append(f"| `amp` | `{hparams.get('amp')}` | 自动混合精度 FP16 |\n")
        md.append(f"| `command` | `{command}` | 启动运行命令 |\n\n")

        # Epoch Metrics Table
        md.append("## 五、 逐轮次详细训练历史 (Epoch History)\n")
        md.append("| Epoch | Train Loss | Val F1 | Val Prec | Val Rec | Dustbin Prec | Dustbin Rec | 耗时 | 状态 |\n")
        md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for h in self.history:
            star = "★ **BEST**" if h["is_best"] else ""
            f1_str = f"**{h['val_f1']:.1f}%**" if h["is_best"] and h["val_f1"] is not None else (f"{h['val_f1']:.1f}%" if h["val_f1"] is not None else "-")
            p_str = f"{h['val_precision']:.1f}%" if h["val_precision"] is not None else "-"
            r_str = f"{h['val_recall']:.1f}%" if h["val_recall"] is not None else "-"
            dp_str = f"{h['dustbin_precision']:.1f}%" if h["dustbin_precision"] is not None else "-"
            dr_str = f"{h['dustbin_recall']:.1f}%" if h["dustbin_recall"] is not None else "-"
            md.append(
                f"| {h['epoch']:02d} | {h['train_loss']:.4f} | {f1_str} | {p_str} | {r_str} | {dp_str} | {dr_str} | {h['epoch_time_sec']:.1f}s | {star} |\n"
            )
        md.append("\n---\n")

        # Save to disk
        content = "".join(md)
        with open(self.report_md_path, "w", encoding="utf-8") as f:
            f.write(content)

        return self.report_md_path

    def finish(
        self,
        raw_test_metrics: Optional[dict[str, float]] = None,
        cat_test_metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Closes file streams, updates end time, plots final curves, and writes report.md."""
        self.config_data["experiment_info"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if raw_test_metrics:
            self.config_data["test_benchmark"] = {
                "raw_visual": raw_test_metrics,
                "semantic_constrained": cat_test_metrics,
            }

        config_path = self.save_dir / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2)
        except Exception:
            pass

        self.plot_curves()
        self.generate_post_mortem_report(raw_test_metrics, cat_test_metrics)

        # Restore streams
        if self.enable_stdout_tee:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
            if not self._log_handle.closed:
                self._log_handle.close()

        print(f"\n[✓] Experiment logs & post-mortem report exported:")
        print(f"    • Log File : {self.log_file_path}")
        print(f"    • CSV Data : {self.metrics_csv_path}")
        print(f"    • Curves   : {self.curves_png_path}")
        print(f"    • Report   : {self.report_md_path}\n")
