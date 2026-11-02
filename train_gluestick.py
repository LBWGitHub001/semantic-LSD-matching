"""GlueStick 与语义增强 GlueStick 统一训练主程序 (GlueStick Training Pipeline).

支持:
1. 原版几何 GlueStick 微调训练 (Point + Wireframe Line);
2. SAM ViT-H 语义融合增强版 (SemanticGlueStick) 训练;
3. 自动场景隔离划分 Train / Val 集 (零数据泄露);
4. 混合精度 AMP 加速与梯度裁剪 (适应 8GB RTX 4060 显存);
5. 自动记录评估指标 (Point/Line Precision, Recall, F1) 与最佳权重保存。
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
gluestick_cands = [ROOT / "third_party" / "GlueStick", ROOT / "GlueStick"]
GLUESTICK_DIR = next((p for p in gluestick_cands if p.is_dir()), gluestick_cands[0])
if str(GLUESTICK_DIR) not in sys.path:
    sys.path.insert(0, str(GLUESTICK_DIR))

from datasets.semantic_gluestick_dataset import SemanticGlueStickDataset, semantic_gluestick_collate_fn
from models.gluestick_wrapper import GlueStickLoss, SemanticGlueStick, compute_matching_metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
    clip_grad: float = 10.0,
) -> dict[str, float]:
    """训练单轮 Epoch。"""
    model.train()
    total_loss = 0.0
    total_pt_loss = 0.0
    total_ln_loss = 0.0
    total_pt_prec, total_pt_rec = 0.0, 0.0
    total_ln_prec, total_ln_rec = 0.0, 0.0
    steps = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        # 将数据送入 GPU
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device, non_blocking=True)
            else:
                batch_gpu[k] = v

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(batch_gpu)
            loss, stats = loss_fn(pred, batch_gpu)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  [!] 警告: 检测到 NaN/Inf Loss，跳过本步")
            continue

        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        scaler.step(optimizer)
        scaler.update()

        # 计算实时指标
        with torch.no_grad():
            metrics = compute_matching_metrics(pred, batch_gpu)

        total_loss += stats["total_loss"]
        total_pt_loss += stats["point_loss"]
        total_ln_loss += stats["line_loss"]
        total_pt_prec += metrics["point_precision"]
        total_pt_rec += metrics["point_recall"]
        total_ln_prec += metrics["line_precision"]
        total_ln_rec += metrics["line_recall"]
        steps += 1

        pbar.set_postfix({
            "Loss": f"{stats['total_loss']:.2f}",
            "PtL": f"{stats['point_loss']:.2f}",
            "LnL": f"{stats['line_loss']:.2f}",
            "PtP": f"{metrics['point_precision']:.2f}",
            "LnP": f"{metrics['line_precision']:.2f}",
        })

    n = max(steps, 1)
    return {
        "train_loss": total_loss / n,
        "train_point_loss": total_pt_loss / n,
        "train_line_loss": total_ln_loss / n,
        "train_point_precision": total_pt_prec / n,
        "train_point_recall": total_pt_rec / n,
        "train_line_precision": total_ln_prec / n,
        "train_line_recall": total_ln_rec / n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    max_eval_steps: int | None = None,
) -> dict[str, float]:
    """验证集全量评测。"""
    model.eval()
    total_loss = 0.0
    total_pt_loss = 0.0
    total_ln_loss = 0.0
    total_pt_prec, total_pt_rec = 0.0, 0.0
    total_ln_prec, total_ln_rec = 0.0, 0.0
    steps = 0

    pbar = tqdm(loader, desc="  Val", leave=False)
    for idx, batch in enumerate(pbar):
        if max_eval_steps and idx >= max_eval_steps:
            break

        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_gpu[k] = v.to(device, non_blocking=True)
            else:
                batch_gpu[k] = v

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(batch_gpu)
            loss, stats = loss_fn(pred, batch_gpu)

        metrics = compute_matching_metrics(pred, batch_gpu)

        total_loss += stats["total_loss"]
        total_pt_loss += stats["point_loss"]
        total_ln_loss += stats["line_loss"]
        total_pt_prec += metrics["point_precision"]
        total_pt_rec += metrics["point_recall"]
        total_ln_prec += metrics["line_precision"]
        total_ln_rec += metrics["line_recall"]
        steps += 1

    n = max(steps, 1)
    pt_p = total_pt_prec / n
    pt_r = total_pt_rec / n
    ln_p = total_ln_prec / n
    ln_r = total_ln_rec / n

    pt_f1 = 2 * pt_p * pt_r / max(pt_p + pt_r, 1e-6)
    ln_f1 = 2 * ln_p * ln_r / max(ln_p + ln_r, 1e-6)

    return {
        "val_loss": total_loss / n,
        "val_point_loss": total_pt_loss / n,
        "val_line_loss": total_ln_loss / n,
        "val_point_precision": pt_p,
        "val_point_recall": pt_r,
        "val_point_f1": pt_f1,
        "val_line_precision": ln_p,
        "val_line_recall": ln_r,
        "val_line_f1": ln_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="GlueStick 几何与语义融合训练管道")
    parser.add_argument("--data-dir", default="data/scannet_semantic_gluestick", help="训练数据集根目录")
    default_weights = ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
    if not default_weights.is_file():
        default_weights = ROOT / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
    parser.add_argument("--weights", default=str(default_weights), help="预训练模型权重路径")
    parser.add_argument("--use-semantics", action="store_true", help="启用 SAM ViT-H 语义融合增强分支")
    parser.add_argument("--freeze-base", action="store_true", help="冻结 GlueStick 主干，仅训练语义适配层")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小 (GlueStick 变长图推荐 1 或 2)")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率 (微调推荐 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--point-weight", type=float, default=1.0, help="特征点匹配损失权重")
    parser.add_argument("--line-weight", type=float, default=1.0, help="线段匹配损失权重")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集场景比例 (默认 15%)")
    parser.add_argument("--save-dir", default="checkpoints/gluestick_train", help="模型检查点保存目录")
    parser.add_argument("--val-interval", type=int, default=1, help="每隔几个 Epoch 进行一次验证")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--max-train-samples", type=int, default=None, help="最多使用的训练样本数 (供快速冒烟测试)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="最多使用的验证样本数 (供快速冒烟测试)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 75)
    print("  🚀 启动 GlueStick 深度匹配模型训练管道 (GlueStick Training Pipeline)")
    print("=" * 75)
    print(f"[*] 数据集路径: {args.data_dir}")
    print(f"[*] 语义增强模式 (use_semantics): {args.use_semantics}")
    print(f"[*] 预训练权重: {args.weights}")
    print(f"[*] 训练参数: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, Device={device}")
    print(f"[*] 检查点保存路径: {save_dir}")
    print("-" * 75)

    # 1. 构建 Train / Val 数据集与加载器
    train_ds = SemanticGlueStickDataset(args.data_dir, split="train", val_ratio=args.val_ratio)
    val_ds = SemanticGlueStickDataset(args.data_dir, split="val", val_ratio=args.val_ratio)

    if args.max_train_samples:
        train_ds.sample_names = train_ds.sample_names[:args.max_train_samples]
    if args.max_val_samples:
        val_ds.sample_names = val_ds.sample_names[:args.max_val_samples]

    print(f"[✓] 数据集隔离装载完成: 训练集 {len(train_ds)} 样本, 验证集 {len(val_ds)} 样本")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=semantic_gluestick_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=semantic_gluestick_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 2. 构建模型
    model = SemanticGlueStick(
        weights=args.weights,
        use_semantics=args.use_semantics,
    ).to(device)

    if args.freeze_base and args.use_semantics:
        print("[*] 冻结 GlueStick 主干参数，仅优化 SAM 语义适配器与门控系数...")
        for param in model.gluestick.parameters():
            param.requires_grad = False
        for param in model.semantic_adapter.parameters():
            param.requires_grad = True
        model.semantic_gate.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"[✓] 模型装载完成: 可训练参数量 {num_trainable:,} / 总参数量 {num_total:,}")

    # 3. 损失函数、优化器与调度器
    loss_fn = GlueStickLoss(
        point_weight=args.point_weight,
        line_weight=args.line_weight,
        nll_balancing=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. 训练主循环
    best_val_loss = float("inf")
    best_line_f1 = -1.0
    history: list[dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\n[Epoch {epoch:02d}/{args.epochs:02d}] LR: {scheduler.get_last_lr()[0]:.2e}")

        # 训练步骤
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            scaler=scaler,
            use_amp=(device.type == "cuda"),
        )
        scheduler.step()

        # 验证步骤
        val_res = {}
        if (epoch % args.val_interval == 0) or (epoch == args.epochs):
            val_res = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                use_amp=(device.type == "cuda"),
            )

        epoch_time = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_sec": round(epoch_time, 2),
            **train_res,
            **val_res,
        }
        history.append(record)

        # 打印当前轮结果
        print(
            f"  Train: Loss={train_res['train_loss']:.3f} "
            f"(Pt={train_res['train_point_loss']:.2f}, Ln={train_res['train_line_loss']:.2f}) | "
            f"Pt-P={train_res['train_point_precision']:.3f}, Ln-P={train_res['train_line_precision']:.3f}"
        )
        if val_res:
            print(
                f"  Val  : Loss={val_res['val_loss']:.3f} | "
                f"Pt-P={val_res['val_point_precision']:.3f}, Pt-R={val_res['val_point_recall']:.3f}, Pt-F1={val_res['val_point_f1']:.3f} | "
                f"Ln-P={val_res['val_line_precision']:.3f}, Ln-R={val_res['val_line_recall']:.3f}, Ln-F1={val_res['val_line_f1']:.3f}"
            )

        # 保存最新检查点
        latest_ckpt = save_dir / "latest_checkpoint.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "conf": vars(args),
            "record": record,
        }, latest_ckpt)

        # 评定最佳模型 (以综合验证 Loss 为基准)
        current_val_loss = val_res.get("val_loss", float("inf"))
        current_line_f1 = val_res.get("val_line_f1", 0.0)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_ckpt = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "conf": vars(args),
                "val_loss": best_val_loss,
                "val_line_f1": current_line_f1,
            }, best_ckpt)
            print(f"  [★] 创下最佳验证 Loss: {best_val_loss:.4f} -> 权重已保存至 {best_ckpt.name}")

        # 持久化训练历史
        with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60.0
    print("\n" + "=" * 75)
    print(f"[✓] 训练全部顺利完成！耗时: {total_time:.2f} 分钟")
    print(f"[✓] 最佳模型保存于: {save_dir / 'best_model.pth'}")
    print(f"[✓] 训练日志已写入: {save_dir / 'train_history.json'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
