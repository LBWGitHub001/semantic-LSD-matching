"""GlueStick 与语义增强 GlueStick 统一训练主程序 (GlueStick Training Pipeline).

支持:
1. 原版几何 GlueStick 微调训练 (Point + Wireframe Line);
2. SAM ViT-H 语义融合增强版 (SemanticGlueStick) 训练;
3. 自动场景隔离划分 Train / Val 集 (零数据泄露);
4. 混合精度 AMP 加速与梯度裁剪 (适应 8GB RTX 4060 显存);
5. 自动记录评估指标 (Point/Line Precision, Recall, F1) 与最佳权重保存;
6. 自动按运行次数创建 run1, run2, run3 等归档目录;
7. 详细运行日志 (train.log)、运行参数 (args.json/args.yaml)、每5轮周期权重、best 和 last 权重持久化存储。
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import json
import logging
from pathlib import Path
import re
import shutil
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


def get_next_run_dir(base_dir: Path, prefix: str = "run") -> Path:
    """根据已有运行目录序号自动探测并创建下一个 run 目录 (如 run1, run2, run3 ...)。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_indices: list[int] = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for item in base_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing_indices.append(int(match.group(1)))
    next_idx = max(existing_indices) + 1 if existing_indices else 1
    run_dir = base_dir / f"{prefix}{next_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_run_logger(log_file: Path) -> logging.Logger:
    """初始化运行详细日志记录器 (同时写入终端与 runX/train.log 文件)。"""
    logger = logging.getLogger(f"GlueStick_{log_file.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 清空已有 handlers，避免多实例或重复配置冲突
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # 1. 文件日志 Handler (带精确时间戳与毫秒级记录)
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # 2. 终端输出 Handler (纯净格式输出，不带时间戳前缀)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def _log_to_file(logger: logging.Logger, msg: str) -> None:
    """仅向文件日志写入内容，避免在终端打断 tqdm 实时进度条。"""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            record = logger.makeRecord(
                name=logger.name,
                level=logging.INFO,
                fn="(step)",
                lno=0,
                msg=msg,
                args=(),
                exc_info=None,
            )
            handler.handle(record)


def log_parameter_groups(model: nn.Module, logger: logging.Logger) -> None:
    """按子模块打印可训练与冻结参数量。"""
    logger.info("[*] 参数分组 (可训练 / 冻结):")
    gluestick = model.gluestick
    for name, module in gluestick.named_children():
        trainable = frozen = 0
        for param in module.parameters():
            count = param.numel()
            if param.requires_grad:
                trainable += count
            else:
                frozen += count
        logger.info(f"    - gluestick.{name}: 可训练 {trainable:,} / 冻结 {frozen:,}")
    for name, param in gluestick.named_parameters():
        if "." in name:
            continue
        state = "可训练" if param.requires_grad else "冻结"
        logger.info(f"    - gluestick.{name}: {state} {param.numel():,}")
    for name, module in model.named_children():
        if name == "gluestick":
            continue
        trainable = frozen = 0
        for param in module.parameters():
            count = param.numel()
            if param.requires_grad:
                trainable += count
            else:
                frozen += count
        logger.info(f"    - {name}: 可训练 {trainable:,} / 冻结 {frozen:,}")
    for name, param in model.named_parameters():
        if "." in name:
            continue
        state = "可训练" if param.requires_grad else "冻结"
        logger.info(f"    - {name}: {state} {param.numel():,}")


def save_run_parameters(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    """将本次训练运行的完整参数配置持久化保存为 args.json 和 args.yaml。"""
    raw_args = vars(args)
    serializable_args: dict[str, Any] = {}
    for k, v in raw_args.items():
        if isinstance(v, Path):
            serializable_args[k] = str(v)
        else:
            serializable_args[k] = v

    # 1. 保存 JSON 参数文件
    json_path = run_dir / "args.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_args, f, indent=2, ensure_ascii=False)

    # 2. 保存 YAML 参数文件 (若环境支持)
    try:
        import yaml
        yaml_path = run_dir / "args.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(serializable_args, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception:
        pass

    return serializable_args


def _create_alias_or_copy(src: Path, dst: Path) -> None:
    """创建相对软链接别名以兼容旧版命名格式，失败时自动安全回退为拷贝。"""
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.name)
    except OSError:
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
    clip_grad: float = 10.0,
    epoch: int = 1,
    total_epochs: int = 1,
    logger: logging.Logger | None = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """训练单轮 Epoch。"""
    model.train()
    total_loss = 0.0
    total_pt_loss = 0.0
    total_ln_loss = 0.0
    total_pt_prec, total_pt_rec = 0.0, 0.0
    total_ln_prec, total_ln_rec = 0.0, 0.0
    steps = 0
    total_steps = len(loader) if hasattr(loader, "__len__") else 0

    pbar = tqdm(loader, desc=f"  Train [{epoch:02d}/{total_epochs:02d}]", leave=False)
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
            warn_msg = f"  [!] 警告: [Epoch {epoch:02d} Step {steps+1}] 检测到 NaN/Inf Loss，跳过本步反向传播"
            if logger:
                logger.warning(warn_msg)
            else:
                print(warn_msg)
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

        if logger and log_interval > 0 and (steps % log_interval == 0 or (total_steps > 0 and steps == total_steps)):
            step_log = (
                f"  [Train] Epoch [{epoch:02d}/{total_epochs:02d}] Step [{steps:04d}/{total_steps:04d}] "
                f"Loss: {stats['total_loss']:.4f} (Pt: {stats['point_loss']:.4f}, Ln: {stats['line_loss']:.4f}) | "
                f"Pt-P: {metrics['point_precision']:.4f}, Ln-P: {metrics['line_precision']:.4f}"
            )
            _log_to_file(logger, step_log)

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
    epoch: int = 1,
    total_epochs: int = 1,
    logger: logging.Logger | None = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """验证集全量评测。"""
    model.eval()
    total_loss = 0.0
    total_pt_loss = 0.0
    total_ln_loss = 0.0
    total_pt_prec, total_pt_rec = 0.0, 0.0
    total_ln_prec, total_ln_rec = 0.0, 0.0
    steps = 0
    total_steps = len(loader) if hasattr(loader, "__len__") else 0
    if max_eval_steps:
        total_steps = min(total_steps, max_eval_steps) if total_steps > 0 else max_eval_steps

    pbar = tqdm(loader, desc=f"  Val   [{epoch:02d}/{total_epochs:02d}]", leave=False)
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

        if logger and log_interval > 0 and (steps % log_interval == 0 or (total_steps > 0 and steps == total_steps)):
            step_log = (
                f"  [Val]   Epoch [{epoch:02d}/{total_epochs:02d}] Step [{steps:04d}/{total_steps:04d}] "
                f"Loss: {stats['total_loss']:.4f} | Pt-P: {metrics['point_precision']:.4f}, Ln-P: {metrics['line_precision']:.4f}"
            )
            _log_to_file(logger, step_log)

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
    default_config_path = ROOT / "configs" / "train_config.json"
    default_weights = ROOT / "third_party" / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
    if not default_weights.is_file():
        default_weights = ROOT / "GlueStick" / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"

    parser = argparse.ArgumentParser(description="GlueStick 几何与语义融合训练管道")
    parser.add_argument("--config", default=str(default_config_path) if default_config_path.is_file() else None, help="JSON 配置文件路径 (默认自动读取 configs/train_config.json)")
    parser.add_argument("--data-dir", default="data/scannet_semantic_gluestick", help="训练数据集根目录")
    parser.add_argument("--weights", default=str(default_weights), help="预训练模型权重路径")
    parser.add_argument("--use-semantics", dest="use_semantics", action="store_true", default=None, help="启用 SAM ViT-H 语义融合增强分支")
    parser.add_argument("--no-semantics", dest="use_semantics", action="store_false", help="禁用 SAM 语义增强分支")
    parser.add_argument("--freeze-base", dest="freeze_base", action="store_true", default=None, help="冻结 GlueStick 主干，仅训练语义适配层")
    parser.add_argument("--no-freeze-base", dest="freeze_base", action="store_false", help="不冻结 GlueStick 主干 (全量微调)")
    parser.add_argument("--freeze-gnn", dest="freeze_gnn", action="store_true", default=None, help="冻结 GNN 参数，仅训练编码器与匹配头")
    parser.add_argument("--no-freeze-gnn", dest="freeze_gnn", action="store_false", help="不冻结 GNN，允许更新图注意力参数")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小 (GlueStick 变长图推荐 1 或 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率 (微调语义适配器推荐 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--point-weight", type=float, default=1.0, help="特征点匹配损失权重")
    parser.add_argument("--line-weight", type=float, default=1.0, help="线段匹配损失权重")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="验证集场景比例 (默认 15%%)")
    parser.add_argument("--save-dir", default="checkpoints/gluestick_train", help="模型检查点保存根目录")
    parser.add_argument("--run-name", default=None, help="指定本次运行子目录名称 (默认根据运行次数自动命名为 run1, run2, run3 等)")
    parser.add_argument("--save-interval", type=int, default=5, help="每隔几个 Epoch 保存一次周期权重 (默认 5)")
    parser.add_argument("--val-interval", type=int, default=1, help="每隔几个 Epoch 进行一次验证")
    parser.add_argument("--log-interval", type=int, default=50, help="每隔多少步向详细日志写入一次记录 (默认 50)")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--max-train-samples", type=int, default=None, help="最多使用的训练样本数 (供快速冒烟测试)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="最多使用的验证样本数 (供快速冒烟测试)")

    # 1. 预解析 --config，将配置文件作为 defaults 注入，使 CLI 显式参数具有更高优先级
    pre_args, _ = parser.parse_known_args()
    config_applied = None
    if pre_args.config:
        cfg_path = Path(pre_args.config).resolve()
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 过滤注释字段
            clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
            parser.set_defaults(**clean_cfg)
            config_applied = cfg_path

    args = parser.parse_args()

    # 默认值兜底
    if args.use_semantics is None:
        args.use_semantics = True
    if args.freeze_base is None:
        args.freeze_base = True
    if args.freeze_gnn is None:
        args.freeze_gnn = False

    # 2. 确定运行存储目录 (支持 run1, run2, run3 ... 递增命名)
    base_save_dir = Path(args.save_dir).resolve()
    base_save_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name:
        run_dir = base_save_dir / args.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = get_next_run_dir(base_save_dir, prefix="run")

    # 3. 初始化详细日志系统 (同时输出终端与 runX/train.log)
    log_file = run_dir / "train.log"
    logger = setup_run_logger(log_file)

    # 4. 持久化记录运行参数 (args.json & args.yaml) 并备份所用配置文件
    serializable_args = save_run_parameters(args, run_dir)
    if config_applied and config_applied.is_file():
        try:
            shutil.copy2(config_applied, run_dir / "train_config.json")
        except Exception:
            pass

    device = torch.device(args.device)

    logger.info("=" * 75)
    logger.info("  🚀 启动 GlueStick 深度匹配模型训练管道 (GlueStick Training Pipeline)")
    logger.info("=" * 75)
    if config_applied:
        logger.info(f"[*] 已成功装载配置文件: {config_applied}")
    logger.info(f"[*] 运行输出目录: {run_dir} (本次运行: {run_dir.name})")
    logger.info(f"[*] 详细运行日志: {log_file}")
    logger.info(f"[*] 参数配置文件: {run_dir / 'args.json'}")
    logger.info(f"[*] 数据集路径: {args.data_dir}")
    logger.info(f"[*] 语义增强模式 (use_semantics): {args.use_semantics}")
    logger.info(f"[*] 冻结 GNN (freeze_gnn): {args.freeze_gnn}")
    logger.info(f"[*] 预训练权重: {args.weights}")
    logger.info(
        f"[*] 训练参数: Epochs={args.epochs}, BatchSize={args.batch_size}, "
        f"LR={args.lr}, Device={device}, SaveInterval={args.save_interval}"
    )
    logger.info("-" * 75)
    logger.info("[*] 完整运行参数明细:")
    for k, v in serializable_args.items():
        logger.info(f"    - {k}: {v}")
    logger.info("-" * 75)

    # 1. 构建 Train / Val 数据集与加载器
    train_ds = SemanticGlueStickDataset(args.data_dir, split="train", val_ratio=args.val_ratio)
    val_ds = SemanticGlueStickDataset(args.data_dir, split="val", val_ratio=args.val_ratio)

    if args.max_train_samples:
        train_ds.sample_names = train_ds.sample_names[:args.max_train_samples]
    if args.max_val_samples:
        val_ds.sample_names = val_ds.sample_names[:args.max_val_samples]

    logger.info(f"[✓] 数据集隔离装载完成: 训练集 {len(train_ds)} 样本, 验证集 {len(val_ds)} 样本")

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
        freeze_gnn=args.freeze_gnn,
    ).to(device)

    if args.freeze_base and args.use_semantics:
        logger.info("[*] 冻结 GlueStick 主干参数，仅优化 SAM 语义适配器与门控系数...")
        for param in model.gluestick.parameters():
            param.requires_grad = False
        for param in model.semantic_adapter.parameters():
            param.requires_grad = True
        model.semantic_gate.requires_grad = True

    if args.freeze_gnn:
        logger.info("[*] 冻结 GNN 参数，仅优化编码器与匹配头...")
        for param in model.gluestick.gnn.parameters():
            param.requires_grad = False
        model.gluestick.gnn.eval()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    logger.info(f"[✓] 模型装载完成: 可训练参数量 {num_trainable:,} / 总参数量 {num_total:,}")
    log_parameter_groups(model, logger)

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
    best_pt_f1 = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []

    start_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            logger.info(f"\n[Epoch {epoch:02d}/{args.epochs:02d}] LR: {scheduler.get_last_lr()[0]:.2e}")

            # 训练步骤
            train_res = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                use_amp=(device.type == "cuda"),
                clip_grad=10.0,
                epoch=epoch,
                total_epochs=args.epochs,
                logger=logger,
                log_interval=args.log_interval,
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
                    epoch=epoch,
                    total_epochs=args.epochs,
                    logger=logger,
                    log_interval=args.log_interval,
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

            # 打印与记录当前轮结果
            logger.info(
                f"  Train: Loss={train_res['train_loss']:.3f} "
                f"(Pt={train_res['train_point_loss']:.2f}, Ln={train_res['train_line_loss']:.2f}) | "
                f"Pt-P={train_res['train_point_precision']:.3f}, Ln-P={train_res['train_line_precision']:.3f} | "
                f"耗时: {epoch_time:.1f}s"
            )
            if val_res:
                logger.info(
                    f"  Val  : Loss={val_res['val_loss']:.3f} | "
                    f"Pt-P={val_res['val_point_precision']:.3f}, Pt-R={val_res['val_point_recall']:.3f}, Pt-F1={val_res['val_point_f1']:.3f} | "
                    f"Ln-P={val_res['val_line_precision']:.3f}, Ln-R={val_res['val_line_recall']:.3f}, Ln-F1={val_res['val_line_f1']:.3f}"
                )

            # 检查点基础数据载荷
            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "conf": serializable_args,
                "record": record,
            }

            # 1. 始终保存最新权重 (last.pth) 与兼容别名 (latest_checkpoint.pth)
            last_ckpt = run_dir / "last.pth"
            torch.save(ckpt_payload, last_ckpt)
            _create_alias_or_copy(last_ckpt, run_dir / "latest_checkpoint.pth")

            # 2. 每 save_interval 轮 (默认 5 轮) 保存一次周期权重 (epoch_*.pth)
            if args.save_interval > 0 and (epoch % args.save_interval == 0):
                epoch_ckpt = run_dir / f"epoch_{epoch}.pth"
                torch.save(ckpt_payload, epoch_ckpt)
                logger.info(f"  [💾] 周期权重检查点已保存: {epoch_ckpt.name}")

            # 3. 评定并保存最佳模型 (best.pth) 与兼容别名 (best_model.pth)
            current_val_loss = val_res.get("val_loss", float("inf"))
            current_line_f1 = val_res.get("val_line_f1", 0.0)
            current_pt_f1 = val_res.get("val_point_f1", 0.0)

            if current_val_loss < best_val_loss:
                prev_loss = best_val_loss
                best_val_loss = current_val_loss
                best_line_f1 = current_line_f1
                best_pt_f1 = current_pt_f1
                best_epoch = epoch
                best_ckpt = run_dir / "best.pth"
                best_payload = {
                    **ckpt_payload,
                    "best_epoch": best_epoch,
                    "val_loss": best_val_loss,
                    "val_line_f1": best_line_f1,
                    "val_point_f1": best_pt_f1,
                }
                torch.save(best_payload, best_ckpt)
                _create_alias_or_copy(best_ckpt, run_dir / "best_model.pth")
                loss_diff_str = (
                    f" (从 {prev_loss:.4f} 下降至 {best_val_loss:.4f})"
                    if prev_loss != float("inf")
                    else f" ({best_val_loss:.4f})"
                )
                logger.info(f"  [★] 创下最佳验证 Loss{loss_diff_str} -> 权重已保存至 {best_ckpt.name}")

            # 4. 持久化训练历史记录
            with open(run_dir / "train_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

    except KeyboardInterrupt:
        logger.warning("\n[!] 检测到用户手动中断训练 (Ctrl+C)！正在安全保存当前状态并退出...")

    # 确保 best.pth 存在 (防止未进行 val 或提前退出时缺少 best)
    best_ckpt = run_dir / "best.pth"
    if not best_ckpt.exists() and (run_dir / "last.pth").exists():
        shutil.copy2(run_dir / "last.pth", best_ckpt)
        _create_alias_or_copy(best_ckpt, run_dir / "best_model.pth")

    # 更新父级目录软链接 latest 指向最新 run 目录
    try:
        latest_symlink = base_save_dir / "latest"
        if latest_symlink.is_symlink() or latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(run_dir.name)
    except OSError:
        pass

    total_time = (time.time() - start_time) / 60.0
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "total_epochs": args.epochs,
        "best_epoch": best_epoch if best_epoch > 0 else None,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "best_line_f1": best_line_f1 if best_line_f1 >= 0 else None,
        "total_time_minutes": round(total_time, 2),
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 75)
    logger.info(f"[✓] 训练流程结束！总耗时: {total_time:.2f} 分钟")
    logger.info(f"[✓] 本次运行归档目录: {run_dir}")
    logger.info(f"[✓] 最佳权重 (best): {run_dir / 'best.pth'} (最佳 Epoch: {best_epoch if best_epoch > 0 else 'N/A'})")
    logger.info(f"[✓] 最终权重 (last): {run_dir / 'last.pth'}")
    logger.info(f"[✓] 周期权重 (epoch): 每 {args.save_interval} 轮权重文件已保存")
    logger.info(f"[✓] 详细运行日志: {run_dir / 'train.log'}")
    logger.info(f"[✓] 运行参数文件: {run_dir / 'args.json'}")
    logger.info(f"[✓] 历史指标数据: {run_dir / 'train_history.json'}")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
