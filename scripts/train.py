#!/usr/bin/env python3
"""v1 预训练入口。

脚本把配置、定长 token 数据和模型训练循环串起来：支持 memmap 数据读取、
单卡/torchrun 数据并行、混合精度、梯度累积、验证、Cosine 学习率和断点恢复。
正式训练前建议先使用 ``--dry-run`` 检查配置、数据和一次前向/反向计算。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# 从仓库根目录启动脚本时，Python 默认不会把 src 放入模块搜索路径。
V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V1_ROOT / "src"))

from data import PackedTokenDataset  # noqa: E402
from model import DecoderOnlyTransformer, ModelConfig  # noqa: E402
from utils.config import load_yaml_config, resolve_v1_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数；模型超参数仍从 YAML 读取。"""
    parser = argparse.ArgumentParser(description="Train the v1 decoder-only model")
    parser.add_argument("--config", type=Path, required=True, help="模型配置，例如 configs/small.yaml")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="运行设备；macOS Apple Silicon 可使用 mps",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="临时覆盖 training.max_steps")
    parser.add_argument("--resume", type=Path, default=None, help="临时覆盖 training.resume")
    parser.add_argument("--dry-run", action="store_true", help="只执行一个 batch 的前向/反向")
    return parser.parse_args()


def mps_available() -> bool:
    """检测 Apple Silicon 的 MPS 后端，同时兼容没有 MPS 属性的旧 PyTorch。"""

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        return False
    is_built = getattr(mps_backend, "is_built", None)
    return bool(is_built() if callable(is_built) else True)


def setup_process(requested_device: str) -> tuple[torch.device, int, int, bool]:
    """初始化设备和 DDP，并返回 device、rank、world size 和主进程标志。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    cuda_available = torch.cuda.is_available()
    mps_is_available = mps_available()
    if requested_device == "auto":
        if cuda_available:
            device = torch.device("cuda", local_rank if distributed else 0)
        elif mps_is_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        device = torch.device("cuda", local_rank if distributed else 0)
    elif requested_device == "mps":
        if not mps_is_available:
            raise RuntimeError("--device mps was requested but MPS is unavailable")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # PyTorch 的 DDP 后端可在 CPU 上使用 gloo；MPS 当前不作为 DDP 设备。
    if device.type == "mps" and distributed:
        raise RuntimeError("MPS training currently supports one process; omit torchrun")
    if device.type == "cuda" and distributed:
        torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size, rank == 0


def cleanup_process(distributed: bool) -> None:
    """释放 DDP 进程组。"""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    """让不同 DDP rank 使用可复现但不完全相同的随机流。"""
    value = int(seed) + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def load_training_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取模型配置及关联的数据配置，并检查必需字段。"""
    path = (config_path if config_path.is_absolute() else V1_ROOT / config_path).resolve()
    config = load_yaml_config(path)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"unsupported config schema_version: {config.get('schema_version')}")
    if not isinstance(config.get("model"), dict) or not config["model"]:
        raise ValueError("model config is empty; fill the selected model YAML first")
    if not isinstance(config.get("training"), dict) or not config["training"]:
        raise ValueError("training config is empty; fill the selected model YAML first")
    data_path = Path(config.get("data_config", "data.yaml"))
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    return config, load_yaml_config(data_path.resolve())


def prepare_data(
    data_config: dict[str, Any], model_config: ModelConfig, training: dict[str, Any]
) -> tuple[PackedTokenDataset, PackedTokenDataset, Path, Path]:
    """检查编码数据与模型长度/词表约束，并创建 memmap 数据集。"""
    encoding = data_config.get("encoding")
    if not isinstance(encoding, dict):
        raise ValueError("data.yaml is missing the encoding mapping")
    sequence_length = int(encoding["sequence_length"])
    if sequence_length != model_config.max_seq_len:
        raise ValueError(
            "model.max_seq_len and data.encoding.sequence_length must match: "
            f"{model_config.max_seq_len} != {sequence_length}"
        )
    output_dir = resolve_v1_path(encoding["output_dir"])
    train_path = output_dir / str(encoding["train_file"])
    validation_path = output_dir / str(encoding["validation_file"])
    train_set = PackedTokenDataset(train_path, sequence_length)
    validation_set = PackedTokenDataset(validation_path, sequence_length)
    manifest_path = resolve_v1_path(encoding["tokenizer_dir"]) / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            actual_vocab = int(json.load(handle).get("actual_vocab_size", -1))
        if actual_vocab != model_config.vocab_size:
            raise ValueError(f"tokenizer vocab {actual_vocab} != model vocab {model_config.vocab_size}")
    if int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.batch_size must be positive")
    return train_set, validation_set, train_path, validation_path


def make_loader(
    dataset: PackedTokenDataset,
    batch_size: int,
    training: dict[str, Any],
    distributed: bool,
    train: bool,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DistributedSampler | None]:
    """创建 DataLoader；DDP 时每个 rank 只读取自己的数据分片。"""
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    workers = max(0, int(training.get("num_workers", 0)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=workers,
        # pin_memory 只对 CUDA 主机到设备传输有意义，MPS/CPU 关闭可避免平台警告。
        pin_memory=bool(training.get("pin_memory", True)),
        drop_last=bool(training.get("drop_last", True)) if train else False,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def build_optimizer(model: nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    """矩阵参数使用 weight decay，Norm/bias 不衰减。"""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and "norm" not in name.lower():
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(training.get("weight_decay", 0.1))},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(training["learning_rate"]),
        betas=(float(training.get("beta1", 0.9)), float(training.get("beta2", 0.95))),
        eps=float(training.get("adam_eps", 1e-8)),
    )


def lr_factor(
    step: int, total_steps: int, warmup_steps: int, minimum_ratio: float
) -> float:
    """线性 warmup 后 cosine 衰减，返回相对峰值学习率的比例。"""
    if warmup_steps > 0 and step < warmup_steps:
        return max(step, 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def precision_settings(requested: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """根据设备能力选择 AMP 类型和是否需要 GradScaler。"""
    name = requested.lower()
    # MPS 的 AMP 支持随 macOS/PyTorch 版本变化；默认使用 fp32 保证可移植性。
    if device.type == "mps":
        if name != "fp32":
            print("提示：MPS 默认使用 fp32，忽略配置中的混合精度请求。", flush=True)
        return None, False
    if device.type != "cuda" or name == "fp32":
        return None, False
    if name == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    if name in {"bf16", "fp16"}:
        if name == "bf16":
            print("警告：GPU 不支持 bf16，自动改用 fp16。", flush=True)
        return torch.float16, True
    raise ValueError("training.precision must be one of bf16, fp16, fp32")


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_batches: int,
) -> float:
    """计算验证集平均 next-token loss。"""
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_batches = torch.zeros(1, device=device)
    with torch.no_grad():
        for batch_index, (input_ids, labels) in enumerate(loader):
            if batch_index >= max_batches:
                break
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            context = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
                       if autocast_dtype is not None and device.type == "cuda" else nullcontext())
            with context:
                _, loss = model(input_ids, labels)
            if loss is not None:
                total_loss += loss.detach()
                total_batches += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
    model.train()
    return float((total_loss / total_batches.clamp_min(1)).item())


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler,
    step: int, epoch: int, config: dict[str, Any]
) -> None:
    """原子写入 checkpoint，避免中断留下半个文件。"""
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    state = {"step": step, "epoch": epoch, "model": unwrapped.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "config": config,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR, scaler: torch.amp.GradScaler,
    device: torch.device
) -> tuple[int, int]:
    """恢复模型、优化器、调度器和 AMP 状态。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    target = model.module if isinstance(model, DistributedDataParallel) else model
    target.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))


def train(args: argparse.Namespace) -> None:
    """执行单卡或 DDP 预训练。"""
    device, rank, world_size, is_main = setup_process(args.device)
    distributed = world_size > 1
    try:
        config, data_config = load_training_config(args.config)
        model_config = ModelConfig.from_mapping(config["model"])
        training = dict(config["training"])
        if args.max_steps is not None:
            training["max_steps"] = args.max_steps
        if args.resume is not None:
            training["resume"] = str(args.resume)
        if device.type != "cuda":
            # 配置默认面向 CUDA；CPU/MPS 不使用 pinned memory。
            training["pin_memory"] = False
        set_seed(int(training.get("seed", 42)), rank)
        train_set, validation_set, train_path, validation_path = prepare_data(
            data_config, model_config, training
        )
        train_loader, train_sampler = make_loader(
            train_set, int(training["batch_size"]), training, distributed, True
        )
        validation_loader, _ = make_loader(
            validation_set, int(training["batch_size"]), training, distributed, False
        )
        if len(train_loader) == 0:
            raise ValueError("training DataLoader is empty; reduce batch_size or check data")

        model = DecoderOnlyTransformer(model_config).to(device)
        if is_main:
            print(f"模型参数量：{model.estimate_parameters():,} | 训练样本：{len(train_set):,} | "
                  f"验证样本：{len(validation_set):,} | device={device} world_size={world_size}", flush=True)
            print(f"训练文件：{train_path}\n验证文件：{validation_path}", flush=True)
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

        optimizer = build_optimizer(model, training)
        accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
        updates_per_epoch = math.ceil(len(train_loader) / accumulation)
        configured_steps = int(training.get("max_steps", 0))
        total_steps = configured_steps or int(training.get("epochs", 1)) * updates_per_epoch
        if total_steps <= 0:
            raise ValueError("training must contain a positive epochs or max_steps")
        warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.0)))
        peak_lr = float(training["learning_rate"])
        minimum_lr = float(training.get("min_learning_rate", 0.0))
        if minimum_lr < 0 or minimum_lr > peak_lr:
            raise ValueError("training.min_learning_rate must be between 0 and learning_rate")
        minimum_ratio = minimum_lr / peak_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: lr_factor(step, total_steps, warmup_steps, minimum_ratio),
        )
        autocast_dtype, use_scaler = precision_settings(str(training.get("precision", "fp32")), device)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        else:  # 兼容较旧的 PyTorch CPU/MPS 环境。
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        start_step, start_epoch = 0, 0
        resume = training.get("resume")
        if resume:
            resume_path = Path(resume)
            if not resume_path.is_absolute():
                resume_path = resolve_v1_path(resume_path)
            start_step, start_epoch = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device
            )
            if is_main:
                print(f"已恢复 checkpoint：{resume_path}，step={start_step}", flush=True)

        if args.dry_run:
            input_ids, labels = next(iter(train_loader))
            input_ids, labels = input_ids.to(device), labels.to(device)
            context = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
                       if autocast_dtype is not None and device.type == "cuda" else nullcontext())
            with context:
                _, loss = model(input_ids, labels)
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            (loss / accumulation).backward()
            if is_main:
                print(f"dry-run 成功：batch={tuple(input_ids.shape)} loss={loss.item():.6f}")
            return

        output_dir = resolve_v1_path(training.get("output_dir", "outputs/pretraining"))
        output_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (output_dir / "train.jsonl").open("a", encoding="utf-8") if is_main else None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        current_step = start_step
        last_log_time = time.perf_counter()
        running_loss = 0.0
        max_epochs = int(training.get("epochs", 1))

        for epoch in range(start_epoch, max_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch_index, (input_ids, labels) in enumerate(train_loader):
                if current_step >= total_steps:
                    break
                input_ids, labels = input_ids.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                is_update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
                sync_context = model.no_sync() if distributed and not is_update else nullcontext()
                with sync_context:
                    context = (torch.autocast(device_type="cuda", dtype=autocast_dtype)
                               if autocast_dtype is not None and device.type == "cuda" else nullcontext())
                    with context:
                        _, loss = model(input_ids, labels)
                    if loss is None:
                        raise RuntimeError("model did not return a training loss")
                    running_loss += float(loss.detach())
                    scaler.scale(loss / accumulation).backward()
                if not is_update:
                    continue
                if float(training.get("max_grad_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                current_step += 1

                log_every = max(1, int(training.get("log_every_steps", 10)))
                if is_main and current_step % log_every == 0:
                    elapsed = max(time.perf_counter() - last_log_time, 1e-6)
                    record = {"step": current_step, "epoch": epoch,
                              "loss": running_loss / accumulation,
                              "learning_rate": optimizer.param_groups[0]["lr"],
                              "steps_per_second": log_every / elapsed}
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    if log_handle:
                        log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log_handle.flush()
                    running_loss, last_log_time = 0.0, time.perf_counter()

                eval_every = int(training.get("eval_every_steps", 0))
                if eval_every > 0 and current_step % eval_every == 0:
                    validation_loss = evaluate(model, validation_loader, device, autocast_dtype,
                                              int(training.get("eval_batches", 100)))
                    if is_main:
                        record = {"step": current_step, "validation_loss": validation_loss,
                                  "validation_perplexity": math.exp(min(validation_loss, 20.0))}
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                        if log_handle:
                            log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                            log_handle.flush()

                save_every = int(training.get("save_every_steps", 0))
                if is_main and save_every > 0 and current_step % save_every == 0:
                    save_checkpoint(output_dir / f"step_{current_step:08d}.pt", model, optimizer,
                                    scheduler, scaler, current_step, epoch, config)
            if current_step >= total_steps:
                break

        if is_main:
            save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler,
                            current_step, max_epochs, config)
            if log_handle:
                log_handle.close()
            print(f"训练完成：step={current_step}，checkpoint={output_dir / 'last.pt'}")
    finally:
        cleanup_process(distributed)


if __name__ == "__main__":
    train(parse_args())
