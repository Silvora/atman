"""Step 04: pretrain an Atman causal language model from scratch.

脚本读取第 3 步的 train/validation NumPy 分片，按需 mmap 到内存。small、medium、
large 只改变网络容量，三档模型共享同一 tokenizer、序列长度和训练数据。模型实现
使用 Transformers 的 GPT2LMHeadModel，但权重完全随机初始化，并未加载 GPT-2。
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
)


STEP_ID = "04_pretraining"
STEP_NAME = "Pretraining"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "03_encoding" / "output" / "manifest.json"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "02_tokenizer" / "output" / "atman_tokenizer"
DEFAULT_OUTPUT_ROOT = STEP_DIR / "output"

# 参数量会随实际词表略微变化。32K 词表下约为 36M、82M、134M。
MODEL_PRESETS = {
    "small": {"n_layer": 6, "n_embd": 512, "n_head": 8},
    "medium": {"n_layer": 8, "n_embd": 768, "n_head": 12},
    "large": {"n_layer": 8, "n_embd": 1024, "n_head": 16},
}

# 单卡默认值偏保守，优先保证三档都能启动；可按显存通过命令行覆盖。
TRAINING_PRESETS = {
    "small": {"batch_size": 16, "gradient_accumulation_steps": 4},
    "medium": {"batch_size": 8, "gradient_accumulation_steps": 8},
    "large": {"batch_size": 4, "gradient_accumulation_steps": 16},
}

DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_WARMUP_RATIO = 0.02
DEFAULT_DROPOUT = 0.1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MAX_EVAL_BATCHES = 50
DEFAULT_LOG_EVERY = 50
DEFAULT_EVAL_EVERY = 1_000
DEFAULT_SAVE_EVERY = 2_000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class DistributedContext:
    """记录 torchrun 为当前进程分配的身份及设备。"""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    backend: str | None


def setup_distributed(requested_device: str, cli_local_rank: int | None):
    """检测 torchrun 环境；多 GPU 使用 NCCL，CPU 联调使用 Gloo。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        context = DistributedContext(False, 0, 0, 1, True, None)
        return context, resolve_device(requested_device)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", cli_local_rank or 0))
    if requested_device == "mps":
        raise ValueError("MPS 不支持本脚本的多进程训练；多卡请使用 CUDA + torchrun")

    use_cuda = requested_device in {"auto", "cuda"} and torch.cuda.is_available()
    if requested_device == "cuda" and not use_cuda:
        raise ValueError("请求了 CUDA 多卡训练，但当前 PyTorch 检测不到 CUDA")

    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank}，但只检测到 {torch.cuda.device_count()} 张 CUDA GPU"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        # Gloo 路径用于无 GPU 环境验证 DDP 逻辑，不建议承担正式预训练。
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    context = DistributedContext(True, rank, local_rank, world_size, rank == 0, backend)
    return context, device


def cleanup_distributed() -> None:
    """释放 torchrun 进程组，确保脚本正常退出。"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """保存与单进程验证时取出 DDP 包装内的原始模型。"""
    return model.module if isinstance(model, DDP) else model


def print_main(context: DistributedContext, message: str) -> None:
    """避免每个 rank 重复打印同一条日志。"""
    if context.is_main:
        print(message)


def sha256_file(path: Path) -> str:
    """计算 tokenizer.json 摘要，确认第 3、4 步使用同一份词表。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoding_manifest(manifest_path: Path) -> tuple[dict, list[Path], list[Path]]:
    """读取第 3 步清单，并检查 train/validation 分片是否完整。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到第 3 步 manifest: {manifest_path.resolve()}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generated_by_step") != "03_encoding":
        raise ValueError("输入 manifest 不是第 3 步生成的")
    if manifest.get("format_version") != 2:
        raise ValueError("第 3 步数据格式不是当前要求的 version 2，请重新编码")
    if manifest.get("status") != "completed":
        raise ValueError(f"第 3 步尚未完成: {manifest.get('status')}")

    def resolve_split(split: str) -> list[Path]:
        split_info = manifest.get("splits", {}).get(split, {})
        paths = [manifest_path.parent / item["file"] for item in split_info.get("shards", [])]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{split} 分片不存在: {missing[0].resolve()}")
        return paths

    train_paths = resolve_split("train")
    validation_paths = resolve_split("validation")
    if not train_paths:
        raise ValueError("第 3 步 manifest 没有 train 分片")
    return manifest, train_paths, validation_paths


class NpyShardDataset(IterableDataset):
    """逐个 mmap 分片，并按 rank/worker 切出互不重复的 token 序列。"""

    def __init__(
        self,
        shard_paths: list[Path],
        seed: int,
        shuffle: bool,
        max_samples: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.seed = seed
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # 只 mmap 文件头来获得行数，不会把 token 数据整体读入内存。
        self.shard_sample_counts = []
        remaining = max_samples
        for path in self.shard_paths:
            token_array = np.load(path, mmap_mode="r", allow_pickle=False)
            if token_array.ndim != 2:
                raise ValueError(f"token 分片不是二维数组: {path}")
            count = int(token_array.shape[0])
            if max_samples:
                count = min(count, max(remaining, 0))
                remaining -= count
            self.shard_sample_counts.append(count)
            del token_array

    def samples_for_rank(self, rank: int) -> int:
        """计算某个 rank 实际获得的序列数，用于统一所有 rank 的训练步数。"""
        return sum(
            max(0, (count - rank + self.world_size - 1) // self.world_size)
            for count in self.shard_sample_counts
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_indices = list(range(len(self.shard_paths)))
        if self.shuffle:
            shard_indices = torch.randperm(len(shard_indices), generator=generator).tolist()

        for shard_index in shard_indices:
            allowed_rows = self.shard_sample_counts[shard_index]
            if allowed_rows <= 0:
                continue
            token_array = np.load(
                self.shard_paths[shard_index],
                mmap_mode="r",
                allow_pickle=False,
            )
            if token_array.ndim != 2:
                raise ValueError(f"token 分片不是二维数组: {self.shard_paths[shard_index]}")

            row_indices = list(range(allowed_rows))
            if self.shuffle:
                row_generator = torch.Generator().manual_seed(
                    self.seed + self.epoch * 100_000 + shard_index
                )
                row_indices = torch.randperm(
                    allowed_rows,
                    generator=row_generator,
                ).tolist()

            # 先按训练进程切分，再在该进程内部按 DataLoader worker 切分。
            row_indices = row_indices[self.rank :: self.world_size]
            if worker is not None:
                row_indices = row_indices[worker.id :: worker.num_workers]

            for row_index in row_indices:
                # mmap 是只读的；复制单行后转 int64，满足 embedding 的索引类型要求。
                yield torch.from_numpy(
                    np.array(token_array[row_index], dtype=np.int64, copy=True)
                )


def resolve_device(requested: str) -> torch.device:
    """auto 按 CUDA、MPS、CPU 的顺序选择可用设备。"""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("请求了 CUDA，但当前 PyTorch 检测不到 CUDA")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("请求了 MPS，但当前 PyTorch 检测不到 MPS")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    """根据设备选择混合精度；bf16 不需要 GradScaler，fp16 需要。"""
    precision = requested
    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        elif device.type == "mps":
            precision = "fp16"
        else:
            precision = "no"

    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("当前设备不支持 bf16，请使用 --mixed-precision fp16 或 no")
        return precision, torch.bfloat16
    if precision == "fp16":
        if device.type not in {"cuda", "mps"}:
            raise ValueError("fp16 混合精度只用于 CUDA 或 MPS")
        return precision, torch.float16
    return "no", None


def build_grad_scaler(device: torch.device, precision: str):
    """只有 fp16 启用损失缩放，以降低半精度梯度下溢风险。"""
    enabled = precision == "fp16" and device.type in {"cuda", "mps"}
    if hasattr(torch, "GradScaler"):
        try:
            return torch.GradScaler(device.type, enabled=enabled)
        except (TypeError, RuntimeError):
            pass
    if hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except (TypeError, RuntimeError):
            pass
    # PyTorch 旧版本只有 CUDA scaler；非 CUDA 环境以 disabled 模式复用其接口。
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_model_config(args: argparse.Namespace, tokenizer, seq_length: int) -> GPT2Config:
    """按档位创建 decoder-only 模型结构，权重随后从零初始化。"""
    preset = MODEL_PRESETS[args.model_size]
    return GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=seq_length,
        n_ctx=seq_length,
        n_embd=preset["n_embd"],
        n_layer=preset["n_layer"],
        n_head=preset["n_head"],
        activation_function="gelu_new",
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        loss_type="ForCausalLM",
        use_cache=False,
    )


def find_latest_checkpoint(output_dir: Path) -> tuple[Path | None, int]:
    """查找 step 最大且同时包含模型配置和权重的 checkpoint。"""
    latest_path = None
    latest_step = 0
    if not output_dir.exists():
        return latest_path, latest_step

    for path in output_dir.glob("checkpoint-step-*"):
        step_text = path.name.removeprefix("checkpoint-step-")
        has_weights = (path / "model.safetensors").is_file() or (
            path / "pytorch_model.bin"
        ).is_file()
        if path.is_dir() and step_text.isdigit() and has_weights and (path / "config.json").is_file():
            step = int(step_text)
            if step > latest_step:
                latest_path = path
                latest_step = step
    return latest_path, latest_step


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    output_dir: Path,
    global_step: int,
) -> Path:
    """同时保存模型和训练状态，使续训能恢复优化器与学习率。"""
    checkpoint_dir = output_dir / f"checkpoint-step-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if scaler.is_enabled():
        state["scaler"] = scaler.state_dict()
    torch.save(state, checkpoint_dir / "training_state.pt")
    return checkpoint_dir


def load_training_state(checkpoint_dir: Path, optimizer, scheduler, scaler) -> bool:
    """恢复优化器、调度器和可选 GradScaler；旧式纯权重目录返回 False。"""
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.is_file():
        return False
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler.is_enabled() and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return True


@torch.no_grad()
def evaluate(model, dataloader, device, dtype, max_batches: int) -> dict | None:
    """在第 3 步固定的 validation split 上计算平均 loss 和 perplexity。"""
    if dataloader is None:
        return None
    model.eval()
    losses = []
    for batch_index, input_ids in enumerate(dataloader, start=1):
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            loss = model(input_ids=input_ids, labels=input_ids).loss
        losses.append(float(loss.detach().cpu()))
        if max_batches and batch_index >= max_batches:
            break
    model.train()
    if not losses:
        return None
    loss = sum(losses) / len(losses)
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "batches": len(losses),
    }


def average_across_ranks(
    value: float,
    context: DistributedContext,
    device: torch.device,
) -> float:
    """把各 rank 的局部 loss 汇总为便于观察的全局平均值。"""
    if not context.enabled:
        return value
    value_tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
    return float(value_tensor.item() / context.world_size)


def evaluate_on_main_rank(
    model,
    dataloader,
    device,
    dtype,
    max_batches: int,
    context: DistributedContext,
) -> dict | None:
    """只让 rank 0 跑完整验证集，其他 rank 在前后同步等待。"""
    if context.enabled:
        dist.barrier()
    result = None
    if context.is_main:
        result = evaluate(unwrap_model(model), dataloader, device, dtype, max_batches)
    if context.enabled:
        dist.barrier()
    return result


def validate_args(args: argparse.Namespace) -> None:
    """在创建模型前检查参数，避免长任务启动后才发现简单配置错误。"""
    positive_names = (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "save_every",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if args.max_steps < 0 or args.max_train_samples < 0:
        raise ValueError("max_steps 和 max_train_samples 不能小于 0")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup_ratio 必须在 [0, 1) 范围内")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout 必须在 [0, 1) 范围内")


def pretrain(args: argparse.Namespace) -> None:
    """执行第 4 步训练、周期验证、checkpoint 和最终模型保存。"""
    validate_args(args)
    context, device = setup_distributed(args.device, args.local_rank)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / args.model_size
    manifest, train_paths, validation_paths = load_encoding_manifest(manifest_path)
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"找不到第 2 步 tokenizer: {tokenizer_dir}")

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if sha256_file(tokenizer_json_path) != manifest.get("tokenizer_json_sha256"):
        raise ValueError("第 2 步 tokenizer 与第 3 步编码数据不匹配")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if len(tokenizer) != int(manifest["tokenizer_vocab_size"]):
        raise ValueError("tokenizer 词表大小与第 3 步 manifest 不一致")

    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        dist.barrier()
    precision, dtype = resolve_precision(args.mixed_precision, device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    seq_length = int(manifest["seq_length"])
    config = build_model_config(args, tokenizer, seq_length)
    resume_checkpoint, resume_step = find_latest_checkpoint(output_dir) if args.auto_resume else (None, 0)
    if resume_checkpoint:
        print_main(context, f"恢复 checkpoint: {resume_checkpoint}")
        model = GPT2LMHeadModel.from_pretrained(resume_checkpoint)
        model.config.use_cache = False
    else:
        model = GPT2LMHeadModel(config)
    # 部分 Transformers 版本从 model.loss_type 而不是 config.loss_type 取值。
    model.loss_type = "ForCausalLM"
    model.to(device)

    train_chunks = int(manifest["splits"]["train"]["num_chunks"])
    validation_chunks = int(manifest["splits"]["validation"]["num_chunks"])
    if args.max_train_samples:
        train_chunks = min(train_chunks, args.max_train_samples)

    train_dataset = NpyShardDataset(
        train_paths,
        seed=args.seed,
        shuffle=True,
        max_samples=args.max_train_samples,
        rank=context.rank,
        world_size=context.world_size,
    )
    if context.enabled:
        # 每个 rank 必须执行相同步数；以样本最少的 rank 为准，丢弃少量尾部样本。
        min_rank_samples = min(
            train_dataset.samples_for_rank(rank) for rank in range(context.world_size)
        )
        batches_per_rank = min_rank_samples // args.batch_size
        steps_per_epoch = batches_per_rank // args.gradient_accumulation_steps
    else:
        min_rank_samples = train_dataset.samples_for_rank(0)
        steps_per_epoch = math.ceil(
            min_rank_samples / args.batch_size / args.gradient_accumulation_steps
        )
    full_steps = steps_per_epoch * args.epochs
    total_steps = min(full_steps, args.max_steps) if args.max_steps else full_steps
    if total_steps <= 0:
        raise ValueError("每个 rank 的 token chunk 不足以完成一次优化，请减小 batch 或梯度累积")
    if resume_step >= total_steps:
        print_main(context, f"已有 checkpoint 达到目标步数: {resume_step}/{total_steps}")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=context.enabled,
    )
    validation_loader = None
    if context.is_main and validation_paths:
        validation_dataset = NpyShardDataset(
            validation_paths,
            seed=args.seed,
            shuffle=False,
            max_samples=0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    if context.enabled:
        ddp_options = {}
        # 新版 PyTorch 用 forward_sync_buffers；旧版仍使用 broadcast_buffers。
        if "forward_sync_buffers" in inspect.signature(DDP).parameters:
            ddp_options["forward_sync_buffers"] = False
        else:
            ddp_options["broadcast_buffers"] = False
        if device.type == "cuda":
            ddp_options.update(
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        model = DDP(model, **ddp_options)
        # 模型参数已由 DDP 同步；不同 rank 使用不同随机流生成 dropout mask。
        torch.manual_seed(args.seed + context.rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + context.rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = build_grad_scaler(device, precision)
    state_loaded = False
    if resume_checkpoint:
        state_loaded = load_training_state(resume_checkpoint, optimizer, scheduler, scaler)
        if not state_loaded:
            print_main(context, "checkpoint 没有训练状态，将保留模型权重但重建优化器")

    start_epoch = resume_step // steps_per_epoch
    steps_inside_epoch = resume_step % steps_per_epoch
    skip_batches = steps_inside_epoch * args.gradient_accumulation_steps
    global_step = resume_step
    optimizer.zero_grad(set_to_none=True)

    print_main(context, f"读取编码数据: {manifest_path}")
    print_main(context, f"读取 tokenizer: {tokenizer_dir}")
    print_main(context, f"输出目录: {output_dir}")
    print_main(context, f"模型: {args.model_size}, 参数量: {count_parameters(unwrap_model(model)):,}")
    print_main(
        context,
        f"设备: {device.type}, world_size={context.world_size}, "
        f"backend={context.backend or 'none'}, mixed_precision={precision}",
    )
    print_main(
        context,
        f"seq_length={seq_length}, train_chunks={train_chunks:,}, "
        f"validation_chunks={validation_chunks:,}",
    )
    print_main(
        context,
        f"batch_size={args.batch_size}, accumulation={args.gradient_accumulation_steps}, "
        f"global_batch={args.batch_size * args.gradient_accumulation_steps * context.world_size}, "
        f"优化步数={total_steps:,}, warmup={warmup_steps:,}",
    )

    started_at = time.time()
    progress = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Pretraining",
        dynamic_ncols=True,
        disable=not context.is_main,
    )
    model.train()

    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        epoch_target_step = min((epoch + 1) * steps_per_epoch, total_steps)
        accumulation_count = 0
        accumulated_loss = 0.0
        skipped = 0

        for input_ids in train_loader:
            if epoch == start_epoch and skipped < skip_batches:
                skipped += 1
                continue

            input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
            should_sync = accumulation_count + 1 >= args.gradient_accumulation_steps
            sync_context = (
                nullcontext()
                if should_sync or not isinstance(model, DDP)
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=dtype is not None,
                ):
                    raw_loss = model(input_ids=input_ids, labels=input_ids).loss
                    scaled_loss = raw_loss / args.gradient_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            accumulation_count += 1
            accumulated_loss += float(raw_loss.detach().cpu())
            if accumulation_count < args.gradient_accumulation_steps:
                continue

            if args.max_grad_norm:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_loss = accumulated_loss / accumulation_count
            step_loss = average_across_ranks(step_loss, context, device)
            accumulation_count = 0
            accumulated_loss = 0.0
            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{step_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if args.log_every and global_step % args.log_every == 0:
                print_main(
                    context,
                    f"step={global_step} loss={step_loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6g}",
                )
            if args.eval_every and global_step % args.eval_every == 0:
                result = evaluate_on_main_rank(
                    model,
                    validation_loader,
                    device,
                    dtype,
                    args.max_eval_batches,
                    context,
                )
                if result:
                    print(
                        f"eval step={global_step} loss={result['loss']:.4f} "
                        f"perplexity={result['perplexity']:.2f} batches={result['batches']}"
                    )
            if global_step % args.save_every == 0:
                if context.is_main:
                    checkpoint = save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        scaler,
                        output_dir,
                        global_step,
                    )
                    print(f"保存 checkpoint: {checkpoint}")
                if context.enabled:
                    dist.barrier()
            if global_step >= epoch_target_step:
                break

        # 最后一小段不足 accumulation 次时也更新，并校正梯度缩放比例。
        if accumulation_count and global_step < epoch_target_step:
            correction = args.gradient_accumulation_steps / accumulation_count
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            if args.max_grad_norm:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            step_loss = average_across_ranks(
                accumulated_loss / accumulation_count,
                context,
                device,
            )
            progress.update(1)
            progress.set_postfix(loss=f"{step_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if global_step >= total_steps:
            break

    progress.close()

    final_eval = evaluate_on_main_rank(
        model,
        validation_loader,
        device,
        dtype,
        args.max_eval_batches,
        context,
    )
    raw_model = unwrap_model(model)
    raw_model.config.use_cache = True
    final_model_dir = output_dir / "final_model"
    if context.is_main:
        final_model_dir.mkdir(parents=True, exist_ok=True)
        raw_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        metadata = {
            "generated_by_step": STEP_ID,
            "generated_by_step_name": STEP_NAME,
            "status": "completed",
            "encoding_manifest": str(manifest_path),
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_json_sha256": manifest.get("tokenizer_json_sha256"),
            "model_size": args.model_size,
            "num_parameters": count_parameters(raw_model),
            "model_config": raw_model.config.to_dict(),
            "seq_length": seq_length,
            "epochs": args.epochs,
            "completed_steps": global_step,
            "batch_size_per_device": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "world_size": context.world_size,
            "global_batch_size": (
                args.batch_size
                * args.gradient_accumulation_steps
                * context.world_size
            ),
            "distributed_backend": context.backend,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "mixed_precision": precision,
            "device": device.type,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "optimizer_state_loaded": state_loaded,
            "final_validation": final_eval,
            "training_seconds": round(time.time() - started_at, 2),
            "final_model_dir": str(final_model_dir),
        }
        metadata_path = output_dir / "training_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("Step 04 完成")
        print(f"完成优化步数: {global_step:,}")
        if final_eval:
            print(
                f"最终验证: loss={final_eval['loss']:.4f}, "
                f"perplexity={final_eval['perplexity']:.2f}"
            )
        print(f"最终模型: {final_model_dir}")
    if context.enabled:
        dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 04: pretrain a small/medium/large Atman causal LM"
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="small")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=0, help="0 表示按 epochs 跑完")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=DEFAULT_MAX_EVAL_BATCHES)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto、cuda、mps 或 cpu")
    # 新版 torchrun 使用环境变量；保留参数形式以兼容会传入该参数的启动器。
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "bf16", "fp16", "no"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    preset = TRAINING_PRESETS[args.model_size]
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = preset["gradient_accumulation_steps"]
    return args


if __name__ == "__main__":
    try:
        pretrain(parse_args())
    finally:
        cleanup_distributed()
