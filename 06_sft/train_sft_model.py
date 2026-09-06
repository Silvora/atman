"""Step 06: supervised fine-tuning for instruction following.

输入是公开指令数据的本地 JSONL 文件。脚本只对回答 token 计算 loss，提示词部分
使用 -100 屏蔽，避免把“复述用户问题”当作学习目标。输出是完整 SFT 模型。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


STEP_ID = "06_sft"
STEP_NAME = "Supervised fine-tuning"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_OUTPUT_ROOT = STEP_DIR / "output"


def resolve_device(requested: str) -> torch.device:
    """auto 按 CUDA、MPS、CPU 的顺序选择训练设备。"""
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


def resolve_precision(requested: str, device: torch.device):
    """返回混合精度名称和 autocast dtype。"""
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
            raise ValueError("当前设备不支持 bf16")
        return precision, torch.bfloat16
    if precision == "fp16":
        if device.type not in {"cuda", "mps"}:
            raise ValueError("fp16 只用于 CUDA 或 MPS")
        return precision, torch.float16
    return "no", None


def build_grad_scaler(device: torch.device, precision: str):
    """fp16 使用损失缩放，bf16 和 fp32 不需要。"""
    enabled = precision == "fp16" and device.type in {"cuda", "mps"}
    try:
        return torch.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError, RuntimeError):
        return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def resolve_base_model(args: argparse.Namespace) -> Path:
    """默认从第 4 步读取对应档位的 final_model。"""
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()
    return (
        PROJECT_ROOT
        / "04_pretraining"
        / "output"
        / args.model_size
        / "final_model"
    )


def normalize_record(record: dict) -> tuple[str, str]:
    """兼容常见 instruction/output、prompt/response 和 messages 格式。"""
    if isinstance(record.get("messages"), list):
        messages = record["messages"]
        assistant_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "assistant"
            ),
            None,
        )
        if assistant_index is None:
            raise ValueError("messages 中没有 assistant 回答")
        response = messages[assistant_index].get("content", "")
        prompt_parts = []
        for message in messages[:assistant_index]:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if content:
                prompt_parts.append(f"<{role}>\n{content}")
        prompt = "\n".join(prompt_parts) + "\n<assistant>\n"
    elif "instruction" in record and "output" in record:
        instruction = str(record.get("instruction", "")).strip()
        extra_input = str(record.get("input", "")).strip()
        prompt = f"用户：{instruction}"
        if extra_input:
            prompt += f"\n补充信息：{extra_input}"
        prompt += "\n助手："
        response = record.get("output", "")
    else:
        prompt = record.get("prompt", "")
        response = record.get("response", record.get("answer", ""))

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("样本缺少有效 prompt/instruction")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("样本缺少有效 response/output")
    return prompt, response.strip()


class JsonlInstructionDataset(Dataset):
    """只保存每行的字节偏移，大型 JSONL 无需一次全部载入内存。"""

    def __init__(self, path: Path, max_samples: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"SFT 数据不存在: {path}")
        self.path = path
        self.offsets = []
        self._file = None
        with path.open("rb") as input_file:
            while True:
                offset = input_file.tell()
                line = input_file.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if max_samples and len(self.offsets) >= max_samples:
                        break
        if not self.offsets:
            raise ValueError("SFT JSONL 没有有效行")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __getitem__(self, index: int) -> tuple[str, str]:
        if self._file is None:
            self._file = self.path.open("rb")
        self._file.seek(self.offsets[index])
        line = self._file.readline().decode("utf-8")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"SFT JSON 错误，样本索引: {index}") from error
        if not isinstance(record, dict):
            raise ValueError(f"SFT 样本必须是 JSON 对象，样本索引: {index}")
        return normalize_record(record)


class SFTCollator:
    """动态分词和 padding，并屏蔽提示词与 padding 对应的标签。"""

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        encoded_examples = []
        for prompt, response in examples:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)
            response_ids.append(self.tokenizer.eos_token_id)

            # 优先保留监督信号；长度不足时从左侧裁掉较早的 prompt。
            if len(response_ids) >= self.max_length:
                response_ids = response_ids[: self.max_length]
                response_ids[-1] = self.tokenizer.eos_token_id
                prompt_ids = []
            else:
                prompt_budget = self.max_length - len(response_ids)
                prompt_ids = prompt_ids[-prompt_budget:]

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids
            encoded_examples.append((input_ids, labels))

        batch_length = max(len(input_ids) for input_ids, _ in encoded_examples)
        input_batch = []
        label_batch = []
        attention_batch = []
        for input_ids, labels in encoded_examples:
            padding = batch_length - len(input_ids)
            input_batch.append(input_ids + [self.tokenizer.pad_token_id] * padding)
            label_batch.append(labels + [-100] * padding)
            attention_batch.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_batch, dtype=torch.long),
            "labels": torch.tensor(label_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
        }


@torch.no_grad()
def evaluate(model, dataloader, device, dtype, max_batches: int) -> dict | None:
    """在固定的 SFT 验证子集上计算平均 loss。"""
    if dataloader is None:
        return None
    model.eval()
    losses = []
    for batch_index, batch in enumerate(dataloader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
        if max_batches and batch_index >= max_batches:
            break
    model.train()
    if not losses:
        return None
    return {"loss": sum(losses) / len(losses), "batches": len(losses)}


def train_sft_model(args: argparse.Namespace) -> None:
    """执行全参数监督微调并保存完整模型。"""
    started_at = time.time()
    torch.manual_seed(args.seed)
    model_path = resolve_base_model(args)
    data_path = Path(args.data_file).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / args.model_size
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到第 4 步模型: {model_path}")
    if (output_dir / "training_metadata.json").exists():
        raise FileExistsError(f"第 6 步输出已存在，请确认后手动清理: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须包含 pad_token 和 eos_token")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.config.use_cache = False
    model.loss_type = "ForCausalLM"
    device = resolve_device(args.device)
    precision, dtype = resolve_precision(args.mixed_precision, device)
    model.to(device)

    dataset = JsonlInstructionDataset(data_path, args.max_samples)
    validation_size = int(len(dataset) * args.validation_ratio)
    if args.validation_ratio > 0 and validation_size == 0 and len(dataset) > 1:
        validation_size = 1
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("SFT 训练集为空，请减少 validation_ratio")
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collator = SFTCollator(tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_size
        else None
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    full_steps = steps_per_epoch * args.epochs
    total_steps = min(full_steps, args.max_steps) if args.max_steps else full_steps
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

    print(f"读取基础模型: {model_path}")
    print(f"读取 SFT 数据: {data_path}")
    print(f"样本: train={train_size:,}, validation={validation_size:,}")
    print(
        f"设备={device}, precision={precision}, batch={args.batch_size}, "
        f"accumulation={args.gradient_accumulation_steps}, steps={total_steps:,}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=total_steps, desc="SFT", dynamic_ncols=True)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for _epoch in range(args.epochs):
        accumulated = 0
        accumulated_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                raw_loss = model(**batch).loss
                loss = raw_loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated += 1
            accumulated_loss += float(raw_loss.detach().cpu())

            if accumulated < args.gradient_accumulation_steps:
                continue

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
            progress.update(1)
            progress.set_postfix(
                loss=f"{accumulated_loss / accumulated:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            accumulated = 0
            accumulated_loss = 0.0
            if global_step >= total_steps:
                break

        # 每轮尾部不足梯度累积次数时，仍完成一次参数更新。
        if accumulated and global_step < total_steps:
            correction = args.gradient_accumulation_steps / accumulated
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
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
            progress.update(1)
        if global_step >= total_steps:
            break
    progress.close()

    validation = evaluate(
        model,
        validation_loader,
        device,
        dtype,
        args.max_eval_batches,
    )
    model.config.use_cache = True
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_model_dir)

    metadata = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "status": "completed",
        "base_model": str(model_path),
        "data_file": str(data_path),
        "model_size": args.model_size,
        "train_samples": train_size,
        "validation_samples": validation_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "completed_steps": global_step,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "mixed_precision": precision,
        "final_validation": validation,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "final_model_dir": str(final_model_dir),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Step 06 完成")
    if validation:
        print(f"验证 loss: {validation['loss']:.4f}")
    print(f"最终模型: {final_model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 06: full-parameter supervised fine-tuning"
    )
    parser.add_argument("--data-file", required=True, help="公开指令数据的本地 JSONL 文件")
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "bf16", "fp16", "no"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positive = (
        args.epochs,
        args.max_length,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.max_grad_norm,
    )
    if any(value <= 0 for value in positive):
        parser.error("epochs、长度、batch、梯度累积、学习率和梯度范数必须大于 0")
    if args.max_steps < 0 or args.max_samples < 0 or args.max_eval_batches < 0:
        parser.error("max_steps、max_samples 和 max_eval_batches 不能小于 0")
    if not 0 <= args.validation_ratio < 1:
        parser.error("validation_ratio 必须在 [0, 1) 范围内")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup_ratio 必须在 [0, 1) 范围内")
    return args


if __name__ == "__main__":
    train_sft_model(parse_args())
