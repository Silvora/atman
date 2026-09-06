"""Step 07: align the SFT model with DPO while training LoRA adapters.

本步骤把 LoRA 作为参数高效训练方式，把 DPO 作为偏好优化目标：基础权重保持冻结，
只训练小型适配器。输入 JSONL 每行至少包含 prompt、chosen、rejected 三个字段。
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path


STEP_ID = "07_lora_dpo"
STEP_NAME = "LoRA DPO alignment"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_OUTPUT_ROOT = STEP_DIR / "output"


def resolve_sft_model(args: argparse.Namespace) -> Path:
    """默认读取第 6 步对应档位的完整 SFT 模型。"""
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()
    return PROJECT_ROOT / "06_sft" / "output" / args.model_size / "final_model"


def render_messages(messages: list[dict]) -> str:
    """把对话数组转换为无需 chat_template 的稳定纯文本格式。"""
    parts = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"<{role}>\n{content}")
    return "\n".join(parts)


def as_text(value) -> str:
    """偏好数据既可使用字符串，也可使用 role/content 消息数组。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return render_messages(value).strip()
    return ""


def normalize_preference(record: dict) -> dict[str, str]:
    """统一常见的 prompt/chosen/rejected 与 instruction 格式。"""
    prompt = as_text(record.get("prompt"))
    if not prompt and record.get("instruction"):
        prompt = f"用户：{str(record['instruction']).strip()}"
        extra_input = str(record.get("input", "")).strip()
        if extra_input:
            prompt += f"\n补充信息：{extra_input}"
        prompt += "\n助手："
    chosen = as_text(record.get("chosen"))
    rejected = as_text(record.get("rejected"))
    if not prompt or not chosen or not rejected:
        raise ValueError("偏好样本必须包含非空的 prompt、chosen、rejected")
    if chosen == rejected:
        raise ValueError("chosen 与 rejected 不能完全相同")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def load_preference_dataset(path: Path, max_samples: int):
    """读取本地 JSONL，并构建 TRL 使用的 Dataset。"""
    try:
        from datasets import Dataset
    except ModuleNotFoundError as error:
        raise SystemExit("缺少 datasets，请先运行: pip install datasets") from error

    if not path.is_file():
        raise FileNotFoundError(f"偏好数据不存在: {path}")
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if max_samples and len(records) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"偏好数据 JSON 错误: {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"偏好样本必须是 JSON 对象: {path}:{line_number}")
            try:
                records.append(normalize_preference(record))
            except ValueError as error:
                raise ValueError(f"偏好样本错误: {path}:{line_number}: {error}") from error
    if not records:
        raise ValueError("偏好 JSONL 没有有效样本")
    return Dataset.from_list(records)


def supported_kwargs(callable_object, values: dict) -> tuple[dict, list[str]]:
    """过滤不同 TRL 版本间已经改名或尚未提供的配置项。"""
    parameters = inspect.signature(callable_object).parameters
    accepted = {key: value for key, value in values.items() if key in parameters}
    skipped = [key for key in values if key not in parameters]
    return accepted, skipped


def train_lora_dpo(args: argparse.Namespace) -> None:
    """使用 TRL DPOTrainer 和 PEFT LoraConfig 训练偏好适配器。"""
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ModuleNotFoundError as error:
        raise SystemExit(
            "第 7 步需要 PEFT、TRL 和 Accelerate，请先运行: "
            "pip install peft trl accelerate"
        ) from error

    started_at = time.time()
    model_path = resolve_sft_model(args)
    data_path = Path(args.data_file).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / args.model_size
    adapter_dir = output_dir / "adapter"
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到第 6 步 SFT 模型: {model_path}")
    if (output_dir / "training_metadata.json").exists():
        raise FileExistsError(f"第 7 步输出已存在，请确认后手动清理: {output_dir}")

    dataset = load_preference_dataset(data_path, args.max_samples)
    validation_size = int(len(dataset) * args.validation_ratio)
    if args.validation_ratio > 0 and validation_size == 0 and len(dataset) > 1:
        validation_size = 1
    if validation_size:
        split = dataset.train_test_split(
            test_size=validation_size,
            seed=args.seed,
            shuffle=True,
        )
        train_dataset = split["train"]
        validation_dataset = split["test"]
    else:
        train_dataset = dataset
        validation_dataset = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.config.use_cache = False
    model.loss_type = "ForCausalLM"

    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    use_bf16 = args.mixed_precision == "bf16" or (
        args.mixed_precision == "auto"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = args.mixed_precision == "fp16" or (
        args.mixed_precision == "auto"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    )
    config_values = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if validation_dataset is not None else "no",
        "evaluation_strategy": "epoch" if validation_dataset is not None else "no",
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    config_kwargs, skipped_config = supported_kwargs(DPOConfig, config_values)
    training_args = DPOConfig(**config_kwargs)

    trainer_values = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
    }
    trainer_kwargs, skipped_trainer = supported_kwargs(DPOTrainer, trainer_values)

    print(f"读取 SFT 模型: {model_path}")
    print(f"读取偏好数据: {data_path}")
    print(
        f"样本: train={len(train_dataset):,}, "
        f"validation={len(validation_dataset) if validation_dataset is not None else 0:,}"
    )
    print(f"LoRA: rank={args.lora_rank}, targets={target_modules}")
    if skipped_config or skipped_trainer:
        print(f"当前 TRL 版本忽略的不兼容参数: {sorted(set(skipped_config + skipped_trainer))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = DPOTrainer(**trainer_kwargs)
    train_result = trainer.train()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    metadata = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "status": "completed",
        "base_sft_model": str(model_path),
        "preference_data": str(data_path),
        "model_size": args.model_size,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "dpo_beta": args.beta,
        "trainable_parameters": trainable,
        "total_parameters_with_adapter": total,
        "train_metrics": train_result.metrics,
        "adapter_dir": str(adapter_dir),
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("Step 07 完成")
    print(f"可训练参数: {trainable:,}/{total:,} ({trainable / total:.2%})")
    print(f"LoRA-DPO adapter: {adapter_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 07: train a LoRA adapter with the DPO preference objective"
    )
    parser.add_argument("--data-file", required=True, help="含 prompt/chosen/rejected 的 JSONL")
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="c_attn,c_proj,c_fc")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "bf16", "fp16", "no"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("epochs、batch_size 和 gradient_accumulation_steps 必须大于 0")
    if args.max_samples < 0 or args.max_steps == 0 or args.max_steps < -1:
        parser.error("max_samples 不能小于 0；max_steps 应为 -1 或正整数")
    if not 0 <= args.validation_ratio < 1:
        parser.error("validation_ratio 必须在 [0, 1) 范围内")
    if args.max_prompt_length >= args.max_length:
        parser.error("max_prompt_length 必须小于 max_length")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        parser.error("LoRA rank 和 alpha 必须大于 0")
    if not 0 <= args.lora_dropout < 1:
        parser.error("lora_dropout 必须在 [0, 1) 范围内")
    return args


if __name__ == "__main__":
    train_lora_dpo(parse_args())
