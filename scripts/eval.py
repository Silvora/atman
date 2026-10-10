#!/usr/bin/env python3
"""v1 基座模型评测入口。

评测脚本完成两件事：

1. 在独立验证集上计算平均交叉熵和 Perplexity；
2. 使用给定 prompt 做一次简单的自回归生成，检查模型是否能正常产生文本。

脚本只读取 checkpoint，不会修改模型参数。验证集默认评测 YAML 中配置的
``eval_batches`` 个 batch；传入 ``--eval-batches 0`` 可以评测完整验证集。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

# 允许从项目根目录或当前 scripts 目录启动。
V1_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = V1_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model import DecoderOnlyTransformer, ModelConfig  # noqa: E402
from train import (  # noqa: E402
    load_training_config,
    make_loader,
    mps_available,
    precision_settings,
    prepare_data,
)
from utils.config import resolve_v1_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析评测参数；模型结构和数据路径仍由 YAML 管理。"""

    parser = argparse.ArgumentParser(description="Evaluate a v1 decoder-only checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="模型配置，例如 configs/small.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评测 checkpoint")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="评测设备；macOS Apple Silicon 可使用 mps",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="验证 batch 数；0 表示完整验证集，默认读取 YAML 的 training.eval_batches",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="临时覆盖验证 batch size")
    parser.add_argument("--prompt", type=str, default=None, help="可选：生成测试 prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="最多生成多少个新 token")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度；0 表示贪心解码")
    parser.add_argument("--top-k", type=int, default=50, help="采样时保留概率最高的 k 个 token；0 表示不限制")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON 评测报告路径；默认写入 outputs/<model>_eval.json",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """选择单进程评测设备。评测不使用 torchrun/DDP。"""

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not mps_available():
            raise RuntimeError("--device mps was requested but MPS is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, model: DecoderOnlyTransformer, device: torch.device) -> dict[str, Any]:
    """加载训练保存的 checkpoint，并兼容 DDP 可能留下的 ``module.`` 前缀。"""

    checkpoint_path = path if path.is_absolute() else resolve_v1_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # 兼容较旧的 PyTorch。
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid checkpoint format: {checkpoint_path}")

    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint model state is not a mapping: {checkpoint_path}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


@torch.inference_mode()
def evaluate_validation(
    model: DecoderOnlyTransformer,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    max_batches: int,
) -> tuple[float, float, int, int]:
    """计算按 token 数加权的平均 loss 和 Perplexity。"""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    for batch_index, (input_ids, labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type="mps", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "mps"
            else torch.autocast(device_type="cpu", dtype=autocast_dtype)
            if autocast_dtype is not None and device.type == "cpu"
            else torch.no_grad()
        )
        with context:
            _, loss = model(input_ids, labels)
        if loss is None:
            raise RuntimeError("model did not return a validation loss")
        count = int(input_ids.numel())
        total_loss += float(loss) * count
        total_tokens += count
        total_batches += 1

    if total_tokens == 0:
        raise RuntimeError("no validation batches were evaluated")
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(min(mean_loss, 20.0))
    return mean_loss, perplexity, total_tokens, total_batches


def load_tokenizer(data_config: dict[str, Any]):
    """加载正式 tokenizer；训练和评测必须使用同一份词表。"""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确提示。
        raise RuntimeError("缺少 tokenizers，请先安装 requirements.txt") from exc
    encoding = data_config.get("encoding", {})
    tokenizer_path = resolve_v1_path(encoding["tokenizer_dir"]) / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer does not exist: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


@torch.inference_mode()
def generate(
    model: DecoderOnlyTransformer,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    """执行简单的 temperature/top-k 自回归生成。"""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = list(encoded.ids)
    if not token_ids:
        raise ValueError("prompt produced no token IDs")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    model.eval()

    for _ in range(max_new_tokens):
        context_ids = token_ids[-model.config.max_seq_len :]
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :].float()
        if temperature == 0:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            next_logits = next_logits / temperature
            if top_k > 0 and top_k < next_logits.size(-1):
                values, indices = torch.topk(next_logits, top_k, dim=-1)
                filtered = torch.full_like(next_logits, float("-inf"))
                next_logits = filtered.scatter(1, indices, values)
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
        token_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    config, data_config = load_training_config(args.config)
    model_config = ModelConfig.from_mapping(config["model"])
    training = dict(config["training"])
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        training["batch_size"] = args.batch_size
    if device.type != "cuda":
        training["pin_memory"] = False

    _, validation_set, _, validation_path = prepare_data(data_config, model_config, training)
    validation_loader, _ = make_loader(
        validation_set,
        int(training["batch_size"]),
        training,
        distributed=False,
        train=False,
    )

    model = DecoderOnlyTransformer(model_config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    precision_name = str(training.get("precision", "fp32"))
    autocast_dtype, _ = precision_settings(precision_name, device)
    eval_batches = args.eval_batches
    if eval_batches is None:
        eval_batches = int(training.get("eval_batches", 100))
    if eval_batches < 0:
        raise ValueError("--eval-batches must be >= 0")

    validation_loss, perplexity, token_count, batch_count = evaluate_validation(
        model, validation_loader, device, autocast_dtype, eval_batches
    )
    result: dict[str, Any] = {
        "checkpoint": str((args.checkpoint if args.checkpoint.is_absolute() else resolve_v1_path(args.checkpoint)).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "model_parameters": model.estimate_parameters(),
        "validation_file": str(validation_path),
        "validation_tokens": token_count,
        "validation_batches": batch_count,
        "validation_loss": validation_loss,
        "validation_perplexity": perplexity,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if args.prompt is not None:
        tokenizer = load_tokenizer(data_config)
        result["prompt"] = args.prompt
        result["generated_text"] = generate(
            model,
            tokenizer,
            args.prompt,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
        print("\n--- 生成结果 ---", flush=True)
        print(result["generated_text"], flush=True)

    report_path = args.report
    if report_path is None:
        model_name = str(config.get("profile", {}).get("name", "model"))
        report_path = V1_ROOT / "outputs" / f"{model_name}_eval.json"
    elif not report_path.is_absolute():
        report_path = resolve_v1_path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n评测报告：{report_path}", flush=True)


if __name__ == "__main__":
    main()
