"""Step 05: evaluate a pretrained Atman model and generate sample text.

本脚本读取第 4 步的 final_model 和第 3 步固定验证集，计算 loss/perplexity，
再对一组固定提示词生成文本。评估结果只写入本步骤的 output 目录，不修改模型。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_ID = "05_generation_evaluation"
STEP_NAME = "Generation and evaluation"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_ENCODING_MANIFEST = PROJECT_ROOT / "03_encoding" / "output" / "manifest.json"
DEFAULT_OUTPUT_ROOT = STEP_DIR / "output"
DEFAULT_PROMPTS = [
    "人工智能的未来",
    "中国古代科技的发展",
    "学习一门新知识时，最重要的是",
    "请解释什么是神经网络：",
]


class ValidationShardDataset(IterableDataset):
    """逐个 mmap 验证集分片，避免把全部 token 载入内存。"""

    def __init__(self, shard_paths: list[Path]) -> None:
        self.shard_paths = shard_paths

    def __iter__(self):
        for shard_path in self.shard_paths:
            token_array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            if token_array.ndim != 2:
                raise ValueError(f"验证分片不是二维数组: {shard_path}")
            for row in token_array:
                # mmap 为只读数组；复制后转为 embedding 需要的 int64。
                yield torch.from_numpy(np.array(row, dtype=np.int64, copy=True))


def resolve_device(requested: str) -> torch.device:
    """auto 按 CUDA、MPS、CPU 的顺序选择推理设备。"""
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


def resolve_model_path(args: argparse.Namespace) -> Path:
    """允许覆盖模型路径；默认读取第 4 步对应档位的最终模型。"""
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()
    return (
        PROJECT_ROOT
        / "04_pretraining"
        / "output"
        / args.model_size
        / "final_model"
    )


def load_validation_shards(manifest_path: Path) -> tuple[dict, list[Path]]:
    """校验第 3 步 manifest，并解析 validation 分片路径。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到第 3 步 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generated_by_step") != "03_encoding":
        raise ValueError("输入 manifest 不是第 3 步生成的")
    if manifest.get("status") != "completed":
        raise ValueError("第 3 步编码尚未完成")

    shards = [
        manifest_path.parent / item["file"]
        for item in manifest.get("splits", {}).get("validation", {}).get("shards", [])
    ]
    if not shards:
        raise ValueError("第 3 步没有 validation 分片，无法进行独立评估")
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"验证分片不存在: {missing[0]}")
    return manifest, shards


def load_prompts(path_text: str | None) -> list[str]:
    """读取纯文本或 JSONL 提示词；未指定时使用固定内置样例。"""
    if not path_text:
        return DEFAULT_PROMPTS
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"提示词文件不存在: {path}")

    prompts = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"提示词 JSON 错误: {path}:{line_number}") from error
                prompt = record.get("prompt")
            else:
                prompt = line
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"提示词为空或不是字符串: {path}:{line_number}")
            prompts.append(prompt.strip())
    if not prompts:
        raise ValueError("提示词文件没有有效内容")
    return prompts


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device, max_batches: int) -> dict:
    """计算 token 加权平均 loss，并转换为 perplexity。"""
    model.eval()
    loss_sum = 0.0
    token_count = 0
    completed_batches = 0
    progress = tqdm(dataloader, desc="Evaluating", dynamic_ncols=True)
    for batch_index, input_ids in enumerate(progress, start=1):
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        outputs = model(input_ids=input_ids, labels=input_ids)
        # 因果语言模型每条序列的第一个 token 没有预测目标。
        predicted_tokens = input_ids.numel() - input_ids.shape[0]
        loss_sum += float(outputs.loss.detach().cpu()) * predicted_tokens
        token_count += predicted_tokens
        completed_batches += 1
        progress.set_postfix(loss=f"{outputs.loss.item():.4f}")
        if max_batches and batch_index >= max_batches:
            break
    progress.close()
    if token_count == 0:
        raise ValueError("验证集没有可评估 token")
    loss = loss_sum / token_count
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 20 else float("inf"),
        "evaluated_tokens": token_count,
        "evaluated_batches": completed_batches,
    }


@torch.no_grad()
def generate_samples(model, tokenizer, prompts: list[str], args, device) -> list[dict]:
    """对固定提示词生成续写，保存完整参数以便之后横向比较模型。"""
    model.eval()
    max_context = int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", 1_024)
    )
    max_prompt_length = max(1, max_context - args.max_new_tokens)
    results = []

    for prompt in tqdm(prompts, desc="Generating", dynamic_ncols=True):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "repetition_penalty": args.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            generate_kwargs.update(
                temperature=args.temperature,
                top_p=args.top_p,
            )
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        completion_ids = generated_ids[0, input_ids.shape[1] :]
        results.append(
            {
                "prompt": prompt,
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "full_text": tokenizer.decode(generated_ids[0], skip_special_tokens=True),
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_tokens": int(completion_ids.shape[0]),
            }
        )
    return results


def evaluate_base_model(args: argparse.Namespace) -> None:
    """运行第 5 步并写出结构化评估结果。"""
    started_at = time.time()
    torch.manual_seed(args.seed)
    model_path = resolve_model_path(args)
    manifest_path = Path(args.encoding_manifest).expanduser().resolve()
    output_dir = Path(args.output_root).expanduser().resolve() / args.model_size
    if not model_path.is_dir():
        raise FileNotFoundError(f"找不到第 4 步最终模型: {model_path}")

    manifest, validation_paths = load_validation_shards(manifest_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.loss_type = "ForCausalLM"
    device = resolve_device(args.device)
    model.to(device)

    dataset = ValidationShardDataset(validation_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    prompts = load_prompts(args.prompts_file)

    print(f"读取模型: {model_path}")
    print(f"读取验证集: {manifest_path}")
    print(f"设备: {device}, 提示词: {len(prompts)} 条")
    metrics = evaluate(model, dataloader, device, args.max_eval_batches)
    generations = generate_samples(model, tokenizer, prompts, args, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"
    with generations_path.open("w", encoding="utf-8") as output_file:
        for record in generations:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "status": "completed",
        "model_size": args.model_size,
        "model_path": str(model_path),
        "encoding_manifest": str(manifest_path),
        "seq_length": manifest.get("seq_length"),
        "device": str(device),
        "metrics": metrics,
        "generation_config": {
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
        },
        "generations_file": generations_path.name,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Step 05 完成")
    print(f"验证 loss: {metrics['loss']:.4f}")
    print(f"验证 perplexity: {metrics['perplexity']:.2f}")
    print(f"评估报告: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 05: evaluate and generate with a pretrained Atman model"
    )
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--encoding-manifest", default=str(DEFAULT_ENCODING_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto、cuda、mps 或 cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("--batch-size 和 --max-new-tokens 必须大于 0")
    if args.max_eval_batches < 0:
        parser.error("--max-eval-batches 不能小于 0")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature 必须大于 0，top_p 必须在 (0, 1] 范围内")
    return args


if __name__ == "__main__":
    evaluate_base_model(parse_args())
