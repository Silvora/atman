"""Step 08: export a standalone model and verify that it can be reloaded.

可导出第 4 步基础模型、第 6 步 SFT 模型，或把第 7 步 LoRA-DPO adapter 合并
进 SFT 基础权重。默认输出 Hugging Face 标准目录；可选调用 llama.cpp 转换器生成
GGUF。所有产物只写入本步骤的 output 目录。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_ID = "08_export_deployment"
STEP_NAME = "Export and deployment"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_OUTPUT_ROOT = STEP_DIR / "output"


def default_base_model(model_size: str) -> Path:
    return PROJECT_ROOT / "04_pretraining" / "output" / model_size / "final_model"


def default_sft_model(model_size: str) -> Path:
    return PROJECT_ROOT / "06_sft" / "output" / model_size / "final_model"


def default_adapter(model_size: str) -> Path:
    return PROJECT_ROOT / "07_lora_dpo" / "output" / model_size / "adapter"


def prepare_output(path: Path, overwrite: bool) -> None:
    """默认拒绝覆盖；只有显式 --overwrite 才会替换旧导出目录。"""
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"导出目录已存在: {path}，可使用 --overwrite 明确覆盖")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_source_model(args: argparse.Namespace):
    """加载完整模型，或把 LoRA adapter 合并为不依赖 PEFT 的普通模型。"""
    if args.source == "base":
        model_path = (
            Path(args.model_path).expanduser().resolve()
            if args.model_path
            else default_base_model(args.model_size)
        )
        adapter_path = None
    else:
        model_path = (
            Path(args.model_path).expanduser().resolve()
            if args.model_path
            else default_sft_model(args.model_size)
        )
        adapter_path = None
        if args.source == "lora-dpo":
            adapter_path = (
                Path(args.adapter_path).expanduser().resolve()
                if args.adapter_path
                else default_adapter(args.model_size)
            )

    if not model_path.is_dir():
        raise FileNotFoundError(f"源模型不存在: {model_path}")
    tokenizer_source = adapter_path if adapter_path and (adapter_path / "tokenizer.json").is_file() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"LoRA-DPO adapter 不存在: {adapter_path}")
        try:
            from peft import PeftModel
        except ModuleNotFoundError as error:
            raise SystemExit("合并 LoRA 需要 PEFT，请先运行: pip install peft") from error
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        # merge_and_unload 返回新的普通模型，不再需要 adapter 才能推理。
        model = model.merge_and_unload()

    model.config.use_cache = True
    return model, tokenizer, model_path, adapter_path


@torch.no_grad()
def verify_export(export_dir: Path, prompt: str, max_new_tokens: int) -> dict:
    """从磁盘重新加载导出物，避免只验证内存中的原模型。"""
    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(export_dir, local_files_only=True)
    model.eval()
    max_context = int(
        getattr(model.config, "n_positions", None)
        or getattr(model.config, "max_position_embeddings", 1_024)
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, max_context - max_new_tokens),
    )
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return {
        "status": "passed",
        "model_class": model.__class__.__name__,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocab_size": len(tokenizer),
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
    }


def convert_to_gguf(export_dir: Path, converter_text: str, outtype: str) -> Path:
    """可选调用用户已有的 llama.cpp convert_hf_to_gguf.py。"""
    converter = Path(converter_text).expanduser().resolve()
    if converter.is_dir():
        converter = converter / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"找不到 GGUF 转换器: {converter}")
    gguf_path = export_dir.parent / f"atman-{outtype}.gguf"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(export_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            outtype,
        ],
        check=True,
    )
    if not gguf_path.is_file():
        raise RuntimeError("转换器执行结束，但没有生成 GGUF 文件")
    return gguf_path


def export_model(args: argparse.Namespace) -> None:
    """导出独立 Hugging Face 模型，按需再生成 GGUF。"""
    started_at = time.time()
    model, tokenizer, model_path, adapter_path = load_source_model(args)
    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / f"{args.model_size}_{args.source.replace('-', '_')}"
    )
    export_dir = output_dir / "huggingface"
    prepare_output(output_dir, args.overwrite)

    print(f"源模型: {model_path}")
    if adapter_path:
        print(f"合并 adapter: {adapter_path}")
    print(f"导出目录: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        export_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(export_dir)

    verification = None
    if not args.skip_verification:
        verification = verify_export(export_dir, args.test_prompt, args.test_new_tokens)
        print(f"重新加载验证: {verification['status']}")

    gguf_path = None
    if args.gguf_converter:
        gguf_path = convert_to_gguf(export_dir, args.gguf_converter, args.gguf_outtype)
        print(f"GGUF 文件: {gguf_path}")

    manifest = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "status": "completed",
        "source": args.source,
        "model_size": args.model_size,
        "source_model": str(model_path),
        "source_adapter": str(adapter_path) if adapter_path else None,
        "adapter_merged": adapter_path is not None,
        "huggingface_dir": str(export_dir),
        "gguf_file": str(gguf_path) if gguf_path else None,
        "verification": verification,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Step 08 完成")
    print(f"导出清单: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 08: merge adapters, export, and reload-check an Atman model"
    )
    parser.add_argument(
        "--source",
        choices=("base", "sft", "lora-dpo"),
        default="lora-dpo",
    )
    parser.add_argument("--model-size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--test-prompt", default="人工智能的未来")
    parser.add_argument("--test-new-tokens", type=int, default=16)
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument(
        "--gguf-converter",
        default=None,
        help="llama.cpp 目录或 convert_hf_to_gguf.py 路径；不填则只导出 HF 格式",
    )
    parser.add_argument(
        "--gguf-outtype",
        choices=("f32", "f16", "bf16", "q8_0", "auto"),
        default="f16",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.test_new_tokens <= 0:
        parser.error("test_new_tokens 必须大于 0")
    if args.adapter_path and args.source != "lora-dpo":
        parser.error("--adapter-path 只在 --source lora-dpo 时有效")
    return args


if __name__ == "__main__":
    export_model(parse_args())
