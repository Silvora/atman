#!/usr/bin/env python3
"""使用正式 Tokenizer 编码文本并打包为定长二进制 token ID。

输入是 preprocess_data.py 生成的 train.jsonl/validation.jsonl，输出是小端
uint16 的扁平二进制文件。每个样本保存 sequence_length + 1 个 token，训练时
可用前 sequence_length 个作为 input、后 sequence_length 个作为 label。

脚本只负责数据编码，不创建模型、不启动训练。正式运行默认拒绝覆盖已有产物；
--max-documents 只能配合显式 --output-dir 做临时冒烟测试。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from src.data.corpus import iter_jsonl_texts  # noqa: E402
from src.utils.config import load_yaml_config, resolve_v1_path  # noqa: E402

try:
    from tokenizers import Tokenizer
except ImportError as exc:  # pragma: no cover - environment error path
    raise RuntimeError(
        "tokenizers is required; install the v1 requirements before running"
    ) from exc


DEFAULT_CONFIG = V1_ROOT / "configs" / "data.yaml"
SUPPORTED_DTYPE = "uint16"


def parse_args() -> argparse.Namespace:
    """定义配置覆盖项和临时测试选项。"""

    parser = argparse.ArgumentParser(
        description="将清洗后的 JSONL 编码为定长 token ID 二进制数据。"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--split",
        choices=("train", "validation", "both"),
        default="both",
        help="编码训练集、验证集或两者（默认：both）",
    )
    parser.add_argument(
        "--tokenizer-dir",
        help="覆盖 data.yaml encoding.tokenizer_dir",
    )
    parser.add_argument(
        "--processed-dir",
        help="覆盖 data.yaml output_dir，便于使用临时处理结果做冒烟测试",
    )
    parser.add_argument(
        "--output-dir",
        help="覆盖 data.yaml encoding.output_dir",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="覆盖 data.yaml encoding.sequence_length",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="每个分区最多处理 N 篇文档，仅用于冒烟测试",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="每处理多少篇文档打印一次进度",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖选定输出文件和编码报告",
    )
    return parser.parse_args()


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需的 YAML/JSON 映射字段。"""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration key {key!r} must be a mapping")
    return value


def read_json(path: Path) -> dict[str, Any]:
    """读取审计报告或 Tokenizer manifest。"""

    if not path.is_file():
        raise FileNotFoundError(f"required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """保存可追溯的编码报告。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_split_paths(
    data_config: dict[str, Any],
    encoding_config: dict[str, Any],
    processed_dir: Path,
    output_dir: Path,
    split: str,
) -> tuple[Path, Path]:
    """返回某个分区的文本输入路径和二进制输出路径。"""

    if split == "train":
        input_path = processed_dir / str(data_config["train_file"])
        output_path = output_dir / str(encoding_config["train_file"])
    elif split == "validation":
        input_path = processed_dir / str(data_config["validation_file"])
        output_path = output_dir / str(encoding_config["validation_file"])
    else:  # pragma: no cover - caller only passes one concrete split
        raise ValueError(f"unsupported split: {split}")
    return input_path, output_path


def tokenizer_eos_id(
    tokenizer: Tokenizer,
    tokenizer_manifest: dict[str, Any],
    append_eos: bool,
) -> int | None:
    """从正式 Tokenizer 中读取 EOS ID，并检查 manifest 与实际词表一致。"""

    if not append_eos:
        return None
    special_tokens = require_mapping(tokenizer_manifest, "special_tokens")
    eos_info = require_mapping(special_tokens, "eos_token")
    eos_token = eos_info.get("token")
    if not isinstance(eos_token, str):
        raise ValueError("tokenizer manifest has no eos_token string")
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None or eos_id != eos_info.get("id"):
        raise ValueError("Tokenizer EOS ID does not match tokenizer manifest")
    if tokenizer.encode(eos_token, add_special_tokens=False).ids != [eos_id]:
        raise ValueError("Tokenizer EOS token is not encoded as one ID")
    return int(eos_id)


def encode_split(
    *,
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    text_field: str,
    sequence_length: int,
    eos_id: int | None,
    max_documents: int,
    progress_every: int,
    dtype: np.dtype[Any],
    vocab_size: int,
) -> dict[str, Any]:
    """流式编码一个分区，并只发布完整的定长样本。"""

    if not input_path.is_file():
        raise FileNotFoundError(f"processed input does not exist: {input_path}")

    block_size = sequence_length + 1
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    token_buffer: list[int] = []
    source_documents = 0
    encoded_documents = 0
    total_tokens = 0
    written_tokens = 0
    maximum_token_id = -1
    started_at = time.time()

    def flush_complete_blocks(handle: Any) -> None:
        """把缓冲区中的完整 block 写入文件，保留末尾不足一个 block 的 token。"""

        nonlocal token_buffer, written_tokens
        complete_tokens = len(token_buffer) - len(token_buffer) % block_size
        if complete_tokens == 0:
            return
        array = np.asarray(token_buffer[:complete_tokens], dtype=dtype)
        array.tofile(handle)
        written_tokens += complete_tokens
        token_buffer = token_buffer[complete_tokens:]

    with temporary_path.open("wb") as output_handle:
        for _, value, _ in iter_jsonl_texts(
            input_path,
            text_field=text_field,
            strict=True,
        ):
            if max_documents and source_documents >= max_documents:
                break
            if not isinstance(value, str):  # defensive check for the strict iterator
                raise AssertionError("strict JSONL iteration returned an error")

            source_documents += 1
            ids = tokenizer.encode(value, add_special_tokens=False).ids
            if eos_id is not None:
                ids.append(eos_id)
            if ids:
                maximum_token_id = max(maximum_token_id, max(ids))
                token_buffer.extend(ids)
                total_tokens += len(ids)
            encoded_documents += 1

            # 限制单次 numpy 转换的规模，避免把整个语料的 token 列表留在内存。
            if len(token_buffer) >= block_size * 512:
                flush_complete_blocks(output_handle)

            if source_documents % progress_every == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                print(
                    f"split={input_path.name} documents={source_documents:,} "
                    f"tokens={total_tokens:,} sequences={written_tokens // block_size:,} "
                    f"speed={source_documents / elapsed:,.0f} docs/s",
                    flush=True,
                )
        flush_complete_blocks(output_handle)

    if maximum_token_id >= vocab_size:
        raise ValueError(
            f"token ID {maximum_token_id} is outside tokenizer vocab size {vocab_size}"
        )

    discarded_tokens = total_tokens - written_tokens
    elapsed_seconds = time.time() - started_at
    os.replace(temporary_path, output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "source_documents": source_documents,
        "encoded_documents": encoded_documents,
        "token_count_before_truncation": total_tokens,
        "written_tokens": written_tokens,
        "discarded_tail_tokens": discarded_tokens,
        "sequence_length": sequence_length,
        "stored_tokens_per_sequence": block_size,
        "sequences": written_tokens // block_size,
        "maximum_token_id": maximum_token_id,
        "bytes": output_path.stat().st_size,
        "elapsed_seconds": elapsed_seconds,
        "documents_per_second": (
            source_documents / elapsed_seconds if elapsed_seconds else None
        ),
    }


def main() -> int:
    """加载配置、编码选定分区并原子写出编码报告。"""

    args = parse_args()
    if args.max_documents < 0:
        raise ValueError("--max-documents must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    config = load_yaml_config(args.config)
    if config.get("schema_version") != 1:
        raise ValueError("only schema_version: 1 is supported")
    encoding_config = require_mapping(config, "encoding")
    sequence_length = int(
        args.sequence_length
        if args.sequence_length is not None
        else encoding_config["sequence_length"]
    )
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if str(encoding_config.get("dtype", SUPPORTED_DTYPE)) != SUPPORTED_DTYPE:
        raise ValueError("only encoding.dtype: uint16 is supported")
    dtype = np.dtype("<u2")

    processed_dir = (
        Path(args.processed_dir).expanduser().resolve()
        if args.processed_dir
        else resolve_v1_path(config["output_dir"])
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_v1_path(encoding_config["output_dir"])
    )
    tokenizer_dir = (
        Path(args.tokenizer_dir).expanduser().resolve()
        if args.tokenizer_dir
        else resolve_v1_path(encoding_config["tokenizer_dir"])
    )
    if args.max_documents and args.output_dir is None:
        raise ValueError(
            "--max-documents requires --output-dir so a partial run cannot replace "
            "formal encoded data"
        )

    report_path = processed_dir / str(config["report_file"])
    audit_report = read_json(report_path)
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_manifest_path = tokenizer_dir / "manifest.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_manifest = read_json(tokenizer_manifest_path)
    vocab_size = int(tokenizer_manifest.get("actual_vocab_size", 0))
    if vocab_size <= 0 or tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match tokenizer manifest")
    eos_id = tokenizer_eos_id(
        tokenizer,
        tokenizer_manifest,
        bool(encoding_config.get("append_eos", True)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "validation") if args.split == "both" else (args.split,)
    output_paths = [
        resolve_split_paths(config, encoding_config, processed_dir, output_dir, split)[1]
        for split in splits
    ]
    encoding_report_path = output_dir / str(encoding_config["report_file"])
    protected = (*output_paths, encoding_report_path)
    temporary = tuple(
        path.with_suffix(path.suffix + ".tmp") for path in (*output_paths, encoding_report_path)
    )
    existing = [path for path in (*protected, *temporary) if path.exists()]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"encoded outputs already exist: {joined}; use --force to replace them"
        )
    if args.force:
        for path in (*protected, *temporary):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    text_field = str(config.get("text_field", "text"))
    results: dict[str, Any] = {}
    try:
        for split in splits:
            input_path, output_path = resolve_split_paths(
                config,
                encoding_config,
                processed_dir,
                output_dir,
                split,
            )
            print(f"encoding split={split} input={input_path}", flush=True)
            results[split] = encode_split(
                tokenizer=tokenizer,
                input_path=input_path,
                output_path=output_path,
                text_field=text_field,
                sequence_length=sequence_length,
                eos_id=eos_id,
                max_documents=args.max_documents,
                progress_every=args.progress_every,
                dtype=dtype,
                vocab_size=vocab_size,
            )
    except Exception:
        for path in (*output_paths, *temporary):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    report = {
        "schema_version": 1,
        "format": "flat little-endian uint16 token IDs",
        "sequence_length": sequence_length,
        "stored_tokens_per_sequence": sequence_length + 1,
        "append_eos": eos_id is not None,
        "eos_token_id": eos_id,
        "text_field": text_field,
        "partial_run": bool(args.max_documents),
        "max_documents_per_split": args.max_documents or None,
        "tokenizer": {
            "directory": str(tokenizer_dir),
            "manifest": str(tokenizer_manifest_path),
            "vocab_size": vocab_size,
        },
        "processed": {
            "directory": str(processed_dir),
            "audit_report": str(report_path),
            "audit_counts": audit_report.get("counts"),
        },
        "splits": results,
    }
    temporary_report = encoding_report_path.with_suffix(
        encoding_report_path.suffix + ".tmp"
    )
    write_json(temporary_report, report)
    os.replace(temporary_report, encoding_report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
