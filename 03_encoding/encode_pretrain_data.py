import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


STEP_ID = "03_encoding"
STEP_NAME = "Encoding"
DEFAULT_INPUT_PATH = "output/01_public_corpus/public_corpus.jsonl"
DEFAULT_TOKENIZER_DIR = "output/02_tokenizer/atman_tokenizer"
DEFAULT_OUTPUT_DIR = "output/03_encoding"


def iter_jsonl_texts(input_path: Path, text_field: str, max_samples: int):
    """逐行读取 JSONL，避免一次性把整个语料文件读进内存。"""
    kept = 0
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if max_samples > 0 and kept >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON") from exc

            text = record.get(text_field, "")
            if not isinstance(text, str) or not text.strip():
                continue

            kept += 1
            yield kept, text


def save_shard(output_dir: Path, shard_index: int, chunks, metadata):
    """保存一个训练数据分片。

    每个 shard 都保存成独立 .pt 文件，后续训练可以按文件逐个加载。
    """
    shard_name = f"pretrain_data_{shard_index:05d}.pt"
    shard_path = output_dir / shard_name
    input_ids = torch.tensor(chunks, dtype=torch.long)
    torch.save(
        {
            "input_ids": input_ids,
            "metadata": {
                **metadata,
                "shard_index": shard_index,
                "num_chunks": len(input_ids),
                "num_tokens": int(input_ids.numel()),
            },
        },
        shard_path,
    )
    return {
        "file": shard_name,
        "num_chunks": len(input_ids),
        "num_tokens": int(input_ids.numel()),
    }


def encode_pretrain_data(args):
    input_path = Path(args.input_path)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到第 1 步输出语料: {input_path.resolve()}")
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"找不到第 2 步 tokenizer: {tokenizer_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 缺少 eos_token，无法安全分隔文章")

    metadata = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "input_path": str(input_path),
        "tokenizer_dir": str(tokenizer_dir),
        "output_dir": str(output_dir),
        "seq_length": args.seq_length,
        "shard_size": args.shard_size,
        "vocab_size": tokenizer.vocab_size,
    }

    token_buffer = []
    shard_chunks = []
    shards = []
    total_documents = 0
    total_chunks = 0
    total_tokens = 0
    shard_index = 0

    progress = tqdm(
        iter_jsonl_texts(input_path, args.text_field, args.max_samples),
        desc="Encoding corpus",
        dynamic_ncols=True,
    )

    for total_documents, text in progress:
        # 不额外加入特殊 token，只在每篇文章末尾加 eos，表示文档边界。
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            token_buffer.extend(token_ids)
            token_buffer.append(tokenizer.eos_token_id)

        # 使用连续 buffer 切块，能最大化利用每篇文章末尾不足 seq_length 的 token。
        while len(token_buffer) >= args.seq_length:
            shard_chunks.append(token_buffer[:args.seq_length])
            token_buffer = token_buffer[args.seq_length:]

            if len(shard_chunks) >= args.shard_size:
                shard = save_shard(output_dir, shard_index, shard_chunks, metadata)
                shards.append(shard)
                total_chunks += shard["num_chunks"]
                total_tokens += shard["num_tokens"]
                shard_index += 1
                shard_chunks = []

        progress.set_postfix(
            docs=total_documents,
            chunks=total_chunks + len(shard_chunks),
            shards=len(shards),
        )

    if shard_chunks:
        shard = save_shard(output_dir, shard_index, shard_chunks, metadata)
        shards.append(shard)
        total_chunks += shard["num_chunks"]
        total_tokens += shard["num_tokens"]

    if not shards:
        raise ValueError("没有生成任何训练分片，请检查输入语料或 seq_length 设置")

    manifest = {
        **metadata,
        "num_documents": total_documents,
        "num_shards": len(shards),
        "num_chunks": total_chunks,
        "num_tokens": total_tokens,
        "remaining_tokens_dropped": len(token_buffer),
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成: 预训练数据已编码")
    print(f"读取文档数: {total_documents}")
    print(f"生成分片数: {len(shards)}")
    print(f"训练样本数: {total_chunks}")
    print(f"训练 token 数: {total_tokens}")
    print(f"manifest: {manifest_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Encode public corpus into pretraining token shards")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--tokenizer-dir", default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--text-field", default="text")
    return parser.parse_args()


if __name__ == "__main__":
    encode_pretrain_data(parse_args())
