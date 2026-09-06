"""Step 03: encode cleaned documents into memory-mappable token shards.

第 1 步的每篇文档先通过稳定哈希分配到 train 或 validation，再分别连续打包为
固定长度 token 序列。这样同一文档不会同时进入两套数据，且重复运行仍得到相同
划分。输出使用 NumPy .npy，而不是巨型 Python list 或 pickle，便于第 4 步 mmap。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


STEP_ID = "03_encoding"
STEP_NAME = "Encoding"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "01_public_corpus" / "output" / "manifest.json"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "02_tokenizer" / "output" / "atman_tokenizer"
DEFAULT_OUTPUT_DIR = STEP_DIR / "output"

TEXT_FIELD = "text"
DEFAULT_SEQ_LENGTH = 1_024
DEFAULT_VALIDATION_RATIO = 0.01
DEFAULT_SHARD_ROWS = 50_000
DEFAULT_BATCH_DOCUMENTS = 128
SPLIT_SEED = 42


def sha256_file(path: Path) -> str:
    """摘要 tokenizer.json，防止编码和训练阶段误用另一份词表。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(manifest_path: Path, tokenizer_dir: Path) -> tuple[dict, list[Path], dict]:
    """校验第 1、2 步输出，并返回语料分片及 tokenizer 元数据。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到第 1 步 manifest: {manifest_path.resolve()}")
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"找不到第 2 步 tokenizer: {tokenizer_dir.resolve()}")

    corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus_manifest.get("generated_by_step") != "01_public_corpus":
        raise ValueError("输入 manifest 不是第 1 步生成的")
    if corpus_manifest.get("status") != "completed":
        raise ValueError(f"第 1 步尚未完成: {corpus_manifest.get('status')}")

    shard_paths = [
        manifest_path.parent / result["file"]
        for result in corpus_manifest.get("source_results", [])
        if result.get("status") == "completed" and result.get("file")
    ]
    if not shard_paths:
        raise ValueError("第 1 步 manifest 中没有语料分片")
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"第 1 步语料分片不存在: {missing[0].resolve()}")

    tokenizer_metadata_path = tokenizer_dir / "metadata.json"
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_metadata_path.is_file() or not tokenizer_json_path.is_file():
        raise FileNotFoundError("第 2 步输出缺少 metadata.json 或 tokenizer.json")
    tokenizer_metadata = json.loads(tokenizer_metadata_path.read_text(encoding="utf-8"))
    if tokenizer_metadata.get("generated_by_step") != "02_tokenizer":
        raise ValueError("tokenizer metadata 不是第 2 步生成的")
    if tokenizer_metadata.get("input_sources_config_sha256") != corpus_manifest.get(
        "sources_config_sha256"
    ):
        raise ValueError("第 1 步语料配置与第 2 步 tokenizer 不匹配")

    actual_hash = sha256_file(tokenizer_json_path)
    if tokenizer_metadata.get("tokenizer_json_sha256") != actual_hash:
        raise ValueError("tokenizer.json 摘要与 metadata.json 不一致")
    return corpus_manifest, shard_paths, tokenizer_metadata


def iter_documents(shard_paths: list[Path], max_documents: int) -> Iterator[tuple[str, str]]:
    """遍历全部 JSONL，返回稳定文档标识和正文。"""
    documents = 0
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if max_documents and documents >= max_documents:
                    return
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON 格式错误: {shard_path.name}:{line_number}"
                    ) from error

                text = record.get(TEXT_FIELD)
                if not isinstance(text, str) or not text.strip():
                    continue
                source = str(record.get("source", shard_path.stem))
                document_id = f"{source}:{shard_path.name}:{line_number}"
                documents += 1
                yield document_id, text


def choose_split(document_id: str, validation_ratio: float) -> str:
    """用稳定哈希做文档级划分，避免同一文档的 token 泄漏到验证集。"""
    if validation_ratio <= 0:
        return "train"
    digest = hashlib.blake2b(
        f"{SPLIT_SEED}:{document_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, "big") / 2**64
    return "validation" if value < validation_ratio else "train"


class NpyShardWriter:
    """以固定内存缓冲写入二维 .npy 分片。"""

    def __init__(
        self,
        output_dir: Path,
        split: str,
        seq_length: int,
        shard_rows: int,
        dtype: np.dtype,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.seq_length = seq_length
        self.shard_rows = shard_rows
        self.dtype = dtype
        self.buffer = np.empty((shard_rows, seq_length), dtype=dtype)
        self.rows = 0
        self.total_rows = 0
        self.shards: list[dict] = []

    def add(self, token_ids: list[int]) -> None:
        if len(token_ids) != self.seq_length:
            raise ValueError("写入 token 序列的长度与 seq_length 不一致")
        self.buffer[self.rows] = token_ids
        self.rows += 1
        if self.rows == self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if self.rows == 0:
            return

        shard_index = len(self.shards)
        file_name = f"{self.split}_{shard_index:05d}.npy"
        final_path = self.output_dir / file_name
        partial_path = final_path.with_suffix(".npy.partial")
        with partial_path.open("wb") as file:
            np.save(file, self.buffer[: self.rows], allow_pickle=False)
        partial_path.replace(final_path)

        num_tokens = self.rows * self.seq_length
        self.shards.append(
            {
                "file": file_name,
                "num_chunks": self.rows,
                "num_tokens": num_tokens,
                "file_bytes": final_path.stat().st_size,
            }
        )
        self.total_rows += self.rows
        self.rows = 0


class TokenPacker:
    """跨文档累计 token，并按 seq_length 连续切块。"""

    def __init__(self, writer: NpyShardWriter, eos_token_id: int) -> None:
        self.writer = writer
        self.eos_token_id = eos_token_id
        self.tokens: list[int] = []
        self.offset = 0
        self.documents = 0
        self.raw_tokens = 0

    def add_document(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.tokens.extend(token_ids)
        self.tokens.append(self.eos_token_id)
        self.documents += 1
        self.raw_tokens += len(token_ids) + 1

        while len(self.tokens) - self.offset >= self.writer.seq_length:
            end = self.offset + self.writer.seq_length
            self.writer.add(self.tokens[self.offset:end])
            self.offset = end

        # 定期丢掉已消费的前缀，避免列表随着整个语料持续增长。
        if self.offset >= 1_000_000:
            self.tokens = self.tokens[self.offset :]
            self.offset = 0

    @property
    def remaining_tokens(self) -> int:
        return len(self.tokens) - self.offset


def ensure_empty_output(output_dir: Path) -> None:
    """拒绝覆盖已有编码结果，避免新旧 tokenizer 数据混在一起。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = list(output_dir.glob("*.npy")) + list(output_dir.glob("*.partial"))
    if (output_dir / "manifest.json").exists() or generated_files:
        raise FileExistsError(
            f"第 3 步输出已存在: {output_dir}\n"
            "请确认不再需要后手动清理，再重新编码。"
        )


def encode_pretrain_data(args: argparse.Namespace) -> None:
    """执行第 3 步：批量分词、文档级划分并保存 mmap 友好分片。"""
    manifest_path = Path(args.input_manifest).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    corpus_manifest, shard_paths, _tokenizer_metadata = load_inputs(
        manifest_path,
        tokenizer_dir,
    )
    ensure_empty_output(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 缺少 eos_token，无法标记文档边界")
    tokenizer.model_max_length = 10**30

    token_dtype = np.dtype("uint16" if len(tokenizer) <= 65_535 else "uint32")
    writers = {
        split: NpyShardWriter(
            output_dir,
            split,
            args.seq_length,
            args.shard_rows,
            token_dtype,
        )
        for split in ("train", "validation")
    }
    packers = {
        split: TokenPacker(writer, tokenizer.eos_token_id)
        for split, writer in writers.items()
    }

    expected_documents = int(corpus_manifest.get("totals", {}).get("documents", 0))
    if args.max_documents:
        expected_documents = min(expected_documents, args.max_documents)
    progress = tqdm(total=expected_documents or None, desc="Encoding", unit="doc", dynamic_ncols=True)

    batch_ids: list[str] = []
    batch_texts: list[str] = []
    total_documents = 0

    def process_batch() -> None:
        nonlocal total_documents
        if not batch_texts:
            return
        encoded_batch = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for document_id, token_ids in zip(batch_ids, encoded_batch):
            split = choose_split(document_id, args.validation_ratio)
            packers[split].add_document(token_ids)
            total_documents += 1
        progress.update(len(batch_texts))
        progress.set_postfix(
            train=writers["train"].total_rows + writers["train"].rows,
            validation=writers["validation"].total_rows + writers["validation"].rows,
        )
        batch_ids.clear()
        batch_texts.clear()

    for document_id, text in iter_documents(shard_paths, args.max_documents):
        batch_ids.append(document_id)
        batch_texts.append(text)
        if len(batch_texts) >= args.batch_documents:
            process_batch()
    process_batch()
    progress.close()

    for writer in writers.values():
        writer.flush()
    if writers["train"].total_rows == 0:
        raise ValueError("没有生成训练序列，请检查语料和 seq_length")
    if args.validation_ratio > 0 and writers["validation"].total_rows == 0:
        raise ValueError("没有生成验证序列；试跑时请增加 --max-documents 或设 --validation-ratio 0")

    split_manifest = {}
    for split in ("train", "validation"):
        writer = writers[split]
        packer = packers[split]
        split_manifest[split] = {
            "documents": packer.documents,
            "raw_tokens_with_eos": packer.raw_tokens,
            "num_chunks": writer.total_rows,
            "num_tokens": writer.total_rows * args.seq_length,
            "remaining_tokens_dropped": packer.remaining_tokens,
            "num_shards": len(writer.shards),
            "shards": writer.shards,
        }

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    manifest = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "format_version": 2,
        "status": "completed",
        "input_manifest": str(manifest_path),
        "input_sources_config_sha256": corpus_manifest.get("sources_config_sha256"),
        "input_shards": [path.name for path in shard_paths],
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_json_sha256": sha256_file(tokenizer_json_path),
        "tokenizer_vocab_size": len(tokenizer),
        "token_dtype": token_dtype.name,
        "seq_length": args.seq_length,
        "validation_ratio": args.validation_ratio,
        "split_seed": SPLIT_SEED,
        "shard_rows": args.shard_rows,
        "batch_documents": args.batch_documents,
        "max_documents": args.max_documents,
        "num_documents": total_documents,
        "splits": split_manifest,
    }
    manifest_path_out = output_dir / "manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Step 03 完成")
    print(f"编码文档: {total_documents:,}")
    print(f"训练 token: {split_manifest['train']['num_tokens']:,}")
    print(f"验证 token: {split_manifest['validation']['num_tokens']:,}")
    print(f"token dtype: {token_dtype.name}")
    print(f"输出 manifest: {manifest_path_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 03: encode Step 01 JSONL into train/validation NumPy shards"
    )
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--tokenizer-dir", default=str(DEFAULT_TOKENIZER_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    parser.add_argument("--batch-documents", type=int, default=DEFAULT_BATCH_DOCUMENTS)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="最多编码多少篇文档；0 表示全部，仅建议试跑时限制。",
    )
    args = parser.parse_args()

    if args.seq_length < 2:
        parser.error("--seq-length 必须至少为 2")
    if not 0 <= args.validation_ratio < 1:
        parser.error("--validation-ratio 必须在 [0, 1) 范围内")
    if args.shard_rows <= 0 or args.batch_documents <= 0:
        parser.error("--shard-rows 和 --batch-documents 必须大于 0")
    if args.max_documents < 0:
        parser.error("--max-documents 不能小于 0")
    return args


if __name__ == "__main__":
    encode_pretrain_data(parse_args())
