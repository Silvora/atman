"""Step 02: train one tokenizer for every Atman model size.

输入是第 1 步的 manifest.json，而不是某一个写死的 JSONL 文件。脚本按照
manifest 中的顺序读取全部语料分片，训练统一的 32K Byte-Level BPE tokenizer。
small、medium、large 共用同一份 tokenizer，模型大小由第 4 步的网络配置决定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


STEP_ID = "02_tokenizer"
STEP_NAME = "Tokenizer"
STEP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP_DIR.parent
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "01_public_corpus" / "output" / "manifest.json"
DEFAULT_OUTPUT_DIR = STEP_DIR / "output" / "atman_tokenizer"

TEXT_FIELD = "text"
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_MODEL_MAX_LENGTH = 1_024
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
SAMPLE_TEXT = "人工智能的未来来自高质量数据、可靠训练和持续评估。"


def sha256_file(path: Path) -> str:
    """计算小型配置文件的摘要，供后续步骤确认 tokenizer 没被替换。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_manifest(manifest_path: Path) -> tuple[dict, list[Path]]:
    """读取第 1 步清单，并返回其中所有已完成的 JSONL 分片。"""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到第 1 步 manifest: {manifest_path.resolve()}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("generated_by_step") != "01_public_corpus":
        raise ValueError("输入 manifest 不是第 1 步生成的")
    if manifest.get("status") != "completed":
        raise ValueError(f"第 1 步尚未完成，当前状态: {manifest.get('status')}")

    source_results = manifest.get("source_results", [])
    shard_paths = [
        manifest_path.parent / result["file"]
        for result in source_results
        if result.get("status") == "completed" and result.get("file")
    ]
    if not shard_paths:
        raise ValueError("第 1 步 manifest 中没有已完成的语料分片")

    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"第 1 步语料分片不存在: {missing[0].resolve()}")
    return manifest, shard_paths


class CorpusTextIterator:
    """逐行遍历多个 JSONL 分片，并记录 tokenizer 实际看到的数据量。"""

    def __init__(self, shard_paths: list[Path], max_documents: int) -> None:
        self.shard_paths = shard_paths
        self.max_documents = max_documents
        self.documents = 0
        self.text_bytes = 0

    def __iter__(self) -> Iterator[str]:
        self.documents = 0
        self.text_bytes = 0

        for shard_path in self.shard_paths:
            with shard_path.open("r", encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    if self.max_documents and self.documents >= self.max_documents:
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

                    self.documents += 1
                    self.text_bytes += len(text.encode("utf-8"))
                    yield text


def build_tokenizer() -> Tokenizer:
    """构建 Byte-Level BPE，保证中文、英文、数字和符号都可无损编码。"""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(args: argparse.Namespace) -> None:
    """训练 tokenizer，并以 Hugging Face 标准格式写入第 2 步 output。"""
    manifest_path = Path(args.input_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    corpus_manifest, shard_paths = load_corpus_manifest(manifest_path)

    if (output_dir / "metadata.json").exists():
        raise FileExistsError(
            f"输出 tokenizer 已存在: {output_dir}\n"
            "为避免覆盖实验，请先确认后手动清理该目录。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    text_iterator = CorpusTextIterator(shard_paths, args.max_documents)
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"读取 manifest: {manifest_path}")
    print(f"读取语料分片: {len(shard_paths)} 个")
    print(f"输出目录: {output_dir}")
    print(
        f"训练配置: vocab_size={args.vocab_size}, "
        f"min_frequency={args.min_frequency}"
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    fast_tokenizer.save_pretrained(output_dir)

    tokenizer_json_path = output_dir / "tokenizer.json"
    sample_ids = fast_tokenizer.encode(SAMPLE_TEXT, add_special_tokens=False)
    sample_decoded = fast_tokenizer.decode(sample_ids, skip_special_tokens=False)
    metadata = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "input_manifest": str(manifest_path),
        "input_sources_config_sha256": corpus_manifest.get("sources_config_sha256"),
        "input_shards": [path.name for path in shard_paths],
        "trained_documents": text_iterator.documents,
        "trained_text_bytes": text_iterator.text_bytes,
        "trained_text_gib": round(text_iterator.text_bytes / 1024**3, 3),
        "vocab_size_requested": args.vocab_size,
        "vocab_size": len(fast_tokenizer),
        "min_frequency": args.min_frequency,
        "model_max_length": args.model_max_length,
        "special_tokens": SPECIAL_TOKENS,
        "tokenizer_json_sha256": sha256_file(tokenizer_json_path),
        "sample": {
            "text": SAMPLE_TEXT,
            "token_ids": sample_ids,
            "decoded": sample_decoded,
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Step 02 完成")
    print(f"训练文档: {text_iterator.documents:,}")
    print(f"实际词表: {len(fast_tokenizer):,}")
    print(f"测试 token 数: {len(sample_ids)}")
    print(f"输出元数据: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 02: train one Byte-Level BPE tokenizer from Step 01 shards"
    )
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument("--model-max-length", type=int, default=DEFAULT_MODEL_MAX_LENGTH)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="最多读取多少篇文档；0 表示使用第 1 步的全部输出。",
    )
    args = parser.parse_args()

    if args.vocab_size <= len(SPECIAL_TOKENS) + 256:
        parser.error("--vocab-size 太小，必须容纳特殊 token 和完整 byte alphabet")
    if args.min_frequency <= 0:
        parser.error("--min-frequency 必须大于 0")
    if args.model_max_length <= 0:
        parser.error("--model-max-length 必须大于 0")
    if args.max_documents < 0:
        parser.error("--max-documents 不能小于 0")
    return args


if __name__ == "__main__":
    train_tokenizer(parse_args())
