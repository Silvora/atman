import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


STEP_ID = "02_tokenizer"
STEP_NAME = "Tokenizer"
DEFAULT_INPUT_PATH = "output/01_public_corpus/public_corpus.jsonl"
DEFAULT_OUTPUT_DIR = "output/02_tokenizer/atman_tokenizer"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def iter_texts(input_path: Path, text_field: str, max_samples: int):
    """从第 1 步的 JSONL 语料中逐行读取文本，避免一次性占用太多内存。"""
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
            yield text


def build_byte_level_bpe_tokenizer() -> Tokenizer:
    """构建 Byte-Level BPE tokenizer。

    Byte-Level BPE 对未知字符更友好，中文、英文、标点混在一起也能稳定编码。
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(args):
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到第 1 步输出语料: {input_path.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_byte_level_bpe_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    print(f"读取语料: {input_path.resolve()}")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"训练配置: vocab_size={args.vocab_size}, min_frequency={args.min_frequency}")

    tokenizer.train_from_iterator(
        iter_texts(input_path, args.text_field, args.max_samples),
        trainer=trainer,
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(output_dir)

    metadata = {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "vocab_size": fast_tokenizer.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": SPECIAL_TOKENS,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_text = args.sample_text
    encoded = fast_tokenizer.encode(sample_text)
    decoded = fast_tokenizer.decode(encoded)

    print("完成: tokenizer 已保存")
    print(f"词表大小: {fast_tokenizer.vocab_size}")
    print(f"元数据: {metadata_path.resolve()}")
    print(f"测试文本: {sample_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer for Atman")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--sample-text", default="人工智能的未来是开放、协作和持续学习。")
    return parser.parse_args()


if __name__ == "__main__":
    train_tokenizer(parse_args())
