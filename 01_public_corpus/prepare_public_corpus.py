import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


STEP_ID = "01_public_corpus"
STEP_NAME = "Public Corpus"
DEFAULT_OUTPUT_PATH = f"output/{STEP_ID}/public_corpus.jsonl"


def clean_text(text: str) -> str:
    """做轻量清洗，保留正文语义，不做过度改写。"""
    # 统一换行和空白，避免模型学到杂乱的排版符号。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 去掉常见控制字符；这些字符通常没有语义，只会污染训练数据。
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def text_hash(text: str) -> str:
    """用哈希做简单去重，比直接存全文更省内存。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_public_dataset(args):
    """读取公开数据集。

    streaming=True 适合大语料，因为它不会一次性把完整数据集放进内存。
    """
    return load_dataset(
        args.dataset_name,
        data_files=args.data_file,
        split=args.split,
        streaming=args.streaming,
    )


def prepare_public_corpus(args):
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = iter_public_dataset(args)
    seen_hashes = set()
    kept = 0
    skipped_short = 0
    skipped_empty = 0
    skipped_duplicate = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        progress = tqdm(dataset, desc="Preparing public corpus", dynamic_ncols=True)
        for sample in progress:
            if args.max_samples > 0 and kept >= args.max_samples:
                break

            raw_text = sample.get(args.text_field, "")
            if not isinstance(raw_text, str) or not raw_text.strip():
                skipped_empty += 1
                continue

            text = clean_text(raw_text)
            if len(text) < args.min_chars:
                skipped_short += 1
                continue

            digest = text_hash(text)
            if digest in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_hashes.add(digest)

            # JSONL 一行一条文本，后续 tokenizer 和编码脚本都容易流式处理。
            record = {
                "text": text,
                "source": args.dataset_name,
                # 标记这条数据是由哪一个教程步骤生成的，方便后续追踪数据来源。
                "generated_by_step": STEP_ID,
                "generated_by_step_name": STEP_NAME,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

            progress.set_postfix(
                kept=kept,
                empty=skipped_empty,
                short=skipped_short,
                duplicate=skipped_duplicate,
            )

    print(f"完成: 保留 {kept} 条文本")
    print(f"跳过: 空文本 {skipped_empty} 条，过短 {skipped_short} 条，重复 {skipped_duplicate} 条")
    print(f"输出文件: {output_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a public text corpus for language model training")
    parser.add_argument("--dataset-name", default="fjcanyue/wikipedia-zh-cn")
    parser.add_argument("--data-file", default="wikipedia-zh-cn-20260501.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=1000000000000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_public_corpus(parse_args())
