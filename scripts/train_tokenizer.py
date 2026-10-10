#!/usr/bin/env python3
"""训练、评测并选择正式 Byte-level BPE Tokenizer。

脚本只读取预处理后的 train/validation，不直接读取 raw。16K 与 32K 候选
使用完全相同的确定性抽样，保证差异只来自词表容量。评测通过后按 YAML 阈值
自动选择正式词表，并复制到 data/tokenizer/final。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator


V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from src.data.corpus import iter_jsonl_texts, percentile  # noqa: E402
from src.utils.config import load_yaml_config, resolve_v1_path  # noqa: E402

try:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
except ImportError as exc:  # pragma: no cover - environment error path
    raise RuntimeError(
        "tokenizers is required; install the v1 requirements before running"
    ) from exc


# Tokenizer 有独立配置，不与任一模型尺寸绑定。
DEFAULT_CONFIG = V1_ROOT / "configs" / "tokenizer.yaml"


def parse_args() -> argparse.Namespace:
    """定义测试或自动化环境需要的覆盖项；正式参数默认来自 YAML。"""

    parser = argparse.ArgumentParser(
        description=(
            "在确定性样本上训练 Byte-level BPE 候选，在 validation 上评测并选择正式词表。"
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--processed-dir",
        help="Override output_dir from the referenced data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output_dir from tokenizer.yaml",
    )
    parser.add_argument(
        "--sample-documents",
        type=int,
        help="Override tokenizer.training.sample_documents",
    )
    parser.add_argument(
        "--eval-documents",
        type=int,
        help="Override tokenizer.evaluation.max_documents",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        help="Override tokenizer.candidate_vocab_sizes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing candidate and final tokenizer directories",
    )
    return parser.parse_args()


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需的配置映射，避免深层 KeyError 掩盖配置结构错误。"""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration key {key!r} must be a mapping")
    return value


def read_json(path: Path) -> dict[str, Any]:
    """读取预处理报告，并要求根节点为 JSON object。"""

    if not path.is_file():
        raise FileNotFoundError(f"required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def resolve_config_reference(config_path: Path, reference: str) -> Path:
    """解析配置文件之间的引用。

    相对路径以当前配置文件所在目录为基准，而不是以调用命令的工作目录为
    基准。因此从任意目录执行脚本，``data_config: data.yaml`` 都保持有效。
    """

    path = Path(reference).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """以稳定、可审阅的 UTF-8 格式保存指标或 manifest。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def selected_document_indices(
    total_documents: int,
    sample_documents: int,
    seed: int,
) -> list[int] | None:
    """生成固定数量的不重复随机行号。

    random.sample(range(...)) 只保存整数索引，不保存正文。排序后可以顺序扫描
    大文件，避免随机磁盘访问。seed 或 sample_documents 改变会改变最终词表。
    """

    sample_size = min(total_documents, sample_documents)
    if sample_size == total_documents:
        return None
    random_generator = random.Random(seed)
    indices = random_generator.sample(range(total_documents), sample_size)
    indices.sort()
    return indices


def iter_selected_texts(
    path: Path,
    *,
    text_field: str,
    selected_indices: list[int] | None,
) -> Iterator[str]:
    """按已排序索引流式产生训练文本，不把百万篇正文同时放入内存。"""

    if selected_indices is None:
        for _, value, _ in iter_jsonl_texts(
            path,
            text_field=text_field,
            strict=True,
        ):
            if not isinstance(value, str):
                raise AssertionError("strict JSONL iteration returned an error")
            yield value
        return

    selection = iter(selected_indices)
    wanted = next(selection, None)
    for zero_based_index, (_, value, _) in enumerate(
        iter_jsonl_texts(path, text_field=text_field, strict=True)
    ):
        if wanted is None:
            break
        if zero_based_index != wanted:
            continue
        if not isinstance(value, str):
            raise AssertionError("strict JSONL iteration returned an error")
        yield value
        wanted = next(selection, None)
    if wanted is not None:
        raise RuntimeError(
            "processed train file ended before all selected indices were read"
        )


def is_cjk(character: str) -> bool:
    """判断字符是否属于常用 CJK 统一表意文字区间。"""

    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    validation_path: Path,
    *,
    text_field: str,
    max_documents: int,
    special_tokens: dict[str, str],
) -> dict[str, Any]:
    """在 validation 上计算压缩率、序列长度和可逆性指标。

    max_documents 只影响评测耗时和统计稳定性，不改变词表。特殊 token 必须
    各自编码为一个 ID；Byte-level BPE 不允许出现未知 token。
    """

    token_lengths: list[int] = []
    documents = 0
    total_tokens = 0
    total_characters = 0
    total_utf8_bytes = 0
    roundtrip_mismatches = 0
    cjk_characters = 0
    cjk_tokens = 0
    latin_characters = 0
    latin_tokens = 0

    for _, value, _ in iter_jsonl_texts(
        validation_path,
        text_field=text_field,
        strict=True,
    ):
        if documents >= max_documents:
            break
        if not isinstance(value, str):
            raise AssertionError("strict JSONL iteration returned an error")

        # 禁用自动特殊 token，测量正文自身的真实 token 数。
        encoding = tokenizer.encode(value, add_special_tokens=False)
        token_count = len(encoding.ids)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)

        documents += 1
        total_tokens += token_count
        total_characters += len(value)
        total_utf8_bytes += len(value.encode("utf-8"))
        token_lengths.append(token_count)
        if decoded != value:
            roundtrip_mismatches += 1

        # 语言子集指标按“正文中占优势的字符类型”归类，只用于压缩率诊断。
        cjk_count = sum(1 for character in value if is_cjk(character))
        latin_count = sum(
            1 for character in value if character.isascii() and character.isalpha()
        )
        if cjk_count > latin_count:
            cjk_characters += cjk_count
            cjk_tokens += token_count
        elif latin_count > cjk_count:
            latin_characters += latin_count
            latin_tokens += token_count

    if documents == 0:
        raise ValueError(f"validation corpus is empty: {validation_path}")

    special_token_metrics: dict[str, Any] = {}
    for name, token in special_tokens.items():
        encoded = tokenizer.encode(token, add_special_tokens=False).ids
        special_token_metrics[name] = {
            "token": token,
            "id": tokenizer.token_to_id(token),
            "encoded_ids": encoded,
            "single_token": len(encoded) == 1,
        }

    return {
        "documents": documents,
        "tokens": total_tokens,
        "unknown_token_id": None,
        "unknown_token_occurrences": 0,
        "characters": total_characters,
        "utf8_bytes": total_utf8_bytes,
        "tokens_per_document_mean": total_tokens / documents,
        "characters_per_token": (
            total_characters / total_tokens if total_tokens else None
        ),
        "utf8_bytes_per_token": (
            total_utf8_bytes / total_tokens if total_tokens else None
        ),
        "cjk_characters_per_token": (
            cjk_characters / cjk_tokens if cjk_tokens else None
        ),
        "latin_characters_per_token": (
            latin_characters / latin_tokens if latin_tokens else None
        ),
        "tokens_per_document_p50": percentile(token_lengths, 0.50),
        "tokens_per_document_p90": percentile(token_lengths, 0.90),
        "tokens_per_document_p95": percentile(token_lengths, 0.95),
        "tokens_per_document_p99": percentile(token_lengths, 0.99),
        "documents_over_512_tokens": sum(
            length > 512 for length in token_lengths
        ),
        "documents_over_1024_tokens": sum(
            length > 1024 for length in token_lengths
        ),
        "documents_over_2048_tokens": sum(
            length > 2048 for length in token_lengths
        ),
        "roundtrip_mismatches": roundtrip_mismatches,
        "roundtrip_success_rate": 1.0 - roundtrip_mismatches / documents,
        "special_tokens": special_token_metrics,
    }


def train_candidate(
    *,
    vocab_size: int,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    text_field: str,
    total_train_documents: int,
    sample_documents: int,
    sample_seed: int,
    min_frequency: int,
    special_tokens: dict[str, str],
    add_prefix_space: bool,
    use_regex: bool,
    eval_documents: int,
    require_exact_roundtrip: bool,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    """训练单个词表候选并写出可独立加载的全部文件。

    vocab_size 会同时影响 embedding 参数量与序列长度；add_prefix_space 和
    use_regex 会改变切分边界；min_frequency 控制低频 BPE 合并。任一参数
    改变都必须重新训练候选并重新编码正式语料。
    """

    selected_indices = selected_document_indices(
        total_train_documents,
        sample_documents,
        sample_seed,
    )
    actual_sample_size = min(total_train_documents, sample_documents)

    # Byte-level 初始表覆盖 256 个字节，因此不设置 UNK。
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=add_prefix_space,
        use_regex=use_regex,
    )
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=list(special_tokens.values()),
        initial_alphabet=ByteLevel.alphabet(),
    )

    started_at = time.time()
    # 每个候选都会重新构造同一组索引和同一顺序的文本迭代器。
    tokenizer.train_from_iterator(
        iter_selected_texts(
            train_path,
            text_field=text_field,
            selected_indices=selected_indices,
        ),
        trainer=trainer,
        length=actual_sample_size,
    )
    training_seconds = time.time() - started_at

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != vocab_size:
        raise RuntimeError(
            f"requested vocab size {vocab_size}, trained {actual_vocab_size}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    # 同时保存统一 tokenizer.json 和 GPT-2 风格 vocab.json/merges.txt。
    tokenizer.save(str(output_dir / "tokenizer.json"))
    tokenizer.model.save(str(output_dir))
    # 从磁盘重新加载后再评测，验证最终发布文件而非内存对象。
    tokenizer = Tokenizer.from_file(str(output_dir / "tokenizer.json"))

    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "add_prefix_space": add_prefix_space,
        "clean_up_tokenization_spaces": False,
        "bos_token": None,
        "eos_token": special_tokens["eos_token"],
        "pad_token": special_tokens["pad_token"],
        "unk_token": None,
        # 角色、Thinking 和工具标记必须在 Transformers 加载时仍被识别为特殊 token。
        "additional_special_tokens": [
            value
            for key, value in special_tokens.items()
            if key not in {"eos_token", "pad_token"}
        ],
    }
    special_tokens_map = {
        "eos_token": special_tokens["eos_token"],
        "pad_token": special_tokens["pad_token"],
        "additional_special_tokens": [
            value
            for key, value in special_tokens.items()
            if key not in {"eos_token", "pad_token"}
        ],
    }
    write_json(output_dir / "tokenizer_config.json", tokenizer_config)
    write_json(output_dir / "special_tokens_map.json", special_tokens_map)

    metrics = evaluate_tokenizer(
        tokenizer,
        validation_path,
        text_field=text_field,
        max_documents=eval_documents,
        special_tokens=special_tokens,
    )
    if require_exact_roundtrip and metrics["roundtrip_mismatches"] != 0:
        raise RuntimeError(
            f"tokenizer {vocab_size} failed round-trip validation on "
            f"{metrics['roundtrip_mismatches']} documents"
        )
    if not all(
        item["single_token"] for item in metrics["special_tokens"].values()
    ):
        raise RuntimeError(
            f"tokenizer {vocab_size} did not preserve every special token"
        )

    manifest = {
        "schema_version": 1,
        "algorithm": "Byte-level BPE",
        "normalizer": None,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": add_prefix_space,
            "use_regex": use_regex,
        },
        "decoder": "ByteLevel",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "min_frequency": min_frequency,
        "sample_documents": actual_sample_size,
        "sample_seed": sample_seed,
        "special_tokens": metrics["special_tokens"],
        "training_seconds": training_seconds,
        "source": source_manifest,
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "manifest.json", manifest)
    return {
        "vocab_size": vocab_size,
        "directory": str(output_dir),
        "metrics": metrics,
        "manifest": manifest,
    }


def main() -> int:
    """校验配置，训练全部候选，执行验收并发布 final。"""

    args = parse_args()
    # 第一阶段：对所有固定设计做显式校验，禁止“配置改了但代码静默忽略”。
    tokenizer_config_path = Path(args.config).expanduser().resolve()
    tokenizer_config = load_yaml_config(tokenizer_config_path)
    if tokenizer_config.get("schema_version") != 1:
        raise ValueError("only schema_version: 1 is supported")

    # tokenizer.yaml 只保存分词器参数；数据路径和字段名由 data.yaml 统一管理。
    data_config_reference = tokenizer_config.get("data_config")
    if not isinstance(data_config_reference, str) or not data_config_reference:
        raise ValueError("tokenizer.data_config must be a non-empty path")
    data_config_path = resolve_config_reference(
        tokenizer_config_path,
        data_config_reference,
    )
    data_config = load_yaml_config(data_config_path)
    if data_config.get("schema_version") != 1:
        raise ValueError("referenced data config only supports schema_version: 1")

    tokenizer_training = require_mapping(tokenizer_config, "training")
    tokenizer_evaluation = require_mapping(tokenizer_config, "evaluation")
    special_tokens = require_mapping(tokenizer_config, "special_tokens")
    pre_tokenizer_config = require_mapping(tokenizer_config, "pre_tokenizer")

    if tokenizer_config.get("algorithm") != "byte_level_bpe":
        raise ValueError("only tokenizer.algorithm: byte_level_bpe is supported")
    if tokenizer_config.get("normalizer") is not None:
        raise ValueError("tokenizer.normalizer must remain null")
    if tokenizer_config.get("bos_token") is not None:
        raise ValueError("tokenizer.bos_token must remain null")
    if tokenizer_config.get("unk_token") is not None:
        raise ValueError("tokenizer.unk_token must remain null")
    required_special_tokens = {
        "eos_token", "pad_token", "im_start_token", "im_end_token",
        "user_token", "assistant_token", "system_token", "tool_token",
        "think_start_token", "think_end_token", "answer_start_token",
        "answer_end_token", "thinking_mode_token", "non_thinking_mode_token",
        "tool_call_token", "tool_response_token", "tool_calls_begin_token",
        "tool_calls_end_token", "observation_token", "reserved_0", "reserved_1",
        "reserved_2", "reserved_3",
    }
    if set(special_tokens) != required_special_tokens:
        missing = sorted(required_special_tokens - set(special_tokens))
        extra = sorted(set(special_tokens) - required_special_tokens)
        raise ValueError(
            "tokenizer.special_tokens keys do not match the v2/v3 contract; "
            f"missing={missing}, extra={extra}"
        )
    values = list(special_tokens.values())
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("all tokenizer.special_tokens values must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("tokenizer.special_tokens values must be unique")
    if not bool(tokenizer_config.get("shared_by_all_model_sizes", True)):
        raise ValueError("the formal tokenizer must be shared by all model sizes")
    if pre_tokenizer_config.get("type") != "byte_level":
        raise ValueError("only tokenizer.pre_tokenizer.type: byte_level is supported")
    if tokenizer_training.get("source_split") != "train":
        raise ValueError("tokenizer.training.source_split must be train")
    if (
        tokenizer_training.get("sample_method")
        != "deterministic_random_indices"
    ):
        raise ValueError(
            "only deterministic_random_indices sampling is supported"
        )
    if tokenizer_training.get("initial_alphabet") != "byte_level":
        raise ValueError("tokenizer.training.initial_alphabet must be byte_level")
    if tokenizer_evaluation.get("source_split") != "validation":
        raise ValueError("tokenizer.evaluation.source_split must be validation")

    # 第二阶段：解析预处理产物。audit_report 中的行数用于精确抽样。
    processed_dir = (
        Path(args.processed_dir).expanduser().resolve()
        if args.processed_dir
        else resolve_v1_path(data_config["output_dir"])
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_v1_path(tokenizer_config["output_dir"])
    )
    train_path = processed_dir / str(data_config["train_file"])
    validation_path = processed_dir / str(data_config["validation_file"])
    report_path = processed_dir / str(data_config["report_file"])
    audit_report = read_json(report_path)

    counts = require_mapping(audit_report, "counts")
    total_train_documents = int(counts["train_documents"])
    if total_train_documents <= 0:
        raise ValueError("processed training corpus contains no documents")
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            f"processed train/validation files are missing under {processed_dir}"
        )

    sample_documents = int(
        args.sample_documents
        if args.sample_documents is not None
        else tokenizer_training["sample_documents"]
    )
    eval_documents = int(
        args.eval_documents
        if args.eval_documents is not None
        else tokenizer_evaluation["max_documents"]
    )
    vocab_sizes = sorted(
        set(
            args.vocab_sizes
            if args.vocab_sizes is not None
            else tokenizer_config["candidate_vocab_sizes"]
        )
    )
    if sample_documents <= 0 or eval_documents <= 0:
        raise ValueError("sample and evaluation document counts must be positive")
    if not vocab_sizes or any(size < 512 for size in vocab_sizes):
        raise ValueError("candidate vocabulary sizes must be at least 512")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_root = output_dir / "candidates"
    final_dir = output_dir / "final"
    comparison_path = output_dir / "comparison.json"
    candidates_root.mkdir(parents=True, exist_ok=True)

    # 默认拒绝覆盖已有 Tokenizer，防止误删正式词表；--force 才允许重建。
    candidate_dirs = [
        candidates_root / f"byte_bpe_{vocab_size}" for vocab_size in vocab_sizes
    ]
    existing = [
        path
        for path in (*candidate_dirs, final_dir, comparison_path)
        if path.exists()
    ]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"tokenizer outputs already exist: {joined}; use --force to replace them"
        )
    if args.force:
        for path in candidate_dirs:
            if path.exists():
                shutil.rmtree(path)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if comparison_path.exists():
            comparison_path.unlink()

    text_field = str(data_config.get("text_field", "text"))
    sample_seed = int(tokenizer_training["sample_seed"])
    min_frequency = int(tokenizer_training["min_frequency"])
    threshold = float(tokenizer_evaluation["max_smaller_token_inflation"])
    if threshold < 0.0:
        raise ValueError("max_smaller_token_inflation must be non-negative")
    add_prefix_space = bool(pre_tokenizer_config["add_prefix_space"])
    use_regex = bool(pre_tokenizer_config["use_regex"])
    require_exact_roundtrip = bool(
        tokenizer_evaluation.get("require_exact_roundtrip", True)
    )
    if not bool(
        tokenizer_evaluation.get("require_zero_unknown_tokens", True)
    ):
        raise ValueError(
            "require_zero_unknown_tokens must remain true for Byte-level BPE"
        )

    source_manifest = {
        # 记录两份配置和实际输入路径，使词表能够追溯到确切的数据处理结果。
        "tokenizer_config": str(tokenizer_config_path),
        "data_config": str(data_config_path),
        "audit_report": str(report_path),
        "source": audit_report.get("source"),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_documents": total_train_documents,
        "validation_documents": counts.get("validation_documents"),
    }

    # 第三阶段：所有候选使用相同 train 样本和 validation。
    results = []
    for vocab_size, candidate_dir in zip(vocab_sizes, candidate_dirs):
        print(
            f"training Byte-level BPE candidate: vocab_size={vocab_size:,}",
            flush=True,
        )
        result = train_candidate(
            vocab_size=vocab_size,
            train_path=train_path,
            validation_path=validation_path,
            output_dir=candidate_dir,
            text_field=text_field,
            total_train_documents=total_train_documents,
            sample_documents=sample_documents,
            sample_seed=sample_seed,
            min_frequency=min_frequency,
            special_tokens={str(key): str(value) for key, value in special_tokens.items()},
            add_prefix_space=add_prefix_space,
            use_regex=use_regex,
            eval_documents=eval_documents,
            require_exact_roundtrip=require_exact_roundtrip,
            source_manifest=source_manifest,
        )
        results.append(result)

    # 第四阶段：最大词表作为序列压缩率基线，在阈值内优先选择更小词表，
    # 从而减少 embedding/LM head 参数。
    baseline = max(results, key=lambda item: item["vocab_size"])
    baseline_tokens = float(baseline["metrics"]["tokens"])
    eligible = []
    for result in results:
        inflation = float(result["metrics"]["tokens"]) / baseline_tokens - 1.0
        result["token_inflation_vs_largest"] = inflation
        if inflation <= threshold:
            eligible.append(result)

    selected = min(eligible, key=lambda item: item["vocab_size"])
    selected_dir = Path(selected["directory"])
    # final 是候选的完整副本，不用符号链接，便于独立打包和迁移。
    shutil.copytree(selected_dir, final_dir)

    selection = {
        "schema_version": 1,
        "selection_rule": (
            "choose the smallest candidate whose validation token count inflation "
            "does not exceed the configured threshold relative to the largest "
            "candidate"
        ),
        "max_smaller_token_inflation": threshold,
        "selected_vocab_size": selected["vocab_size"],
        "selected_candidate": str(selected_dir),
        "final_directory": str(final_dir),
        "candidates": [
            {
                "vocab_size": result["vocab_size"],
                "directory": result["directory"],
                "validation_tokens": result["metrics"]["tokens"],
                "characters_per_token": result["metrics"][
                    "characters_per_token"
                ],
                "utf8_bytes_per_token": result["metrics"][
                    "utf8_bytes_per_token"
                ],
                "roundtrip_success_rate": result["metrics"][
                    "roundtrip_success_rate"
                ],
                "token_inflation_vs_largest": result[
                    "token_inflation_vs_largest"
                ],
            }
            for result in results
        ],
    }
    write_json(comparison_path, selection)
    write_json(final_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
