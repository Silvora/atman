#!/usr/bin/env python3
"""审计、清理、精确去重并划分预训练语料。

正式流程只做数据层变换，不执行 tokenization。脚本使用流式 JSONL 读取和
磁盘 SQLite 去重索引，内存占用不会随文档数量线性增长。所有会改变正文或
划分结果的参数均来自 YAML；修改后必须重新生成 train/validation。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from src.data.corpus import (  # noqa: E402
    JsonlRecordError,
    ReservoirSampler,
    iter_jsonl_texts,
    normalize_text,
    percentile,
    sha256_text,
)
from src.utils.config import load_yaml_config, resolve_v1_path  # noqa: E402


DEFAULT_CONFIG = V1_ROOT / "configs" / "data.yaml"


def parse_args() -> argparse.Namespace:
    """定义命令行覆盖项；未指定的参数全部从 YAML 读取。"""

    parser = argparse.ArgumentParser(
        description=(
            "流式校验和清理 JSONL，精确去重，并稳定划分 train/validation。"
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--input",
        default="full",
        help="'full', 'mini', or an explicit JSONL path (default: full)",
    )
    parser.add_argument(
        "--output-dir",
        help="覆盖 data.yaml 中的 output_dir",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=0,
        help="Process at most N source lines; intended only for smoke tests",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print progress after this many source lines",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing processed split in the selected output directory",
    )
    return parser.parse_args()


def require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需的 YAML 映射字段；类型不符时在入口处立即失败。"""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration key {key!r} must be a mapping")
    return value


def resolve_input(
    argument: str,
    data_config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Path, str, dict[str, Any] | None]:
    """把 full/mini/自定义路径解析为源文件和对应完整性元数据。"""

    manifest_files = require_mapping(manifest, "files")
    if argument == "full":
        item = require_mapping(manifest_files, "full")
        return resolve_v1_path(data_config["source_file"]), "full", item
    if argument == "mini":
        item = require_mapping(manifest_files, "mini")
        return resolve_v1_path(data_config["smoke_file"]), "mini", item
    return Path(argument).expanduser().resolve(), "custom", None


def load_manifest(path: Path) -> dict[str, Any]:
    """加载独立的数据来源与完整性清单。"""

    if not path.is_file():
        raise FileNotFoundError(f"data manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported data manifest: {path}")
    return manifest


def validate_source_file(
    source_path: Path,
    expected: dict[str, Any] | None,
) -> None:
    """在耗时处理前验证文件存在性和正式文件字节数。

    部分读取仍检查整个文件的字节数；完整运行结束后继续核对行数和 SHA-256。
    """

    if not source_path.is_file():
        raise FileNotFoundError(f"source corpus does not exist: {source_path}")
    if expected is None:
        return
    expected_bytes = expected.get("bytes")
    if expected_bytes is not None and source_path.stat().st_size != int(expected_bytes):
        raise ValueError(
            "source file size does not match manifest: "
            f"expected {expected_bytes}, found {source_path.stat().st_size}"
        )


def stable_validation_assignment(
    text_digest: bytes,
    *,
    seed: int,
    validation_ratio: float,
) -> bool:
    """根据正文哈希稳定决定记录是否进入 validation。

    修改 seed 或 validation_ratio 会改变划分；文件顺序变化不会影响同一文本
    的归属。使用 64-bit 阈值可以避免按总行数抽样带来的顺序依赖。
    """

    seed_bytes = seed.to_bytes(8, byteorder="big", signed=False)
    split_digest = hashlib.sha256(seed_bytes + text_digest).digest()
    bucket = int.from_bytes(split_digest[:8], byteorder="big", signed=False)
    threshold = int(validation_ratio * (1 << 64))
    return bucket < threshold


def jsonl_line(text_field: str, text: str) -> str:
    """序列化为紧凑 UTF-8 JSONL，保留非 ASCII 正文以便人工检查。"""

    return (
        json.dumps(
            {text_field: text},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def remove_if_exists(path: Path) -> None:
    """只删除调用方已经解析出的具体文件，不递归处理目录。"""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    """执行完整的流式预处理并原子发布三个正式产物。"""

    args = parse_args()
    if args.max_documents < 0:
        raise ValueError("--max-documents must be non-negative")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    # 第一阶段：加载并验证配置。尽早报错，避免处理数小时后才发现参数无效。
    config = load_yaml_config(args.config)
    if config.get("schema_version") != 1:
        raise ValueError("only schema_version: 1 is supported")
    data_config = config
    cleaning = require_mapping(data_config, "cleaning")
    manifest_path = resolve_v1_path(data_config["manifest_file"])
    manifest = load_manifest(manifest_path)

    text_field = str(data_config.get("text_field", "text"))
    source_path, source_variant, expected = resolve_input(
        args.input,
        data_config,
        manifest,
    )
    limited_run = args.max_documents > 0
    validate_source_file(source_path, expected)

    # --max-documents 只能写入显式测试目录，防止部分结果覆盖正式语料。
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else resolve_v1_path(data_config["output_dir"])
    )
    if limited_run and args.output_dir is None:
        raise ValueError(
            "--max-documents requires --output-dir so a partial run cannot replace "
            "the formal processed corpus"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / str(data_config["train_file"])
    validation_path = output_dir / str(data_config["validation_file"])
    report_path = output_dir / str(data_config["report_file"])
    database_path = output_dir / ".dedup.sqlite3"
    temporary_train = train_path.with_suffix(train_path.suffix + ".tmp")
    temporary_validation = validation_path.with_suffix(
        validation_path.suffix + ".tmp"
    )
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")

    # 正式文件和中断遗留的临时文件都受保护。默认拒绝覆盖；只有显式
    # --force 才清理这些精确路径。
    protected_paths = (train_path, validation_path, report_path)
    working_paths = (
        temporary_train,
        temporary_validation,
        temporary_report,
        database_path,
        database_path.with_name(database_path.name + "-wal"),
        database_path.with_name(database_path.name + "-shm"),
    )
    existing = [
        path for path in (*protected_paths, *working_paths) if path.exists()
    ]
    if existing and not args.force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"processed outputs already exist: {joined}; use --force to replace them"
        )
    if args.force:
        for path in (
            *protected_paths,
            *working_paths,
        ):
            remove_if_exists(path)

    # 第二阶段：验证所有会改变正文、去重或划分结果的配置。
    validation_ratio = float(data_config["validation_ratio"])
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    split_seed = int(data_config["split_seed"])
    if data_config.get("split_method") != "stable_text_hash":
        raise ValueError(
            "only split_method: stable_text_hash is supported"
        )
    deduplicate = bool(data_config.get("exact_deduplication", True))
    unicode_normalization = cleaning.get("unicode_normalization")
    if unicode_normalization not in {None, "NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(
            "unicode_normalization must be null, NFC, NFD, NFKC, or NFKD"
        )
    if bool(cleaning.get("simplified_traditional_conversion", False)):
        raise ValueError(
            "simplified/traditional conversion is not supported by this pipeline"
        )
    reservoir_size = int(data_config.get("statistics_sample_size", 100_000))
    length_sample = ReservoirSampler(reservoir_size, split_seed)

    counts = {
        "source_lines": 0,
        "valid_json_records": 0,
        "invalid_json_records": 0,
        "empty_after_cleaning": 0,
        "exact_duplicates": 0,
        "train_documents": 0,
        "validation_documents": 0,
    }
    invalid_examples: list[dict[str, Any]] = []
    source_hasher = hashlib.sha256()
    normalized_utf8_bytes = 0
    normalized_characters = 0
    minimum_characters: int | None = None
    maximum_characters = 0
    started_at = time.time()

    # SQLite PRIMARY KEY 提供精确去重。相比 Python set，它把 846 万条哈希
    # 放在磁盘上，避免常驻内存过大；WAL + NORMAL 在安全性和吞吐之间取平衡。
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID"
    )

    try:
        with (
            temporary_train.open("w", encoding="utf-8", newline="\n") as train_file,
            temporary_validation.open(
                "w", encoding="utf-8", newline="\n"
            ) as validation_file,
        ):
            # 第三阶段：单次流式扫描同时完成源文件哈希、清理、去重、划分和统计。
            for line_number, value, raw_line in iter_jsonl_texts(
                source_path,
                text_field=text_field,
                strict=False,
            ):
                if args.max_documents and counts["source_lines"] >= args.max_documents:
                    break

                counts["source_lines"] += 1
                source_hasher.update(raw_line)

                if isinstance(value, JsonlRecordError):
                    counts["invalid_json_records"] += 1
                    if len(invalid_examples) < 20:
                        invalid_examples.append(
                            {
                                "line_number": line_number,
                                "message": value.message,
                            }
                        )
                    continue

                counts["valid_json_records"] += 1
                text = normalize_text(
                    value,
                    normalize_line_endings=bool(
                        cleaning.get("normalize_line_endings", True)
                    ),
                    remove_nul=bool(cleaning.get("remove_nul", True)),
                    remove_control_characters=bool(
                        cleaning.get("remove_control_characters", True)
                    ),
                    strip_outer_whitespace=bool(
                        cleaning.get("strip_outer_whitespace", True)
                    ),
                    unicode_normalization=unicode_normalization,
                    lowercase=bool(cleaning.get("lowercase", False)),
                )
                if not text:
                    counts["empty_after_cleaning"] += 1
                    continue

                text_digest = sha256_text(text)
                if deduplicate:
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO seen(digest) VALUES (?)",
                        (text_digest,),
                    )
                    if cursor.rowcount == 0:
                        counts["exact_duplicates"] += 1
                        continue

                character_count = len(text)
                encoded_size = len(text.encode("utf-8"))
                normalized_characters += character_count
                normalized_utf8_bytes += encoded_size
                length_sample.add(character_count)
                minimum_characters = (
                    character_count
                    if minimum_characters is None
                    else min(minimum_characters, character_count)
                )
                maximum_characters = max(maximum_characters, character_count)

                # 同一规范化文本始终进入同一分区，避免重复内容跨集合泄漏。
                if stable_validation_assignment(
                    text_digest,
                    seed=split_seed,
                    validation_ratio=validation_ratio,
                ):
                    validation_file.write(jsonl_line(text_field, text))
                    counts["validation_documents"] += 1
                else:
                    train_file.write(jsonl_line(text_field, text))
                    counts["train_documents"] += 1

                if counts["source_lines"] % args.progress_every == 0:
                    connection.commit()
                    elapsed = max(time.time() - started_at, 1e-9)
                    print(
                        f"processed={counts['source_lines']:,} "
                        f"train={counts['train_documents']:,} "
                        f"validation={counts['validation_documents']:,} "
                        f"duplicates={counts['exact_duplicates']:,} "
                        f"speed={counts['source_lines'] / elapsed:,.0f} lines/s",
                        flush=True,
                    )

        connection.commit()
    finally:
        connection.close()

    # 第四阶段：只有完整源文件的行数和哈希都符合 manifest，才允许发布结果。
    elapsed_seconds = time.time() - started_at
    if expected is not None and not limited_run:
        expected_documents = expected.get("documents")
        if (
            expected_documents is not None
            and counts["source_lines"] != int(expected_documents)
        ):
            raise RuntimeError(
                "source document count does not match manifest: "
                f"expected {expected_documents}, found {counts['source_lines']}"
            )
        expected_sha256 = expected.get("sha256")
        if (
            expected_sha256 is not None
            and source_hasher.hexdigest() != str(expected_sha256)
        ):
            raise RuntimeError(
                "source SHA-256 does not match manifest: "
                f"expected {expected_sha256}, found {source_hasher.hexdigest()}"
            )

    sampled_lengths = length_sample.items
    unique_documents = counts["train_documents"] + counts["validation_documents"]
    report = {
        "schema_version": 1,
        "source": {
            "path": str(source_path),
            "variant": source_variant,
            "bytes": source_path.stat().st_size,
            "processed_sha256": source_hasher.hexdigest(),
            "sha256_scope": (
                "processed prefix"
                if limited_run
                else "complete source file"
            ),
            "text_field": text_field,
            "max_documents": args.max_documents or None,
            "manifest": str(manifest_path),
            "provider": manifest.get("provider"),
            "repo_id": manifest.get("repo_id"),
            "revision": manifest.get("revision"),
        },
        "split": {
            "method": "sha256(seed || normalized_text_sha256)",
            "seed": split_seed,
            "validation_ratio": validation_ratio,
        },
        "cleaning": dict(cleaning),
        "counts": counts,
        "text_statistics": {
            "unique_documents": unique_documents,
            "characters": normalized_characters,
            "utf8_bytes": normalized_utf8_bytes,
            "characters_per_document_mean": (
                normalized_characters / unique_documents
                if unique_documents
                else None
            ),
            "characters_min": minimum_characters,
            "characters_max": maximum_characters,
            "characters_p50_approx": percentile(sampled_lengths, 0.50),
            "characters_p90_approx": percentile(sampled_lengths, 0.90),
            "characters_p95_approx": percentile(sampled_lengths, 0.95),
            "characters_p99_approx": percentile(sampled_lengths, 0.99),
            "statistics_sample_size": len(sampled_lengths),
        },
        "invalid_examples": invalid_examples,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "source_lines_per_second": (
                counts["source_lines"] / elapsed_seconds
                if elapsed_seconds > 0
                else None
            ),
        },
        "outputs": {
            "train": str(train_path),
            "validation": str(validation_path),
            "report": str(report_path),
        },
    }

    # 先写 .tmp，再通过 os.replace 原子替换；进程中断不会留下“看似完成”的
    # train.jsonl、validation.jsonl 或报告。
    with temporary_report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    os.replace(temporary_train, train_path)
    os.replace(temporary_validation, validation_path)
    os.replace(temporary_report, report_path)
    for path in (
        database_path,
        database_path.with_name(database_path.name + "-wal"),
        database_path.with_name(database_path.name + "-shm"),
    ):
        remove_if_exists(path)

    print(
        json.dumps(
            {
                "train_documents": counts["train_documents"],
                "validation_documents": counts["validation_documents"],
                "exact_duplicates": counts["exact_duplicates"],
                "invalid_json_records": counts["invalid_json_records"],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
