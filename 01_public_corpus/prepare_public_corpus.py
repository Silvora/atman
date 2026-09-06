"""Step 01: build the Atman public pretraining corpus.

本步骤只负责数据工程，不训练 tokenizer，也不训练模型：

1. 从多个公开数据集流式读取中文文本。
2. 统一清洗并过滤明显的乱码、短文本和非中文文本。
3. 在全部来源之间做内容指纹去重。
4. 按来源输出 JSONL 分片，并生成 manifest.json 供后续步骤读取。

默认不预设清洗后的语料总大小：脚本只处理 sources.json 明确列出的文件，并
保留其中所有通过质量检查的文本。采用 streaming=True 边下载边处理；需要做
更小的试验时，还可以显式设置每个来源的容量上限或样本数上限。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STEP_ID = "01_public_corpus"
STEP_NAME = "Public Corpus"
STEP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = STEP_DIR / "output"
DEFAULT_SOURCES_CONFIG = STEP_DIR / "sources.json"
GIB = 1024**3


@dataclass(frozen=True)
class SourceSpec:
    """描述一个 Hugging Face 数据源及其读取方式。"""

    source_id: str
    dataset: str
    url: str
    license_note: str
    text_fields: tuple[str, ...] = ("text", "content")
    title_fields: tuple[str, ...] = ("title",)
    config: str | None = None
    split: str = "train"
    data_files: str | tuple[str, ...] | None = None
    download_bytes: int | None = None
    requires_access: bool = False


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"<[^>]{1,500}>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
HASH_NORMALIZE_RE = re.compile(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")


def utc_now() -> str:
    """返回可写入清单的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def clean_text(text: str) -> str:
    """轻量清洗正文，同时尽量保留段落结构和真实语言风格。"""
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)

    # 邮箱、手机号和 URL 对基础语言能力帮助很小，同时可能携带隐私信息。
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = URL_RE.sub("<URL>", text)

    lines: list[str] = []
    previous = None
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t\u3000]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line

    return "\n".join(lines).strip()


def quality_rejection_reason(
    text: str,
    min_chars: int,
    min_chinese_ratio: float,
) -> str | None:
    """返回拒绝原因；返回 None 表示文本通过基础质量检查。"""
    if not text:
        return "empty"
    if len(text) < min_chars:
        return "short"
    if text.count("\ufffd") / len(text) > 0.001:
        return "garbled"

    useful_chars = re.findall(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]", text)
    if not useful_chars:
        return "low_information"
    chinese_count = len(CHINESE_RE.findall(text))
    if chinese_count / len(useful_chars) < min_chinese_ratio:
        return "low_chinese_ratio"

    # 大量重复单字通常来自乱码、装饰线或抓取失败页面。
    if len(useful_chars) >= 200:
        most_common = Counter(useful_chars).most_common(1)[0][1]
        if most_common / len(useful_chars) > 0.20:
            return "repetitive"
    return None


def split_long_text(text: str, max_chars: int) -> Iterable[str]:
    """将异常长文档按段落切开，避免单条样本无限增大。"""
    if len(text) <= max_chars:
        yield text
        return

    current: list[str] = []
    current_size = 0
    for paragraph in text.split("\n"):
        if current and current_size + len(paragraph) + 1 > max_chars:
            yield "\n".join(current).strip()
            current = []
            current_size = 0

        # 极长且没有换行的段落只能按字符长度硬切分。
        while len(paragraph) > max_chars:
            if current:
                yield "\n".join(current).strip()
                current = []
                current_size = 0
            yield paragraph[:max_chars]
            paragraph = paragraph[max_chars:]

        if paragraph:
            current.append(paragraph)
            current_size += len(paragraph) + 1

    if current:
        yield "\n".join(current).strip()


def text_fingerprint(text: str) -> bytes:
    """生成跨来源去重指纹；忽略空白和标点造成的无意义差异。"""
    normalized = HASH_NORMALIZE_RE.sub("", text).casefold()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


def extract_text(sample: dict[str, Any], source: SourceSpec) -> str | None:
    """兼容不同数据集常见的 text/content/title 字段。"""
    body = next(
        (
            sample[field]
            for field in source.text_fields
            if isinstance(sample.get(field), str) and sample[field].strip()
        ),
        None,
    )
    if body is None:
        return None

    title = next(
        (
            sample[field].strip()
            for field in source.title_fields
            if isinstance(sample.get(field), str) and sample[field].strip()
        ),
        "",
    )
    if title and not body.lstrip().startswith(title):
        return f"{title}\n\n{body}"
    return body


def iter_source(source: SourceSpec, seed: int, shuffle_buffer: int):
    """以流式模式打开数据集，避免先下载数百 GB 的完整仓库。"""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
    }
    if source.config:
        kwargs["name"] = source.config
    if source.data_files:
        # 显式绑定到配置中的 split，避免自定义文件被 datasets 默认归入 train。
        selected_files = (
            list(source.data_files)
            if isinstance(source.data_files, tuple)
            else source.data_files
        )
        kwargs["data_files"] = {source.split: selected_files}

    # HF_TOKEN 不写进代码和命令行，避免 token 出现在 shell 历史中。
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token

    dataset = load_dataset(**kwargs)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def load_sources_config(path: str | Path) -> tuple[list[SourceSpec], str]:
    """读取并校验数据源配置，返回来源列表和配置哈希。"""
    config_path = Path(path).expanduser().resolve()
    raw_bytes = config_path.read_bytes()
    try:
        config = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法解析数据源配置 {config_path}: {error}") from error

    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("数据源配置必须包含非空 sources 数组")

    sources: list[SourceSpec] = []
    seen_ids: set[str] = set()
    required_fields = ("source_id", "dataset", "url", "license_note", "data_files")
    for index, item in enumerate(raw_sources, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"sources[{index}] 必须是 JSON 对象")
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise ValueError(f"sources[{index}] 缺少字段: {', '.join(missing)}")

        source_id = str(item["source_id"])
        if not re.fullmatch(r"[a-z0-9_]+", source_id):
            raise ValueError(f"source_id 只能包含小写字母、数字和下划线: {source_id}")
        if source_id in seen_ids:
            raise ValueError(f"source_id 重复: {source_id}")

        raw_data_files = item["data_files"]
        if isinstance(raw_data_files, str) and raw_data_files.strip():
            data_files: str | tuple[str, ...] = raw_data_files
        elif (
            isinstance(raw_data_files, list)
            and raw_data_files
            and all(isinstance(path, str) and path.strip() for path in raw_data_files)
        ):
            # tuple 是不可变的，和 frozen dataclass 的语义一致；datasets 同样接受它。
            data_files = tuple(raw_data_files)
        else:
            raise ValueError(f"{source_id}.data_files 必须是非空字符串或字符串数组")

        raw_download_bytes = item.get("download_bytes")
        download_bytes = None
        if raw_download_bytes is not None:
            try:
                download_bytes = int(raw_download_bytes)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source_id}.download_bytes 必须是整数") from error
            if download_bytes <= 0:
                raise ValueError(f"{source_id}.download_bytes 必须大于 0")

        sources.append(
            SourceSpec(
                source_id=source_id,
                dataset=str(item["dataset"]),
                url=str(item["url"]),
                license_note=str(item["license_note"]),
                text_fields=tuple(item.get("text_fields", ("text", "content"))),
                title_fields=tuple(item.get("title_fields", ("title",))),
                config=item.get("config"),
                split=str(item.get("split", "train")),
                data_files=data_files,
                download_bytes=download_bytes,
                requires_access=bool(item.get("requires_access", False)),
            )
        )
        seen_ids.add(source_id)

    config_hash = hashlib.sha256(raw_bytes).hexdigest()
    return sources, config_hash


def select_sources(value: str, available_sources: list[SourceSpec]) -> list[SourceSpec]:
    """解析 --sources；all 表示按配置顺序处理全部来源。"""
    if value.strip().lower() == "all":
        return list(available_sources)

    requested = [item.strip() for item in value.split(",") if item.strip()]
    known = {source.source_id: source for source in available_sources}
    unknown = [source_id for source_id in requested if source_id not in known]
    if unknown:
        raise ValueError(f"未知数据源: {', '.join(unknown)}")
    if not requested:
        raise ValueError("--sources 不能为空")
    return [known[source_id] for source_id in requested]


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """原子更新清单，进程中断时不会只留下半个 JSON 文件。"""
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temp_path.replace(path)


def load_existing_fingerprints(
    output_dir: Path,
    manifest: dict[str, Any],
) -> set[bytes]:
    """恢复已完成来源的指纹，使中断重跑仍能跨来源去重。"""
    seen: set[bytes] = set()
    completed = [
        result
        for result in manifest.get("source_results", [])
        if result.get("status") == "completed"
    ]
    for result in completed:
        shard_path = output_dir / result["file"]
        if not shard_path.is_file():
            raise FileNotFoundError(f"清单记录为已完成，但文件不存在: {shard_path}")
        print(f"恢复去重指纹: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                seen.add(text_fingerprint(record["text"]))
    return seen


def initial_manifest(
    sources: list[SourceSpec],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """建立可追踪、可恢复的第1步输出清单。"""
    return {
        "generated_by_step": STEP_ID,
        "generated_by_step_name": STEP_NAME,
        "sources_config": str(Path(args.sources_config).expanduser().resolve()),
        "sources_config_sha256": args.sources_config_sha256,
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        # null 表示完整处理；数字表示每个来源最多保留多少正文。
        "max_text_bytes_per_source": (
            int(args.max_gb_per_source * GIB) if args.max_gb_per_source else None
        ),
        "max_text_gib_per_source": args.max_gb_per_source,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_chinese_ratio": args.min_chinese_ratio,
        },
        "selected_sources": [source.source_id for source in sources],
        "selected_download_bytes": sum(source.download_bytes or 0 for source in sources),
        "selected_download_gb": round(
            sum(source.download_bytes or 0 for source in sources) / 1_000_000_000,
            3,
        ),
        "selected_download_gib": round(
            sum(source.download_bytes or 0 for source in sources) / GIB,
            3,
        ),
        "source_catalog": [asdict(source) for source in sources],
        "source_results": [],
        "totals": {},
    }


def validate_resume(
    manifest: dict[str, Any],
    sources: list[SourceSpec],
    args: argparse.Namespace,
) -> None:
    """只允许相同配置续跑，避免把两套实验意外混进同一目录。"""
    expected_sources = [source.source_id for source in sources]
    expected_max_bytes = (
        int(args.max_gb_per_source * GIB) if args.max_gb_per_source else None
    )
    expected_filters = {
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "min_chinese_ratio": args.min_chinese_ratio,
    }
    if manifest.get("sources_config_sha256") != args.sources_config_sha256:
        raise ValueError("sources.json 已改变，请清空本步骤 output 后重新开始")
    if manifest.get("selected_sources") != expected_sources:
        raise ValueError("续跑时 --sources 必须与已有 manifest.json 完全一致")
    if manifest.get("max_text_bytes_per_source") != expected_max_bytes:
        raise ValueError("续跑时 --max-gb-per-source 必须与已有 manifest.json 完全一致")
    if manifest.get("filters") != expected_filters:
        raise ValueError("续跑时过滤参数必须与已有 manifest.json 完全一致")


def process_source(
    source: SourceSpec,
    max_text_bytes: int | None,
    output_dir: Path,
    seen: set[bytes],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """处理一个来源并写入独立分片，返回该来源的完整统计。"""
    from tqdm import tqdm

    final_path = output_dir / f"corpus_{source.source_id}.jsonl"
    partial_path = final_path.with_suffix(".jsonl.partial")
    if partial_path.exists():
        # partial 文件没有可靠断点位置，重新处理该来源比拼接重复数据更安全。
        partial_path.unlink()

    stats = {
        "processed": 0,
        "kept": 0,
        "text_bytes": 0,
        "empty": 0,
        "short": 0,
        "garbled": 0,
        "low_information": 0,
        "low_chinese_ratio": 0,
        "repetitive": 0,
        "duplicate": 0,
    }
    dataset = iter_source(source, args.seed, args.shuffle_buffer)

    completion_reason = "dataset_exhausted"
    source_limit_reached = False
    with partial_path.open("w", encoding="utf-8") as output_file:
        progress = tqdm(dataset, desc=source.source_id, unit="doc", dynamic_ncols=True)
        for sample in progress:
            if args.max_samples_per_source and stats["processed"] >= args.max_samples_per_source:
                completion_reason = "max_samples_reached"
                break
            if max_text_bytes is not None and stats["text_bytes"] >= max_text_bytes:
                completion_reason = "max_gb_reached"
                break

            stats["processed"] += 1
            raw_text = extract_text(sample, source)
            if raw_text is None:
                stats["empty"] += 1
                continue

            text = clean_text(raw_text)
            reason = quality_rejection_reason(
                text,
                min_chars=args.min_chars,
                min_chinese_ratio=args.min_chinese_ratio,
            )
            if reason:
                stats[reason] += 1
                continue

            for chunk in split_long_text(text, args.max_chars):
                if len(chunk) < args.min_chars:
                    stats["short"] += 1
                    continue
                fingerprint = text_fingerprint(chunk)
                if fingerprint in seen:
                    stats["duplicate"] += 1
                    continue

                chunk_bytes = len(chunk.encode("utf-8"))
                if (
                    max_text_bytes is not None
                    and stats["text_bytes"] + chunk_bytes > max_text_bytes
                ):
                    completion_reason = "max_gb_reached"
                    source_limit_reached = True
                    break
                record = {
                    "text": chunk,
                    "source": source.source_id,
                    "generated_by_step": STEP_ID,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen.add(fingerprint)
                stats["kept"] += 1
                stats["text_bytes"] += chunk_bytes

                if max_text_bytes is not None and stats["text_bytes"] >= max_text_bytes:
                    completion_reason = "max_gb_reached"
                    source_limit_reached = True
                    break

            if source_limit_reached:
                break

            if stats["processed"] % 200 == 0:
                postfix = {
                    "kept": stats["kept"],
                    "text_gib": f"{stats['text_bytes'] / GIB:.2f}",
                    "duplicate": stats["duplicate"],
                }
                if max_text_bytes is not None:
                    postfix["max_gib"] = f"{max_text_bytes / GIB:.2f}"
                progress.set_postfix(**postfix)

    partial_path.replace(final_path)
    return {
        "source_id": source.source_id,
        "status": "completed",
        "completion_reason": completion_reason,
        "file": final_path.name,
        "max_text_bytes": max_text_bytes,
        "max_text_gib": round(max_text_bytes / GIB, 3) if max_text_bytes else None,
        "output_file_bytes": final_path.stat().st_size,
        **stats,
        "text_gib": round(stats["text_bytes"] / GIB, 3),
        "completed_at": utc_now(),
    }


def update_totals(manifest: dict[str, Any]) -> None:
    """将所有已完成来源的统计汇总到 manifest.totals。"""
    completed = [
        result
        for result in manifest["source_results"]
        if result.get("status") == "completed"
    ]
    manifest["totals"] = {
        "completed_sources": len(completed),
        "documents": sum(result.get("kept", 0) for result in completed),
        "text_bytes": sum(result.get("text_bytes", 0) for result in completed),
        "text_gib": round(sum(result.get("text_bytes", 0) for result in completed) / GIB, 3),
        "output_file_bytes": sum(result.get("output_file_bytes", 0) for result in completed),
    }


def prepare_public_corpus(
    args: argparse.Namespace,
    available_sources: list[SourceSpec],
) -> None:
    """执行 Atman 的第1步。"""
    sources = select_sources(args.sources, available_sources)
    max_text_bytes = (
        int(args.max_gb_per_source * GIB) if args.max_gb_per_source else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_resume(manifest, sources, args)
        print(f"发现已有清单，将按来源继续: {manifest_path.resolve()}")
    else:
        existing_shards = list(output_dir.glob("corpus_*.jsonl"))
        if existing_shards:
            raise FileExistsError(
                f"输出目录已有语料分片但没有 manifest.json，请换一个 --output-dir: {output_dir}"
            )
        manifest = initial_manifest(sources, args)
        write_manifest(manifest_path, manifest)

    completed_ids = {
        result["source_id"]
        for result in manifest["source_results"]
        if result.get("status") == "completed"
    }
    if len(completed_ids) == len(sources):
        totals = manifest.get("totals", {})
        print(f"第1步已经处理完成: {manifest_path.resolve()}")
        print(f"有效正文: {totals.get('text_gib', 0):.3f} GiB")
        return

    seen = load_existing_fingerprints(output_dir, manifest)

    print(f"输出目录: {output_dir.resolve()}")
    if args.max_gb_per_source:
        print(f"每个来源正文上限: {args.max_gb_per_source:.2f} GiB")
    else:
        print("正文容量: 不设固定目标，处理完每个来源")
    print(f"数据来源: {len(sources)} 个")

    for source in sources:
        if source.source_id in completed_ids:
            print(f"跳过已完成来源: {source.source_id}")
            continue

        size_message = (
            f"每来源上限 {max_text_bytes / GIB:.2f} GiB"
            if max_text_bytes is not None
            else "处理到数据集结束"
        )
        print(
            f"\n处理 {source.source_id}: "
            f"{size_message}\n"
            f"地址 {source.url}\n"
            f"许可 {source.license_note}"
        )
        try:
            result = process_source(
                source=source,
                max_text_bytes=max_text_bytes,
                output_dir=output_dir,
                seen=seen,
                args=args,
            )
        except BaseException as error:
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["last_error"] = {
                "source_id": source.source_id,
                "type": type(error).__name__,
                "message": str(error),
                "time": utc_now(),
            }
            manifest["updated_at"] = utc_now()
            update_totals(manifest)
            write_manifest(manifest_path, manifest)
            raise

        manifest["source_results"].append(result)
        manifest.pop("last_error", None)
        manifest["updated_at"] = utc_now()
        update_totals(manifest)
        write_manifest(manifest_path, manifest)

    update_totals(manifest)
    manifest["status"] = "completed"
    manifest["updated_at"] = utc_now()
    write_manifest(manifest_path, manifest)

    totals = manifest["totals"]
    print("\nStep 01 完成")
    print(f"保留文档: {totals['documents']:,}")
    print(f"有效正文: {totals['text_gib']:.3f} GiB")
    print(f"输出清单: {manifest_path.resolve()}")


def print_sources(sources: list[SourceSpec]) -> None:
    """列出配置中登记的全部来源、地址和访问要求。"""
    for index, source in enumerate(sources, start=1):
        access = "需先授权" if source.requires_access else "公开读取"
        files_count = len(source.data_files) if isinstance(source.data_files, tuple) else 1
        estimated_size = (
            f", 约 {source.download_bytes / 1_000_000_000:.3f} GB"
            if source.download_bytes is not None
            else ""
        )
        print(f"{index:02d}. {source.source_id} ({access}, {files_count} 个文件{estimated_size})")
        print(f"    dataset: {source.dataset}")
        print(f"    url:     {source.url}")
        print(f"    license: {source.license_note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 01: stream, clean, filter, deduplicate, and shard public Chinese corpora"
    )
    parser.add_argument(
        "--sources-config",
        default=str(DEFAULT_SOURCES_CONFIG),
        help=f"数据源 JSON 配置，默认 {DEFAULT_SOURCES_CONFIG}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-gb-per-source",
        type=float,
        default=None,
        help="每个来源最多保留的有效正文 GiB；默认不限",
    )
    parser.add_argument(
        "--sources",
        default="all",
        help="all 或逗号分隔的 source_id，默认处理全部来源",
    )
    parser.add_argument("--min-chars", type=int, default=200, help="最短文档字符数")
    parser.add_argument("--max-chars", type=int, default=100_000, help="单条输出的最大字符数")
    parser.add_argument(
        "--min-chinese-ratio",
        type=float,
        default=0.30,
        help="有效字符中中文字符的最低比例",
    )
    parser.add_argument("--seed", type=int, default=42, help="流式洗牌随机种子")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10_000,
        help="流式洗牌缓冲区；0 表示不洗牌",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="每个来源最多检查多少条；0 表示不限，主要用于试跑",
    )
    parser.add_argument("--list-sources", action="store_true", help="只显示数据源清单")
    args = parser.parse_args()

    if args.max_gb_per_source is not None and args.max_gb_per_source <= 0:
        parser.error("--max-gb-per-source 必须大于 0")
    if args.max_samples_per_source < 0:
        parser.error("--max-samples-per-source 不能小于 0")
    if args.shuffle_buffer < 0:
        parser.error("--shuffle-buffer 不能小于 0")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        parser.error("字符长度参数不合法")
    if not 0 <= args.min_chinese_ratio <= 1:
        parser.error("--min-chinese-ratio 必须在 0 到 1 之间")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    configured_sources, sources_config_hash = load_sources_config(cli_args.sources_config)
    cli_args.sources_config_sha256 = sources_config_hash
    if cli_args.list_sources:
        print_sources(configured_sources)
    else:
        prepare_public_corpus(cli_args, configured_sources)
