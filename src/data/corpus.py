"""语料流式处理公共组件。

该模块只负责与具体模型无关的基础能力：JSONL 解析、文本清理、稳定哈希、
蓄水池抽样和分位数计算。所有函数均按单条记录工作，避免把 GB 级语料整体
载入内存。修改清理规则会改变文本哈希、数据划分和最终 Tokenizer，因此必须
重建全部下游产物。
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class JsonlRecordError:
    """无法解析的 JSONL 记录；保留行号便于定位原始数据。"""

    # 原始文件中的 1-based 行号。
    line_number: int
    # JSON 解码、类型或字段校验错误。
    message: str


def normalize_text(
    text: str,
    *,
    normalize_line_endings: bool = True,
    remove_nul: bool = True,
    remove_control_characters: bool = True,
    strip_outer_whitespace: bool = True,
    unicode_normalization: str | None = None,
    lowercase: bool = False,
) -> str:
    """按照 YAML 开关清理单篇文本。

    参数影响：
    - 换行、NUL、控制字符和首尾空白属于非语言性清理；
    - Unicode 规范化和 lowercase 会改变正文内容及 token 分布；
    - 不执行简繁转换，也不改写内部标点或空白。
    """
    if normalize_line_endings:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if remove_nul:
        text = text.replace("\x00", "")
    if remove_control_characters:
        text = "".join(
            character
            for character in text
            if character in {"\n", "\t"}
            or unicodedata.category(character) != "Cc"
        )
    if unicode_normalization is not None:
        text = unicodedata.normalize(unicode_normalization, text)
    if lowercase:
        text = text.lower()
    if strip_outer_whitespace:
        text = text.strip()
    return text


def iter_jsonl_texts(
    path: Path,
    *,
    text_field: str,
    strict: bool = True,
) -> Iterator[tuple[int, str | JsonlRecordError, bytes]]:
    """逐行返回正文或结构化错误，同时保留原始字节用于文件哈希。

    strict=True 时遇到第一条错误立即终止，适合读取已经生成的处理结果；
    strict=False 时错误作为 JsonlRecordError 返回，适合审计外部原始语料。
    """
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise TypeError("JSON value is not an object")
                text = record.get(text_field)
                if not isinstance(text, str):
                    raise TypeError(
                        f"field {text_field!r} is missing or is not a string"
                    )
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                error = JsonlRecordError(line_number, str(exc))
                if strict:
                    raise ValueError(
                        f"{path}:{line_number}: {error.message}"
                    ) from exc
                yield line_number, error, raw_line
                continue
            yield line_number, text, raw_line


def sha256_text(text: str) -> bytes:
    """返回规范化正文的 SHA-256，用于精确去重和稳定数据划分。"""

    return hashlib.sha256(text.encode("utf-8")).digest()


class ReservoirSampler:
    """固定内存的确定性蓄水池抽样器。

    capacity 越大，长度分位数越稳定，但会线性增加内存；seed 相同且输入顺序
    相同时结果完全一致。它只影响统计报告，不影响训练集内容。
    """

    def __init__(self, capacity: int, seed: int) -> None:
        """建立固定容量和固定随机种子的抽样器。"""

        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._random = random.Random(seed)
        self.items: list[int] = []
        self.seen = 0

    def add(self, value: int) -> None:
        """接收一个观测值，并按蓄水池概率保留或替换。"""

        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(value)
            return
        replacement = self._random.randrange(self.seen)
        if replacement < self.capacity:
            self.items[replacement] = value


def percentile(values: list[int], probability: float) -> float | None:
    """对内存中的统计样本做线性插值分位数计算。"""

    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
