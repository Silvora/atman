"""语料处理公共接口；脚本应从这里导入稳定能力。"""

from .corpus import (
    JsonlRecordError,
    ReservoirSampler,
    iter_jsonl_texts,
    normalize_text,
    percentile,
    sha256_text,
)

__all__ = [
    "JsonlRecordError",
    "ReservoirSampler",
    "iter_jsonl_texts",
    "normalize_text",
    "percentile",
    "sha256_text",
]
from .packed import PackedTokenDataset

__all__.append("PackedTokenDataset")
