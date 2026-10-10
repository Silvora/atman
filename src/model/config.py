"""Decoder-only Transformer 配置和参数量工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """模型结构参数。

    ``max_seq_len`` 必须与编码数据的 sequence_length 一致。v1 使用学习型
    位置嵌入、Pre-LN、GELU 和标准密集多头注意力，保持结构简单且便于验证。
    """

    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    dropout: float = 0.0
    bias: bool = True
    tie_embeddings: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ModelConfig":
        """从 YAML 映射读取配置，并在模型创建前检查结构约束。"""

        config = cls(
            vocab_size=int(value["vocab_size"]),
            max_seq_len=int(value["max_seq_len"]),
            d_model=int(value["d_model"]),
            n_layers=int(value["n_layers"]),
            n_heads=int(value["n_heads"]),
            ffn_dim=int(value["ffn_dim"]),
            dropout=float(value.get("dropout", 0.0)),
            bias=bool(value.get("bias", True)),
            tie_embeddings=bool(value.get("tie_embeddings", True)),
        )
        if min(
            config.vocab_size,
            config.max_seq_len,
            config.d_model,
            config.n_layers,
            config.n_heads,
            config.ffn_dim,
        ) <= 0:
            raise ValueError("all model dimensions must be positive")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return config

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。"""

        return self.d_model // self.n_heads


def count_parameters(module: Any, *, trainable_only: bool = False) -> int:
    """统计参数量，供日志和配置验收使用。"""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
