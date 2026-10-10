"""v1 Decoder-only Transformer 模型。"""

from .config import ModelConfig
from .decoder import DecoderOnlyTransformer

__all__ = ["DecoderOnlyTransformer", "ModelConfig"]
