"""v1 的 Pre-LN Decoder-only Transformer。"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """标准密集多头因果注意力。

    使用 PyTorch 2.x 的 scaled_dot_product_attention，在 CUDA 上可以自动选择
    更高效的 kernel；is_causal=True 保证当前位置不会读取未来 token。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.output = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, channels = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        # [batch, sequence, channels] -> [batch, heads, sequence, head_dim]
        query = query.view(
            batch_size, sequence_length, self.config.n_heads, self.config.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.config.n_heads, self.config.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.config.n_heads, self.config.head_dim
        ).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.output(attention)


class MLP(nn.Module):
    """Transformer 前馈网络，使用 GELU 激活。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input = nn.Linear(config.d_model, config.ffn_dim, bias=config.bias)
        self.activation = nn.GELU()
        self.output = nn.Linear(config.ffn_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.output(self.activation(self.input(hidden_states))))


class TransformerBlock(nn.Module):
    """一个 Pre-LN Transformer block。"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(
            self.norm_attention(hidden_states)
        )
        hidden_states = hidden_states + self.mlp(self.norm_mlp(hidden_states))
        return hidden_states


class DecoderOnlyTransformer(nn.Module):
    """用于 Causal Language Modeling 的 Decoder-only Transformer。

    输入形状为 ``[batch, sequence]``，输出形状为 ``[batch, sequence, vocab]``。
    labels 由数据集提前右移后传入，模型内部只计算交叉熵，不自动添加特殊 token。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            # 权重共享可以明显减少 embedding 与输出层参数量。
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """使用 GPT 类正态初始化；线性层和嵌入层共享同一初始化尺度。"""

        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """执行前向计算并可选计算 next-token 交叉熵。"""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
            dtype=torch.long,
        )
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions)[None, :, :]
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return logits, loss

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DecoderOnlyTransformer":
        """从 YAML 的 model 映射创建模型。"""

        return cls(ModelConfig.from_mapping(value))

    def estimate_parameters(self) -> int:
        """返回模型参数量，供 dry-run 和训练日志使用。"""

        return sum(parameter.numel() for parameter in self.parameters())
