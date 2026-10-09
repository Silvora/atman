"""读取 encode_data.py 生成的定长 uint16 token 数据。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """把扁平二进制 token ID 映射为 input/label 样本。

    文件中的每条记录包含 sequence_length + 1 个 ID。前 sequence_length 个
    是输入，后 sequence_length 个是右移后的标签。每次取样只复制一个样本，
    不会把整份训练文件载入内存。
    """

    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.path.is_file():
            raise FileNotFoundError(f"encoded dataset does not exist: {self.path}")

        self._tokens = np.memmap(
            self.path,
            mode="r",
            dtype=np.dtype("<u2"),
        )
        stored_tokens = self.sequence_length + 1
        if len(self._tokens) % stored_tokens != 0:
            raise ValueError(
                f"{self.path} contains {len(self._tokens)} tokens, which is not "
                f"divisible by {stored_tokens}"
            )
        self._length = len(self._tokens) // stored_tokens

    def __len__(self) -> int:
        """返回定长样本数。"""

        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """复制一条样本并返回 input_ids 与 labels。"""

        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        stored_tokens = self.sequence_length + 1
        start = index * stored_tokens
        # copy=True 避免 torch 从只读 memmap 创建可写性不明确的 tensor。
        values = np.array(
            self._tokens[start : start + stored_tokens],
            dtype=np.int64,
            copy=True,
        )
        return (
            torch.from_numpy(values[:-1]),
            torch.from_numpy(values[1:]),
        )
