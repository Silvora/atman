# Step 03: Encoding

本步骤使用第 2 步 tokenizer，把第 1 步全部 JSONL 文本编码为固定长度 token 序列。输出是 NumPy `.npy` 分片，第 4 步可以通过 mmap 按需读取，不必把整个数据集装进内存。

## 输入与输出

输入：

```text
01_public_corpus/output/manifest.json
02_tokenizer/output/atman_tokenizer/
```

输出：

```text
03_encoding/output/
├── train_00000.npy
├── train_00001.npy
├── validation_00000.npy
└── manifest.json
```

每个 `.npy` 是形状为 `[num_chunks, seq_length]` 的二维整数数组。32K 词表默认使用 `uint16`，比 `torch.long` 少 75% 磁盘空间；第 4 步取出一行时才转换为模型需要的 `int64`。

## 为什么在这里划分验证集

每篇文档先通过稳定哈希进入 `train` 或 `validation`，然后两边分别打包：

- 同一文档不会同时出现在训练集和验证集。
- 重复运行的划分保持一致。
- 第 4 步和未来第 5 步能使用完全相同的验证数据。
- 文档末尾加入 `<eos>`，让模型知道文章边界。

短文档会连续打包，以减少 padding 和 token 浪费。每个 split 最后不足一个 `seq_length` 的少量 token 会丢弃，并记录在 manifest。

## 正式运行

```bash
python 03_encoding/encode_pretrain_data.py
```

默认配置：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--seq-length` | 1024 | 每个训练序列的 token 数 |
| `--validation-ratio` | 0.01 | 文档级验证集比例 |
| `--shard-rows` | 50000 | 每个 `.npy` 最多包含的序列数 |
| `--batch-documents` | 128 | 一次提交给 fast tokenizer 的文档数 |
| `--max-documents` | 0 | 0 表示编码全部文档 |

`seq_length=1024` 是三档模型共用的稳妥起点。它比 512 能学习更长依赖，又比 2048 明显节省显存和注意力计算。

## 小规模验证

需要先用第 2 步试验 tokenizer，并把输出隔离：

```bash
python 03_encoding/encode_pretrain_data.py \
  --tokenizer-dir 02_tokenizer/output_smoke/atman_tokenizer \
  --output-dir 03_encoding/output_smoke \
  --max-documents 10000 \
  --shard-rows 1000
```

如果样本太少而没有产生验证序列，可以增大 `--max-documents`，或者仅为吞吐测试设置 `--validation-ratio 0`。

## 完成检查

```bash
python -m json.tool 03_encoding/output/manifest.json
```

进入第 4 步前确认：

- `status` 为 `completed`。
- `format_version` 为 `2`。
- `splits.train.num_tokens` 大于 0。
- `splits.validation.num_tokens` 大于 0。
- tokenizer SHA-256 与第 2 步 metadata 一致。
