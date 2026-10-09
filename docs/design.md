# Atman v1 设计文档

> 状态：数据过滤、划分和 Tokenizer 已完成；数据编码、模型结构与训练参数待确定。

## 1. 数据源

正式语料采用 ModelScope `gongjy/minimind_dataset`：

```text
revision: 74aad49fa4443e7ed640d44bc4e9c7d1fe71ada5
formal:   pretrain_t2t.jsonl
smoke:    pretrain_t2t_mini.jsonl
field:    text
```

正式文件包含 8,468,827 条记录，文件大小为 8,275,074,893 bytes。mini 文件只用于流程检查，不与正式文件合并，也不作为验证集。

数据仓库同时标记 Apache-2.0 和 CC-BY-NC-2.0。数据发布、模型发布及商业用途需要继续保留并核验上游许可记录。

来源、revision、许可、文件大小、行数和 SHA-256 统一保存在 `data/raw/minimind/manifest.json`；`configs/data.yaml` 只保存管线实际需要的路径和处理参数。

## 2. 数据处理

数据处理采用流式流程：

```text
JSONL 校验
  → 非语言性清洗
  → 规范化文本 SHA-256 精确去重
  → 稳定 hash 划分
  → train.jsonl / validation.jsonl
  → audit_report.json
```

清洗规则：

- 统一 CRLF/CR 为 LF；
- 删除 NUL；
- 删除换行和制表符之外的控制字符；
- 删除文本首尾空白；
- 不执行 Unicode NFKC；
- 不转小写；
- 不执行简繁转换；
- 不改写正文标点和内部空白。

精确去重使用规范化文本的 SHA-256，并通过临时 SQLite 索引控制内存占用。

数据划分：

```yaml
train: 99.5%
validation: 0.5%
seed: 42
method: sha256(seed || normalized_text_sha256)
```

划分结果只取决于文本和种子，不取决于文件顺序。

## 3. Tokenizer

算法采用 Byte-level BPE：

```yaml
normalizer: null
add_prefix_space: false
use_regex: true
min_frequency: 2
initial_alphabet: byte_level
```

特殊 token：

```text
<|endoftext|>
<|padding|>
```

不设置 BOS 和 UNK。Byte-level 初始字节表必须保证任意有效 UTF-8 文本均可编码。

所有模型规格使用同一个正式 Tokenizer。

## 4. 候选词表

使用相同训练样本生成两个候选：

```text
16,384
32,000
```

从 train 分区中使用种子 42 确定性抽取 1,000,000 条记录。抽样索引按升序流式读取，不把全部文本载入内存。

正式词表按照以下规则自动选择：

1. 32K 作为压缩率基线；
2. 计算 16K 相对 32K 的验证集 token 数增幅；
3. 如果增幅不超过 15%，选择 16K；
4. 否则选择 32K。

该规则同时控制词表参数成本和序列计算成本。

正式评测结果为：16K 在 41,999 条验证数据上产生 8,547,688 个 token，32K 产生 7,766,602 个 token；16K 相对 32K 的 token 增幅为 10.06%，低于 15% 阈值，因此最终选择 16,384 词表。两个候选的 encode/decode 往返一致率均为 100%，未知 token 数均为 0。

## 5. Tokenizer 验收

候选 Tokenizer 在 validation 分区上最多评测 50,000 条记录，至少输出：

- 总 token 数；
- 平均 tokens/document；
- characters/token；
- UTF-8 bytes/token；
- P50/P90/P95/P99 文档 token 长度；
- 超过 512/1024/2048 tokens 的文档数量；
- encode/decode 往返一致率；
- 特殊 token ID 与单 token 验证；
- 候选之间的 token 数增幅。

硬性要求：

```text
未知 token 数：0
encode/decode 往返一致率：100%
每个特殊 token：正好一个 ID
重新加载 tokenizer.json 后结果一致
```

## 6. 输出

```text
data/processed/
├── train.jsonl
├── validation.jsonl
└── audit_report.json

data/encoded/
├── train.bin
├── validation.bin
└── encoding_report.json

data/tokenizer/
├── candidates/
│   ├── byte_bpe_16384/
│   └── byte_bpe_32000/
├── comparison.json
└── final/
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.json
    ├── merges.txt
    ├── metrics.json
    ├── manifest.json
    └── selection.json
```

## 7. 执行入口

正式数据处理：

```bash
python scripts/preprocess_data.py --config configs/data.yaml
```

候选 Tokenizer 训练、评测和选择：

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
```

两条命令默认拒绝覆盖已有正式产物。只有明确需要重新生成时才使用 `--force`。

上述过滤和 Tokenizer 流程已经完成。`data/processed/*.jsonl` 保存清洗后的文本，模型训练使用其后的定长 token-ID 数据。

文本编码入口已经实现。它使用 `data/tokenizer/final/tokenizer.json`，在每篇文档末尾加入 EOS，并将连续 token 流切分为 `sequence_length + 1` 的小端 `uint16` 样本。正式编码命令为：

```bash
python scripts/encode_data.py --config configs/data.yaml
```

正式编码已经完成：`train.bin` 含 837,900 条样本，`validation.bin` 含 4,192 条样本；每条记录保存 2,049 个 `uint16` ID，用前 2,048 个作为输入、后 2,048 个作为标签。训练入口通过 memmap 读取这些文件。

预训练入口为 `scripts/train.py`，负责读取模型 YAML、构建 Decoder-only Transformer、梯度累积、混合精度、验证、日志、checkpoint 和断点恢复。正式训练前可运行：

```bash
python scripts/train.py --config configs/small.yaml --device cuda --dry-run
```

## 8. 已确定的训练实现

- Decoder-only Transformer 使用 Pre-LN、GELU、学习型位置嵌入和因果注意力；
- small、medium、large 的层数和隐藏维度写入对应 YAML；
- 上下文长度固定为 2,048；
- 优化器使用 AdamW，训练入口提供 warmup + cosine 学习率；
- 支持单卡和 `torchrun` DDP，具体设备和有效 batch 仍由正式实验确定。
