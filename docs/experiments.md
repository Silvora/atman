# Atman v1 实验记录

## 2026-10-08：数据与 Tokenizer 管线冒烟验证

目的：验证 JSONL 读取、清洗、精确去重、稳定划分、Byte-level BPE 训练、评测和自动选择流程。

输入：

```text
source: pretrain_t2t_mini.jsonl
source_lines: 2,000
split_seed: 42
validation_ratio: 0.005
```

数据处理结果：

```text
train_documents: 1,991
validation_documents: 9
exact_duplicates: 0
invalid_json_records: 0
elapsed_seconds: 0.188
```

Tokenizer 冒烟参数：

```text
sample_documents: 1,000
candidate_vocab_sizes: 512, 1,024
evaluation_documents: 9
```

Tokenizer 结果：

| 词表 | 验证集 tokens | characters/token | 往返一致率 | 相对 1,024 增幅 |
|---:|---:|---:|---:|---:|
| 512 | 3,546 | 0.6393 | 100% | 34.01% |
| 1,024 | 2,646 | 0.8568 | 100% | 0% |

自动选择结果为 1,024，符合 15% 阈值规则。两个特殊 token 的 ID 分别为 0 和 1，均为单 token。

本次只验证管线正确性，不代表正式 Tokenizer 指标。正式流程使用 16,384 和 32,000 两个候选。

## 2026-10-08：正式语料过滤与 Tokenizer

输入和完整性校验：

```text
source: pretrain_t2t.jsonl
source_lines: 8,468,827
source_sha256: 31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d
split_seed: 42
validation_ratio: 0.005
```

过滤与划分结果：

```text
valid_json_records: 8,468,827
invalid_json_records: 0
empty_after_cleaning: 0
exact_duplicates: 3,749
train_documents: 8,423,079
validation_documents: 41,999
elapsed_seconds: 894.663
```

Tokenizer 正式候选使用训练集中的 1,000,000 条确定性样本，最多使用 50,000 条验证数据评测：

| 词表 | 验证集 tokens | characters/token | 往返一致率 | 相对 32K 增幅 |
|---:|---:|---:|---:|---:|
| 16,384 | 8,547,688 | 1.8881 | 100% | 10.06% |
| 32,000 | 7,766,602 | 2.0780 | 100% | 0% |

按照 15% 最大增幅规则，正式选择 16,384 词表。特殊 token 均为单 token，未知 token 数为 0。

本阶段已经完成文本过滤、数据划分和 Tokenizer 选择；当时尚未将全量文本编码成 token ID 定长训练块，也尚未开始模型训练。

## 2026-10-09：文本编码和定长打包冒烟测试

使用正式处理后的文本和 16,384 词表进行临时测试，每个分区读取 100 条文档，
上下文长度设置为 128。输出格式为小端 `uint16`，每个样本保存 129 个 token，
其中 128 个用于输入长度，额外 1 个用于构造下一个 token 的标签。

```text
train:      16,825 tokens → 130 sequences, 55 tail tokens discarded
validation: 15,079 tokens → 116 sequences, 115 tail tokens discarded
maximum_token_id: 16,383
EOS: 0
```

编码入口验证通过；随后已完成正式全量编码，结果记录在下一个编码验收条目中。

## 2026-10-09：正式 token ID 编码

使用正式 Tokenizer 和 `sequence_length=2048` 完成全量编码。输出为小端 `uint16`，每条记录保存 2,049 个 ID（2,048 个输入 token 加 1 个右移标签所需 token）。

```text
train:      837,900 sequences, 1,716,857,100 written tokens
validation: 4,192 sequences, 8,589,408 written tokens
maximum_token_id: 16,383
EOS: 0
partial_run: false
```

模型和训练入口已经实现，正式预训练尚未启动。
