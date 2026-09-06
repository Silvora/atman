# Step 02: Tokenizer

本步骤读取第 1 步的清洗语料，训练一份 Atman 自己的 32K Byte-Level BPE tokenizer。small、medium、large 三档模型共用它，不需要训练三份 tokenizer。

## 输入与输出

输入清单：

```text
01_public_corpus/output/manifest.json
```

脚本不会猜测文件名，而是读取清单中的全部 `corpus_*.jsonl` 分片。输出固定写入本步骤目录：

```text
02_tokenizer/output/atman_tokenizer/
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── metadata.json
```

`metadata.json` 标记 `generated_by_step=02_tokenizer`，同时记录输入分片、实际训练文档数、词表大小和 `tokenizer.json` 的 SHA-256。第 3 步会用摘要检查 tokenizer 是否被换过。

## 为什么统一使用 32K

- 模型参数量由层数、隐藏维度等决定，tokenizer 不需要跟着模型档位复制三份。
- 32K 对中文、英文、数字和代码是较平衡的起点。
- 词表继续增大会增加每个模型的 embedding 和输出层参数。
- Byte-Level BPE 包含完整 byte alphabet，不认识的字符也能无损编码。

本项目把输入 embedding 与输出语言模型头共享权重，因此词表相关参数约等于
`vocab_size × hidden_size`：

| 模型档位 | 隐藏维度 | 32K 词表参数 | 占该模型总参数的比例 |
|---|---:|---:|---:|
| `small` | 512 | 16.4M | 约 46% |
| `medium` | 768 | 24.6M | 约 30% |
| `large` | 1024 | 32.8M | 约 24% |

如果改成 64K，上表的词表参数会翻倍，对 36M 的 small 尤其浪费；如果缩到
16K，中文和中英混合文本通常会被切成更多 token，同样的 1024 长度能容纳的内容
变少。32K 不是“对标 Qwen 就照搬”的数字，而是在本项目 36M、82M、134M 三档
模型之间做出的统一折中。之后只有在真实语料上统计出明显更好的压缩率时，才值得
重训 tokenizer；模型换档不需要重训。

BPE 的合并统计主要使用 CPU 和内存，换 CUDA 通常不会明显加速。大语料首次训练可能较久，但只训练一次，后续三个模型都可以复用。

## 正式运行

在项目根目录运行：

```bash
python 02_tokenizer/train_tokenizer.py
```

默认配置：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--vocab-size` | 32000 | 目标词表大小 |
| `--min-frequency` | 2 | 子词至少出现的次数 |
| `--model-max-length` | 1024 | tokenizer 声明的默认上下文长度 |
| `--max-documents` | 0 | 0 表示读取所有第 1 步文本 |

## 小规模验证

试跑必须使用独立输出目录，避免试验 tokenizer 被后续步骤误用：

```bash
python 02_tokenizer/train_tokenizer.py \
  --max-documents 10000 \
  --vocab-size 8000 \
  --output-dir 02_tokenizer/output_smoke/atman_tokenizer
```

正式输出存在 `metadata.json` 时脚本会拒绝覆盖。需要重训时，应先确认旧 tokenizer 已经不再被第 3、4 步使用，再手动清理对应输出目录。

## 完成检查

```bash
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('02_tokenizer/output/atman_tokenizer'); print(len(t), t.encode('人工智能'))"
```

确认词表能加载、中文能编码后，再进入第 3 步。
