# Step 02: Tokenizer

本步骤负责训练或准备分词器。分词器的作用是把自然语言文本转换成模型能理解的整数 token id。

第 1 步已经生成了清洗后的公开语料：

```text
output/01_public_corpus/public_corpus.jsonl
```

第 2 步会基于这个文件训练 tokenizer，并把结果保存到：

```text
output/02_tokenizer/atman_tokenizer/
```

## 为什么要有分词器

神经网络不能直接读字符串，它只能处理数字。  
分词器负责完成这件事：

```text
中文文本
  ↓
token
  ↓
token id
  ↓
模型输入
```

例如一句话：

```text
人工智能的未来是
```

会被编码成类似：

```text
[1234, 5678, 90, ...]
```

模型真正学习的是这些数字序列。

## 为什么这里训练自己的 tokenizer

你之前使用的是 GPT-2 tokenizer。它能处理中文，但它主要不是为中文训练的，所以中文文本经常会被切得很碎。  
切得太碎会带来两个问题：

- 同样一句中文会变成更多 token，训练更慢。
- 模型更难学到稳定的中文词语和表达结构。

所以教程里建议把“自训练 tokenizer”作为第 2 步。这样模型、数据、分词器都属于同一套训练流程。

## 当前方案

本教程使用 Byte-Level BPE tokenizer。

选择它的原因：

- 能处理中文、英文、数字、标点和特殊符号。
- 不容易因为生僻字符产生大量 `<unk>`。
- 和 GPT 类自回归语言模型比较搭。
- 能保存成 Hugging Face Transformers 可直接加载的格式。

## 输入

默认输入：

```text
output/01_public_corpus/public_corpus.jsonl
```

每一行格式：

```json
{"text": "一段清洗后的文本", "generated_by_step": "01_public_corpus"}
```

脚本会读取其中的 `text` 字段。

## 输出

默认输出目录：

```text
output/02_tokenizer/atman_tokenizer/
```

主要文件包括：

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
metadata.json
```

其中 `metadata.json` 会记录这个 tokenizer 是由第 2 步生成的：

```json
{
  "generated_by_step": "02_tokenizer",
  "generated_by_step_name": "Tokenizer"
}
```

## 运行方式

在项目根目录运行：

```bash
python 02_tokenizer/train_tokenizer.py
```

训练更小的 tokenizer：

```bash
python 02_tokenizer/train_tokenizer.py --vocab-size 16000
```

限制最多读取多少条文本：

```bash
python 02_tokenizer/train_tokenizer.py --max-samples 5000
```

指定输入输出：

```bash
python 02_tokenizer/train_tokenizer.py \
  --input-path output/01_public_corpus/public_corpus.jsonl \
  --output-dir output/02_tokenizer/atman_tokenizer
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-path` | `output/01_public_corpus/public_corpus.jsonl` | 第 1 步生成的清洗语料 |
| `--output-dir` | `output/02_tokenizer/atman_tokenizer` | tokenizer 输出目录 |
| `--vocab-size` | `32000` | 词表大小 |
| `--min-frequency` | `2` | token 至少出现多少次才进入词表 |
| `--max-samples` | `0` | 最多读取多少条文本，`0` 表示不限 |
| `--text-field` | `text` | JSONL 里的文本字段名 |

## 如何检查效果

训练完成后，脚本会打印一个测试句子的编码和解码结果。重点看：

- 解码后是否能还原原句。
- 中文是否被切得过碎。
- tokenizer 文件是否保存到 `output/02_tokenizer/`。

## 下一步

完成 tokenizer 后，进入：

```text
03_encoding
```

第 3 步会使用这个 tokenizer，把清洗语料编码成固定长度的训练数据。
