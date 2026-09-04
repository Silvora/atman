# Step 01: Public Corpus

本步骤负责准备公开语料。它是整个模型训练流程的入口，目标不是直接训练模型，而是先得到一份干净、统一、可复用的文本语料库。

## 这一阶段做什么

输入：

- 公开文本数据集，例如中文维基百科、公开书籍、新闻、问答社区、开源文档等。

输出：

- 清洗后的 JSONL 文件，例如：

```text
output/01_public_corpus/public_corpus.jsonl
```

每一行是一条独立文本：

```json
{"text": "这里是一段清洗后的中文文本。", "source": "fjcanyue/wikipedia-zh-cn", "generated_by_step": "01_public_corpus", "generated_by_step_name": "Public Corpus"}
```

后面的分词器训练、语料编码、预训练都会基于这个文件继续进行。

## 为什么要单独做这一步

模型不是直接从网页、PDF、HTML 或混乱数据里学习，而是从 token 序列里学习。  
如果原始文本里有大量乱码、重复内容、广告、导航栏、格式符号，模型会把这些噪声也学进去。

所以第 1 步的核心价值是：

- 统一数据格式，方便后续处理。
- 去掉明显无效文本，降低训练噪声。
- 做简单去重，避免模型反复记住同一段内容。
- 保留来源字段，后续排查数据质量时能定位问题。
- 所有生成物都放进 `output/`，并用步骤编号标记来源，避免多个阶段的文件混在一起。

## 当前脚本

本目录提供：

```text
prepare_public_corpus.py
```

它会从 Hugging Face Datasets 上读取公开数据集，抽取文本字段，做基础清洗，然后保存成 `.jsonl`。

默认数据集是：

```text
fjcanyue/wikipedia-zh-cn
```

默认数据文件是：

```text
wikipedia-zh-cn-20260501.json
```

## 基础运行

在项目根目录运行：

```bash
python 01_public_corpus/prepare_public_corpus.py
```

默认输出：

```text
output/01_public_corpus/public_corpus.jsonl
```

## 常用参数

只取前 10000 篇文章：

```bash
python 01_public_corpus/prepare_public_corpus.py --max-samples 10000
```

修改输出路径：

```bash
python 01_public_corpus/prepare_public_corpus.py --output-path output/01_public_corpus/wiki_zh_clean.jsonl
```

指定文本字段名：

```bash
python 01_public_corpus/prepare_public_corpus.py --text-field text
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--dataset-name` | `fjcanyue/wikipedia-zh-cn` | 公开数据集名称 |
| `--data-file` | `wikipedia-zh-cn-20260501.json` | 数据集文件名 |
| `--split` | `train` | 使用的数据分片 |
| `--text-field` | `text` | 文本字段名 |
| `--output-path` | `output/01_public_corpus/public_corpus.jsonl` | 输出文件 |
| `--max-samples` | `10000` | 最多保留多少条文本，设为 `0` 表示不限 |
| `--min-chars` | `80` | 文本清洗后少于该字符数会被丢弃 |
| `--streaming` | 开启 | 流式读取，避免一次性加载大数据集 |

## 质量判断

生成后可以打开 `output/01_public_corpus/public_corpus.jsonl` 抽查几行，重点看：

- 是否主要是正文文本。
- 是否存在大量乱码。
- 是否存在 HTML 标签、菜单、广告。
- 是否大量重复。
- 文本长度是否适合训练。

如果这一步数据质量不好，后面即使训练 loss 降了，生成效果也会很差。

## 下一步

生成公开语料后，进入：

```text
02_tokenizer
```

如果复用现成 tokenizer，可以跳过专门训练 tokenizer，直接进入：

```text
03_encoding
```
