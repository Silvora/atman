# Step 03: Encoding

本步骤负责把清洗后的文本语料转换成模型预训练可以直接读取的 token 数据。

前两步的产物是：

```text
output/01_public_corpus/public_corpus.jsonl
output/02_tokenizer/atman_tokenizer/
```

第 3 步会读取这两个输入，然后输出到：

```text
output/03_encoding/
```

## 为什么需要编码

模型训练时不能直接读取中文字符串，它需要整数 token id。  
第 2 步 tokenizer 负责定义“文本如何变成 token”，第 3 步则把整份语料提前编码好。

这样做有两个好处：

- 训练时不用重复分词，速度更稳定。
- 数据格式固定，后面 small / medium / large 三个模型可以共用同一份预训练数据。

## 输入

清洗后的公开语料：

```text
output/01_public_corpus/public_corpus.jsonl
```

训练好的 tokenizer：

```text
output/02_tokenizer/atman_tokenizer/
```

## 输出

默认输出目录：

```text
output/03_encoding/
```

主要文件：

```text
pretrain_data_00000.pt
pretrain_data_00001.pt
manifest.json
```

每个 `.pt` 文件是一个数据分片，里面包含：

```text
input_ids: [num_chunks, seq_length]
metadata: 当前分片的来源和统计信息
```

`manifest.json` 会记录所有分片、总样本数、总 token 数、使用的 tokenizer 和生成步骤：

```json
{
  "generated_by_step": "03_encoding",
  "generated_by_step_name": "Encoding"
}
```

## 为什么使用分片

如果一次性把全部 token 块放进一个大 `.pt` 文件，数据变大后容易占用大量内存。  
分片保存更适合真实训练流程：

- 单个文件更小。
- 中断后更容易排查。
- 后续训练可以按 shard 加载。
- 适合从 2GB 语料扩展到更大语料。

## 运行方式

在项目根目录运行：

```bash
python 03_encoding/encode_pretrain_data.py
```

默认会生成长度为 `1024` 的训练样本：

```text
output/03_encoding/pretrain_data_00000.pt
```

如果想更快训练，可以改成 `512`：

```bash
python 03_encoding/encode_pretrain_data.py --seq-length 512
```

如果只想快速测试流程：

```bash
python 03_encoding/encode_pretrain_data.py --max-samples 1000 --seq-length 512
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-path` | `output/01_public_corpus/public_corpus.jsonl` | 第 1 步生成的清洗语料 |
| `--tokenizer-dir` | `output/02_tokenizer/atman_tokenizer` | 第 2 步生成的 tokenizer |
| `--output-dir` | `output/03_encoding` | 第 3 步输出目录 |
| `--seq-length` | `1024` | 每条训练样本的 token 长度 |
| `--shard-size` | `50000` | 每个分片保存多少条样本 |
| `--max-samples` | `0` | 最多读取多少条文本，`0` 表示不限 |
| `--text-field` | `text` | JSONL 中的文本字段名 |

## 和模型大小的关系

tokenizer 和编码数据建议统一一份，small / medium / large 三个模型共用。  
模型大小主要在第 4 步通过 hidden size、层数、注意力头数来区分。

不过 `seq_length` 会影响训练速度和显存：

| seq_length | 适合场景 |
|---:|---|
| `512` | 快速教程、低显存设备 |
| `1024` | 推荐默认，质量和速度平衡 |
| `2048` | large 模型或更认真训练 |

注意力计算成本大约随序列长度平方增长，所以 `2048` 会明显比 `1024` 慢。

## 下一步

完成编码后，进入：

```text
04_pretraining
```

第 4 步会读取 `output/03_encoding/manifest.json` 中列出的数据分片，开始预训练。
