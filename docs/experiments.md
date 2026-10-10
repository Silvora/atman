# v1-dense 实验记录

## 数据继承记录

数据从 `v1-base/data` 复制而来。复制内容包括原始语料、处理后的 JSONL、Tokenizer 和 2,048 长度编码文件。

| 内容 | 状态 |
|---|---|
| 原始语料与 manifest | 已复制 |
| processed train/validation | 已复制 |
| 16,384 词表 Tokenizer | 已复制 |
| 2,048 编码数据 | 已复制，用于基线检查 |
| 32K 编码数据 | 待生成 |
| 128K 编码数据 | 待生成 |

## 实验记录模板

每次实验记录以下字段：

```text
日期：
配置：small / medium / large
上下文长度：
训练 token 数：
GPU 和数量：
精度：
全局 batch：
学习率与 warmup：
训练速度：
峰值显存：
训练 loss：
验证 loss / perplexity：
checkpoint：
结论：
```

当前尚未启动 Dense 正式训练。
