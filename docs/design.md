# v1-moe 设计文档

## 1. 设计边界

本版本研究稀疏专家和高效注意力结构。数据来源、清洗和 Tokenizer 与 `v1-base` 对齐，MoE、MLA、上下文长度和训练统计独立维护。

本分支使用带控制 token 的 Tokenizer 契约，覆盖对话、Thinking、工具调用和扩展标记。复制来的旧编码数据在新词表发布后必须废弃并重新编码。

## 2. MLA 设计

MLA 将历史 Key/Value 压缩到潜在表示，在注意力计算和增量解码时减少需要保存的缓存。实现必须明确记录：

- latent dimension；
- Query/Key 的 RoPE 使用方式；
- 压缩 KV 是否与位置编码解耦；
- 训练时完整前向与推理时增量前向的差异；
- KV Cache 的元素数量和字节数。

第一阶段先实现可读、可验证的版本，再针对 CUDA kernel 和通信做优化。

## 3. MoE 设计

每个 Transformer block 的 FFN 由共享专家和路由专家组成：

```text
x → router → top-k experts → weighted sum
                  ↘ shared expert
```

每个 token 只进入 `top_k` 个路由专家，所有 token 经过共享专家。训练时需要记录：

- 总专家数 `num_experts`；
- 每个 token 激活的路由专家数 `top_k`；
- shared expert 数量；
- 专家容量因子和 token overflow；
- 路由负载均衡损失；
- 每个专家接收的 token 数和路由熵。

建议从少量专家开始：small 使用 8 个路由专家、top-2；medium 使用 16 个、top-2；large 再评估 32 个、top-4。实际配置要根据显存、通信和训练稳定性验收后冻结。

## 4. 长上下文

| 阶段 | sequence length | 用途 |
|---|---:|---|
| 结构验收 | 2,048 | 快速检查模型、路由和 MLA 缓存 |
| 第一阶段 | 32,768 | 正式长上下文训练 |
| 扩展阶段 | 131,072 | 长文档和缓存效率实验 |

`data/encoded` 中复制的文件按 2,048 保存，每条记录的布局不适合直接作为 32K/128K 样本。正式训练必须重新编码，并为每种长度生成独立报告。

## 5. 建议配置基线

| 配置 | d_model | 层数 | 总专家数 | top-k | 第一阶段上下文 |
|---|---:|---:|---:|---:|---:|
| small | 512 | 8 | 8 | 2 | 32K |
| medium | 768 | 12 | 16 | 2 | 32K |
| large | 1024 | 24 | 32 | 4 | 32K |

MLA latent dimension、共享专家 FFN 维度和路由专家 FFN 维度需要在实现前单独记录，因为它们决定实际参数量和激活计算量。

## 6. 验收项目

- top-k 路由的输出与手工计算一致；
- 专家容量、overflow 和负载均衡统计正确；
- 不同 batch/rank 下路由统计可聚合；
- MLA 完整前向与增量 KV Cache 输出一致；
- 2K、32K、128K 下缓存大小和吞吐可测量；
- checkpoint 能恢复 router、experts、MLA 和优化器状态；
- loss、路由损失和验证 perplexity 均无 NaN。

## 7. 训练与初始化

MoE/MLA checkpoint 与 Dense checkpoint 结构不同。v1-moe 默认从本分支初始化并进行预训练；如果实验要迁移 Dense 权重，必须记录只迁移哪些 embedding、attention 或 norm 参数，以及未匹配参数的初始化方式。
