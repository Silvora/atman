# Atman v1-dense：现代 Dense 长上下文模型

> 状态：项目脚手架和架构方案已建立；Dense 模型、长上下文编码、评测和正式训练尚未实现。

## 1. 版本定位

`v1-dense` 是 v1 阶段的现代 Dense 模型方案。它保留 Decoder-only 自回归语言模型作为主干，引入当前常见的效率和长上下文设计，用于研究结构升级对训练稳定性、显存、吞吐和生成质量的影响。

本版本与 `v1-base` 使用同一份原始语料、清洗结果和正式 Tokenizer 起步。复制来的 `data/encoded` 是上下文长度 2,048 的基线数据，正式长上下文训练前必须按本版本的序列长度重新编码。

本版本的 Tokenizer 额外预留对话、Thinking、工具调用和少量扩展 token；`v1-base` 保持最基础的 EOS/PAD 词表，不使用这些控制 token。

## 2. 设计目标

```text
同一份语料与词表
  → 现代 Dense Decoder-only Transformer
  → RoPE + RMSNorm + SwiGLU + GQA
  → 32K 基础上下文
  → 128K 长上下文扩展
  → 预训练、评测与成本记录
```

需要记录的结果包括：

- 与 `v1-base` 相同训练 token 预算下的 Loss 和 Perplexity；
- 不同上下文长度下的显存、吞吐和有效利用率；
- GQA 对 KV Cache 和推理速度的影响；
- RoPE 长度扩展对长文档任务的影响；
- Dense 模型在 small、medium、large 规模下的参数量和训练成本。

## 3. 架构方案

| 组件 | 设计基线 | 作用 |
|---|---|---|
| 主体 | Pre-LN Decoder-only Transformer | 保持自回归语言模型训练目标 |
| 位置编码 | RoPE | 支持相对位置信息和长上下文扩展 |
| 归一化 | RMSNorm | 减少归一化计算并改善深层训练稳定性 |
| 注意力 | GQA | 减少 KV Cache 和推理显存 |
| 前馈层 | SwiGLU | 提高参数利用率和非线性表达能力 |
| 注意力实现 | SDPA/FlashAttention | 降低长序列训练的内存访问开销 |
| 推理 | KV Cache | 避免逐 token 生成时重复计算历史 token |
| 激活方式 | Dense | 每个 token 经过所有 Transformer 层和 FFN |

上下文分两步处理：先以 32K 完成结构和训练链路验收，再根据显存和数据条件扩展到 128K。长度扩展需要重新编码或重新打包定长 token 数据，并单独记录 RoPE 参数。

## 4. 目录结构

```text
./
├── README.md
├── configs/                  # 数据、Tokenizer 和 Dense 模型配置
├── data/                     # 从 v1-base 复制的公共数据与 Tokenizer
├── src/                      # Dense 模型、数据和训练实现
├── scripts/                  # 下载、处理、编码、训练和评测入口（eval.py 当前为占位）
├── outputs/                  # checkpoint、日志和样例
├── evals/                    # 固定提示词与评测结果
└── docs/
    ├── design.md             # 架构和长上下文设计
    └── experiments.md        # 实验记录
```

## 5. 与 v1-base 的关系

- 数据来源、清洗规则、划分结果和 Tokenizer 保持一致，便于比较模型结构；
- 模型结构、编码序列长度、训练配置和评测结果独立保存；
- 不直接修改 `v1-base` 的模型代码和实验结果；
- `data/encoded` 的 2,048 长度文件仅用于基线复现，不能直接代表本版本的 32K/128K 训练数据。

## 6. 开发顺序

1. 实现 RoPE、RMSNorm、SwiGLU、GQA 和 KV Cache；
2. 用短序列完成张量形状、因果性和缓存一致性检查；
3. 生成 32K 定长 token 数据并完成 small 预训练冒烟；
4. 运行 medium 主实验，记录吞吐、显存和验证集指标；
5. 扩展 128K 长上下文并记录长度扩展的收益和代价；
6. 使用固定提示词与 `v1-base` 对比生成结果。

Tokenizer 配置改变后，必须先重新训练词表，再重新编码数据；复制来的旧 `data/tokenizer` 和 `data/encoded` 只代表基础词表与 2K 基线。

## 7. 当前状态

- 公共数据已从 `v1-base` 复制完成；
- 基础目录、脚本和配置已建立；
- 现代 Dense 架构设计已确定；
- Dense 模型代码、长上下文编码和正式训练待实现；
- 当前不把复制来的 2,048 数据标记为长上下文训练数据。

## 8. 平台运行

训练入口支持 Linux CUDA、Apple Silicon MPS 和 CPU。macOS 先使用 `--device mps --dry-run` 验证；多卡 `torchrun` 仅用于 CUDA 或 CPU Gloo。下载脚本需要 Bash，Windows 建议使用 WSL。
