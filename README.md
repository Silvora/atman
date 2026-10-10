# Atman v1-moe：MoE、MLA 与长上下文模型

## 1. 版本定位

`v1-moe` 是 v1 阶段的稀疏专家模型方案。它在 Decoder-only 自回归语言模型基础上研究 Mixture-of-Experts、Multi-head Latent Attention 和长上下文，目标是比较总参数量、激活参数量、训练成本、KV Cache 和模型能力之间的关系。

本版本使用从 `v1-base` 复制的同一份数据和 Tokenizer 作为起点。复制来的编码数据上下文长度为 2,048，只用于数据链路和基线检查；MLA 与长上下文正式训练需要按本版本的序列长度重新编码和验证。

本版本的 Tokenizer 额外预留对话、Thinking、工具调用和少量扩展 token；`v1-base` 保持最基础的 EOS/PAD 词表，不使用这些控制 token。

## 2. 设计目标

```text
同一份语料与词表
  → MLA 注意力
  → 共享专家 + 路由专家的 MoE FFN
  → 32K 基础上下文
  → 128K 长上下文扩展
  → 记录激活参数、路由负载、吞吐和验证指标
```

重点观察：

- 激活参数量与推理速度的关系；
- expert routing 的负载均衡和 token 丢弃情况；
- MLA 对 KV Cache 大小和长上下文吞吐的影响；
- MoE 与 Dense 模型在相同激活计算量下的 Loss 和 Perplexity；
- 长上下文训练带来的通信、显存和稳定性成本。

## 3. 架构方案

| 组件 | 设计基线 | 作用 |
|---|---|---|
| 主体 | Decoder-only Transformer | 保留自回归建模和统一训练目标 |
| 注意力 | MLA | 用潜在压缩表示降低 KV Cache 成本 |
| 前馈层 | DeepSeek 风格 MoE | 每个 token 只激活少量专家 |
| 专家结构 | Shared expert + routed experts | 保留通用能力并增加专家分工 |
| 路由 | Top-k routing | 控制每个 token 的激活专家数量 |
| 负载控制 | routing balance loss/统计 | 避免少数专家过载或失活 |
| 位置编码 | RoPE 及长上下文扩展 | 支持 32K 到 128K 实验 |
| 推理 | 压缩 KV Cache | 降低长序列生成的缓存占用 |

MoE 和 MLA 都会改变张量形状、通信方式和 checkpoint 格式，因此本分支必须独立实现和验证，不能把 Dense 分支的权重文件直接当作完整初始化结果。

## 4. 目录结构

```text
./
├── README.md
├── configs/                  # MoE、MLA、数据和 Tokenizer 配置
├── data/                     # 从 v1-base 复制的数据与 Tokenizer
├── src/                      # MLA、MoE、路由和训练实现
├── scripts/                  # 数据、训练、评测和检查入口
├── outputs/                  # checkpoint、路由统计和训练日志
├── evals/                    # 质量、长上下文和效率评测
└── docs/
    ├── design.md
    └── experiments.md
```

## 5. 与其他 v1 分支的关系

- `v1-base` 提供早期 Dense 基线和公共数据版本；
- `v1-dense` 提供现代 Dense 长上下文对照组；
- `v1-moe` 在独立架构中研究 MoE、MLA 和长上下文；
- 三个分支共享数据来源记录，但模型代码、配置、checkpoint 和实验结论独立保存。

## 6. 开发顺序

1. 先实现 MLA 的压缩 KV 表示和增量解码；
2. 实现单机小规模 MoE、专家路由和负载统计；
3. 在短序列上验证 dense attention 与 MLA 的输出差异和缓存一致性；
4. 生成 32K token 数据，完成 small MoE 结构验收；
5. 再扩展 medium/large、专家数量和 128K 上下文；
6. 记录激活参数、路由熵、专家负载、吞吐、显存和验证指标。

Tokenizer 配置改变后，必须先重新训练词表，再重新编码数据；复制来的旧 `data/tokenizer` 和 `data/encoded` 只代表基础词表与 2K 基线。

## 7. 当前状态

- 公共数据已从 `v1-base` 复制完成；
- 基础目录、脚本和配置已建立；
- MoE + MLA + 长上下文设计已确定；
- MLA、MoE 路由、长上下文编码和正式训练待实现；
- 当前复制来的 2,048 编码数据不作为正式长上下文训练数据。

## 8. 平台运行

训练入口支持 Linux CUDA、Apple Silicon MPS 和 CPU。macOS 先使用 `--device mps --dry-run` 验证；多卡 `torchrun` 仅用于 CUDA 或 CPU Gloo。下载脚本需要 Bash，Windows 建议使用 WSL。
