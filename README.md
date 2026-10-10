# 大语言模型技术演进：生成、对齐与推理

简体中文 | [English](README_en.md)

> 能力与边界说明
>
> 基于公开论文、技术报告和主流实现整理。时间范围用于描述技术演进，不代表严格的学术分期。

## 项目分支

本仓库使用独立分支保存三个模型项目，`main` 只维护技术演进说明文档：

| 分支 | 项目 | 当前重点 |
|---|---|---|
| [`v1/base`](https://github.com/Silvora/atman/tree/v1/base) | 基础 Decoder-only 模型 | 数据处理、Tokenizer、预训练、评测和生成闭环 |
| [`v1/dense`](https://github.com/Silvora/atman/tree/v1/dense) | 现代 Dense 模型 | RoPE、RMSNorm、SwiGLU、GQA 和长上下文 |
| [`v1/moe`](https://github.com/Silvora/atman/tree/v1/moe) | MoE/MLA 模型 | 专家路由、MLA、稀疏激活和长上下文 |

三个模型分支是独立项目，各自维护代码、配置、实验记录和训练产物。Dense 与 MoE 当前处于架构脚手架阶段，复制的 2,048 上下文数据只用于基线检查。

## 1. 总览

| 版本 | 时间范围 | 核心范式 | 代表性模型或系统 | 一句话概括 |
|---|---|---|---|---|
| **v1** | 2018–2022 | 规模化预训练与 ICL | GPT-2、GPT-3、PaLM、Chinchilla | 能生成连贯文本并通过上下文示例完成任务，但可控性和可靠性有限 |
| **v2** | 2022–2024 | 指令对齐与开源生态 | ChatGPT、Llama 2/3、Mistral、Qwen2 | 从文本生成模型发展为能够遵循指令的通用助手 |
| **v3** | 2024–至今 | 推理增强与 Agent 系统 | o1、DeepSeek-R1/V3/V4、Qwen3、Claude 4 | 通过额外推理计算和工具交互完成更复杂的任务 |

这里的 v1、v2、v3 是对技术阶段的归纳标签。代表性模型可能跨越阶段，单个模型也可能同时包含多个阶段的技术特征。

## 2. 能力与边界

### 2.1 v1：规模化预训练与 ICL

#### 主要能力

- 通过大规模自回归预训练获得通用文本生成能力；
- 在 zero-shot 和 few-shot 提示下完成部分翻译、问答、摘要、分类和代码任务；
- 通过上下文示例表现出早期 ICL 能力；
- 生成连贯长文本，并在部分任务中表现出知识记忆和浅层模式推理；
- 模型规模、数据规模和训练计算量成为能力提升的重要变量。

GPT-3 的公开报告展示了无需参数更新、仅通过文本任务描述和少量示例完成任务的能力。[OpenAI GPT-3 介绍](https://openai.com/index/language-models-are-few-shot-learners/)

#### 主要边界

- 指令遵循能力不稳定，模型容易把任务描述当作普通文本继续生成；
- 没有经过系统化的指令对齐、偏好优化或专门推理训练；
- 没有可靠的原生工具调用和长时程任务执行能力；
- 上下文窗口通常较短，但不同模型存在差异；
- 事实性、格式控制、安全性和输出稳定性有限；
- 开源模型和配套工具生态处于形成阶段，尚未达到后续阶段的成熟度。

#### 常见失败模式

跑题、重复生成、自信幻觉、格式不稳定、对有害请求缺乏可靠约束，以及复杂数学和逻辑任务中的错误。

#### 阶段概括

v1 的主要突破是通过规模化预训练获得通用语言能力，并出现 ICL；主要限制是模型知道如何生成文本，却还不能稳定理解并执行用户意图。

### 2.2 v2：指令对齐与开源生态

#### 主要能力

- 通过 SFT、RLHF、DPO 等方法提升指令遵循和对话质量；
- 能够根据系统消息、用户指令和对话历史调整输出行为；
- 安全性、有用性、格式控制和拒答行为得到改善；
- RoPE、RMSNorm、SwiGLU、GQA 等结构逐渐成为常见设计；
- 上下文长度在部分模型中扩展到 32K、128K 或更长；
- 开源权重、训练框架、推理框架和微调工具形成较完整的生态；
- 出现基础 Function Calling 和单轮工具调用能力。

InstructGPT 的研究表明，监督微调结合人类反馈可以显著改善模型对用户意图的遵循，同时降低部分不真实和有害输出。[InstructGPT 论文](https://arxiv.org/abs/2203.02155)

#### 主要边界

- 深度推理能力仍然脆弱，复杂数学、代码和逻辑任务容易出错；
- 多步规划和长时程任务的目标保持能力有限；
- 工具调用通常依赖外部编排，错误恢复能力较弱；
- 知识截止、事实幻觉和领域可靠性问题仍然存在；
- 长上下文不等于稳定使用全部上下文，检索和注意力分配仍可能失效；
- 多模态和原生 Agent 能力仍处于快速发展阶段。

#### 常见失败模式

复杂推理错误、事实幻觉、工具参数错误、对话目标漂移、过度拒答，以及长任务中无法持续维护状态。

#### 阶段概括

v2 的核心变化是把“能够生成文本”的模型训练成“能够按照指令交互”的助手，但推理深度、长时程可靠性和自主行动能力仍有限。

### 2.3 v3：推理增强与 Agent 系统

#### 主要能力

- 通过推理数据、冷启动训练和推理强化学习提升复杂问题求解能力；
- 使用测试时额外计算、候选生成、验证和自我修正处理困难任务；
- 支持 Thinking / Non-Thinking 或类似的推理模式控制；
- 通过多轮工具调用完成搜索、代码执行、文件操作和环境交互；
- 具备任务分解、计划执行、结果检查和错误恢复能力；
- 部分前沿模型提供百万级上下文；
- MoE、稀疏激活和高效推理结构用于降低大规模系统的成本；
- 推理模型与 Agent 框架逐渐结合。

OpenAI 对 o1 的公开说明将强化学习、额外思考时间和复杂推理任务作为核心特征。[OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/)

Qwen3 的公开技术说明展示了长链思维冷启动、推理强化学习、Thinking 模式融合和通用强化学习的组合流程。[Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)

Claude 4 已公开展示带工具使用的 extended thinking，说明推理过程和工具交互开始形成统一系统。[Claude 4 官方说明](https://www.anthropic.com/news/claude-4)

#### 主要边界

- 长时程任务仍可能发生目标漂移、错误累积和计划失效；
- 额外推理计算会带来更高延迟、显存和调用成本；
- 工具可以降低部分事实错误，但工具选择和结果解释仍可能失败；
- 开放环境中的可靠性、权限控制和安全边界尚未完全解决；
- 持续学习、自主演化和稳定世界模型仍不是普遍能力；
- 百万级上下文只在部分系统中可用，也不保证所有内容都能被有效利用。

DeepSeek-V4 的公开资料将百万级上下文和 Agent 能力列为重要方向，但这代表部分前沿系统的能力，不等于所有模型的统一标准。[DeepSeek-V4 官方说明](https://deepseek.com/en/news/v4-preview/)

#### 常见失败模式

长任务中途偏离目标、推理成本过高、工具调用链中断、开放环境下错误恢复失败，以及无工具时仍出现事实性错误。

#### 阶段概括

v3 将模型从“回答问题”推进到“为完成任务进行推理和行动”。其优势主要体现在结构化、可验证的任务中；在开放、长期和高风险环境中仍需要约束、监控和人工监督。

## 3. 横向比较

| 维度 | v1 | v2 | v3 |
|---|---|---|---|
| 主要目标 | 学习语言分布和通用生成 | 遵循指令并进行对话 | 推理、规划并完成多步任务 |
| 指令遵循 | 弱且不稳定 | 明显增强 | 更强，并可结合模式控制 |
| 专门推理训练 | 基本没有 | 少量或不系统 | 推理数据、RL 和验证机制成为重要组成部分 |
| ICL | 早期形成 | 更稳定 | 与推理和工具使用结合 |
| 工具调用 | 基本没有 | 基础 Function Calling | 多轮工具调用和 Agent 编排 |
| 上下文 | 通常较短，模型差异较大 | 部分模型达到 32K–128K 或更长 | 部分前沿模型达到百万级 |
| 对齐与安全 | 有限 | SFT、RLHF、DPO 等成为主流方法 | 对齐与推理、工具和环境约束结合 |
| 典型失败 | 跑题、重复、幻觉 | 浅层推理、格式错误、工具不稳 | 长时程漂移、成本高、开放环境不可靠 |

## 4. Thinking 的阶段变化

Thinking 不应简单理解为模型是否输出 `<think>` 标签，而应区分提示引导、专门训练和推理时计算：

- **v1**：可以通过 few-shot 或 Chain-of-Thought 提示诱导有限的中间推理，但没有系统化推理训练；
- **v2**：部分对齐模型可以生成较长的 CoT，但推理质量、可控性和验证能力仍不稳定；
- **v3**：推理数据、推理强化学习、测试时计算、验证器和可切换 Thinking 模式形成较完整的技术组合。

因此，Thinking 更适合作为 v3 的核心开发方向；v1 和 v2 可以研究提示引导与浅层推理，但不能把它们等同于系统化推理模型。

## 5. 关键技术演进

| 技术点 | v1 | v2 | v3 |
|---|---|---|---|
| 训练范式 | 自回归预训练 | 预训练 → SFT → RLHF/DPO | 预训练/对齐 → 推理训练 → 验证奖励或 Agent 训练 |
| 注意力机制 | 标准密集注意力为主 | GQA 等效率结构逐渐普及 | 稀疏注意力、MLA 等在部分系统中出现 |
| 位置编码 | 学习型绝对位置等早期方案 | RoPE 成为常见方案 | RoPE 外推、长上下文扩展和其他变体并存 |
| 模型结构 | Dense Transformer 为主 | Dense 为主，早期 MoE 开始实用 | 大规模 MoE 和稀疏激活在部分系统中普及 |
| 对齐方法 | 基本没有系统对齐 | SFT、RLHF、DPO | 过程监督、可验证奖励和推理 RL |
| 推理方式 | 单次前向生成 | 单次生成与有限 CoT | Test-time Compute、验证、自我修正和 Hybrid Thinking |
| 工具使用 | 基本没有 | 单轮或简单调用 | 多轮调用、规划、执行和错误恢复 |

## 6. 参考资料

### v1

- [Language Models are Few-Shot Learners（GPT-3）](https://openai.com/index/language-models-are-few-shot-learners/)
- [An Empirical Analysis of Compute-Optimal Large Language Model Training（Chinchilla）](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)
- GPT-2、Scaling Laws、PaLM 等公开论文和技术报告

### v2

- [Training Language Models to Follow Instructions with Human Feedback（InstructGPT）](https://arxiv.org/abs/2203.02155)
- Llama 2、Llama 3、Mistral、Qwen2 等模型报告
- DPO、SFT、RLHF 相关论文和开源训练框架

### v3

- [Introducing OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/)
- [DeepSeek-R1 Release](https://api-docs.deepseek.com/news/news250120/)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Introducing Claude 4](https://www.anthropic.com/news/claude-4)
- [DeepSeek-V4 Preview](https://deepseek.com/en/news/v4-preview/)

## 7. 备注

- v1、v2、v3 是分析技术演进的归纳标签，不是厂商统一定义的行业标准；
- 同一模型可能同时包含多个阶段的技术，代表模型不等于阶段边界；
- 上下文长度、MoE、MLA、工具调用和 Thinking 都存在模型间差异，不能仅凭单个代表模型概括整个阶段；
- Thinking 不等于结果正确，验证、工具和外部约束仍然必要；
- 时代划分会随着新模型和新训练方法出现而变化，后续技术阶段需要根据公开证据重新定义。
