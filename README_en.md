# LLM Evolution: Generation, Alignment, and Reasoning

[简体中文](README.md) | English

> A description of the capabilities and boundaries associated with v1, v2, and v3.
>
> Compiled from public papers, technical reports, and major implementations. The time ranges describe broad technical development and are not strict academic periods.

## 1. Overview

| Version | Time range | Core paradigm | Representative models or systems | Summary |
|---|---|---|---|---|
| **v1** | 2018–2022 | Scaled pre-training and ICL | GPT-2, GPT-3, PaLM, Chinchilla | Generates coherent text and performs some tasks from context examples, but remains difficult to control |
| **v2** | 2022–2024 | Instruction alignment and open-model ecosystem | ChatGPT, Llama 2/3, Mistral, Qwen2 | Develops from a text generator into a general assistant that follows instructions |
| **v3** | 2024–present | Enhanced reasoning and agent systems | o1, DeepSeek-R1/V3/V4, Qwen3, Claude 4 | Uses additional reasoning compute and tool interaction for more complex tasks |

The labels v1, v2, and v3 summarize technical stages. Representative models may span multiple stages, and a single model may combine features from more than one stage.

## 2. Capabilities and Boundaries

### 2.1 v1: Scaled Pre-training and ICL

#### Capabilities

- General text generation from large-scale autoregressive pre-training;
- Zero-shot and few-shot performance on some translation, question-answering, summarization, classification, and coding tasks;
- Early in-context learning from examples placed in the prompt;
- Coherent long-form text and limited knowledge recall or shallow pattern reasoning;
- Clear effects from model scale, data scale, and training compute.

GPT-3 demonstrated task completion from textual task descriptions and a few examples without parameter updates.[OpenAI GPT-3](https://openai.com/index/language-models-are-few-shot-learners/)

#### Boundaries

- Unstable instruction following;
- No systematic instruction alignment, preference optimization, or dedicated reasoning training;
- No reliable native tool use or long-horizon task execution;
- Usually shorter context windows, with substantial model-to-model variation;
- Limited factuality, format control, safety, and output stability;
- An open-model and tooling ecosystem that was still forming.

#### Common failures

Topic drift, repetition, confident hallucinations, unstable formatting, weak control over harmful requests, and errors on complex mathematical or logical tasks.

#### Summary

The main achievement of v1 was general language ability through scale and pre-training, including early ICL. Its main limitation was that it could generate text without reliably understanding and executing user intent.

### 2.2 v2: Instruction Alignment and Open Models

#### Capabilities

- Improved instruction following and dialogue through SFT, RLHF, DPO, and related methods;
- Behavior conditioned on system messages, user instructions, and conversation history;
- Better helpfulness, safety, format control, and refusal behavior;
- RoPE, RMSNorm, SwiGLU, and GQA becoming common architectural choices;
- Context windows extended to 32K, 128K, or longer in some models;
- A mature ecosystem of open weights, training libraries, inference systems, and fine-tuning tools;
- Early function calling and single-turn tool use.

InstructGPT showed that supervised fine-tuning with human feedback can improve instruction following while reducing some untruthful and harmful outputs.[InstructGPT paper](https://arxiv.org/abs/2203.02155)

#### Boundaries

- Fragile deep reasoning on difficult mathematics, coding, and logic;
- Limited goal retention in multi-step and long-horizon tasks;
- Tool use often dependent on external orchestration, with weak recovery from errors;
- Persistent knowledge-cutoff, factuality, and domain-reliability problems;
- Long context does not guarantee effective use of every part of the context;
- Multimodal and native agent capabilities still developing.

#### Common failures

Reasoning errors, factual hallucinations, incorrect tool arguments, goal drift, over-refusal, and loss of state during long tasks.

#### Summary

The central change in v2 was turning a text generator into an instruction-following assistant. Deep reasoning, long-horizon reliability, and autonomous action remained limited.

### 2.3 v3: Enhanced Reasoning and Agent Systems

#### Capabilities

- Better complex problem solving through reasoning data, cold-start training, and reasoning reinforcement learning;
- Additional test-time compute, candidate generation, verification, and self-correction;
- Thinking / Non-Thinking or similar mode controls;
- Multi-turn tool use for search, code execution, file operations, and environment interaction;
- Task decomposition, planning, result checking, and error recovery;
- Million-token context in some frontier systems;
- MoE, sparse activation, and efficient inference structures for reducing large-system cost;
- Increasing integration between reasoning models and agent frameworks.

OpenAI describes o1 through reinforcement learning, additional thinking time, and complex reasoning tasks.[OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/)

Qwen3 describes a combination of long-CoT cold start, reasoning reinforcement learning, thinking-mode fusion, and general reinforcement learning.[Qwen3](https://qwenlm.github.io/blog/qwen3/)

Claude 4 publicly describes extended thinking with tool use, showing the integration of reasoning and interaction.[Claude 4](https://www.anthropic.com/news/claude-4)

#### Boundaries

- Goal drift, accumulated errors, and plan failure in long-horizon tasks;
- Higher latency, memory use, and cost from additional reasoning compute;
- Tool use reduces some factual errors but tool selection and result interpretation can still fail;
- Reliability, permissions, and safety in open environments remain unresolved;
- Continuous learning, self-evolution, and stable world models are not general capabilities;
- Million-token context is available only in some systems and does not guarantee effective use of all content.

DeepSeek-V4 presents million-token context and agent capabilities as major directions, but those capabilities describe selected frontier systems rather than a universal standard.[DeepSeek-V4](https://deepseek.com/en/news/v4-preview/)

#### Common failures

Goal drift during long tasks, high reasoning cost, broken tool chains, failed recovery in open environments, and factual errors without external tools.

#### Summary

v3 moves from answering questions toward reasoning and acting to complete tasks. It is strongest on structured and verifiable tasks and still needs constraints, monitoring, and human oversight in open, long-running, high-risk environments.

## 3. Cross-Stage Comparison

| Dimension | v1 | v2 | v3 |
|---|---|---|---|
| Main objective | Learn language distributions and generate text | Follow instructions and hold conversations | Reason, plan, and complete multi-step tasks |
| Instruction following | Weak and unstable | Significantly improved | Stronger, with mode control in some systems |
| Dedicated reasoning training | Generally absent | Limited or inconsistent | Reasoning data, RL, and verification become important |
| ICL | Emerges early | More stable | Combined with reasoning and tool use |
| Tool use | Generally absent | Basic function calling | Multi-turn tools and agent orchestration |
| Context | Usually shorter, with model variation | 32K–128K or longer in some models | Million-token context in some frontier systems |
| Alignment and safety | Limited | SFT, RLHF, and DPO become common | Combined with reasoning, tools, and environment constraints |
| Typical failures | Drift, repetition, hallucination | Shallow reasoning, formatting errors, unstable tools | Long-horizon drift, high cost, unreliable open environments |

## 4. How Thinking Changes Across Stages

Thinking should be separated into prompt-induced reasoning, dedicated training, and test-time compute. It is not defined simply by whether a model emits a `<think>` tag:

- **v1:** Few-shot or Chain-of-Thought prompts can induce limited intermediate reasoning, without systematic reasoning training;
- **v2:** Some aligned models generate longer CoT, but quality, control, and verification remain unstable;
- **v3:** Reasoning data, reasoning RL, test-time compute, verifiers, and switchable Thinking modes form a more complete technical combination.

Thinking is therefore a central development direction for v3. v1 and v2 can study prompt-induced and shallow reasoning without being equivalent to systematic reasoning models.

## 5. Key Technical Changes

| Topic | v1 | v2 | v3 |
|---|---|---|---|
| Training | Autoregressive pre-training | Pre-training → SFT → RLHF/DPO | Pre-training/alignment → reasoning training → verifiable-reward or agent training |
| Attention | Standard dense attention | GQA and other efficiency structures become common | Sparse attention, MLA, and other variants in some systems |
| Position encoding | Learned absolute positions and other early methods | RoPE becomes common | RoPE extrapolation, long-context extensions, and other variants |
| Model structure | Mostly dense Transformers | Mostly dense, with early practical MoE | Large-scale MoE and sparse activation in some systems |
| Alignment | Little systematic alignment | SFT, RLHF, and DPO | Process supervision, verifiable rewards, and reasoning RL |
| Inference | Single forward generation | Single generation and limited CoT | Test-time compute, verification, self-correction, and Hybrid Thinking |
| Tool use | Generally absent | Single-turn or simple calls | Multi-turn calls, planning, execution, and recovery |

## 6. References

### v1

- [Language Models are Few-Shot Learners](https://openai.com/index/language-models-are-few-shot-learners/)
- [An Empirical Analysis of Compute-Optimal Large Language Model Training](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)
- Public papers and reports on GPT-2, Scaling Laws, and PaLM

### v2

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Model reports for Llama 2, Llama 3, Mistral, and Qwen2
- Research and open frameworks for SFT, DPO, and RLHF

### v3

- [Introducing OpenAI o1](https://openai.com/index/introducing-openai-o1-preview/)
- [DeepSeek-R1 Release](https://api-docs.deepseek.com/news/news250120/)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Introducing Claude 4](https://www.anthropic.com/news/claude-4)
- [DeepSeek-V4 Preview](https://deepseek.com/en/news/v4-preview/)

## 7. Notes

- v1, v2, and v3 are analytical labels for technical development, not an industry-wide standard;
- The same model may combine features from multiple stages, and representative models do not define strict boundaries;
- Context length, MoE, MLA, tool use, and Thinking vary by model and cannot be generalized from one example;
- Thinking does not guarantee correctness; verification, tools, and external constraints remain necessary;
- The stage boundaries may change as new models and training methods emerge.
