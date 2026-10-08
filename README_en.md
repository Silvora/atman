# Atman LLM Evolution Lab

[简体中文](README.md)

> A multi-version engineering project that builds language-model systems across major eras of AI development.

## 1. Project Scope

This project is not a sequence of upgrades to one model, and v1, v2, and v3 are not intended to compete on one common leaderboard.

Each version is an independent product of a different development era, with its own architecture, data pipeline, training method, capability goals, and evaluation criteria. After a version is completed, its historical characteristics, experiment results, and limitations are preserved before work moves to the next stage.

If a new and sufficiently stable AI paradigm emerges, the project may continue with v4, v5, and later versions without retrofitting completed historical versions.

The v1/v2/v3 labels are internal architecture-version names and are not tied to any vendor's model naming.

## 2. Version Roadmap

| Project version | Development stage | Main version objective | Status |
|---|---|---|---|
| **v1** | Foundation language models and scaled pre-training | Build the full Tokenizer, Decoder-only Transformer, pre-training, generation, and basic evaluation pipeline from scratch | Structure and design |
| **v2** | Instruction alignment and modern open models | Implement modern architectures, SFT, multi-turn chat, preference alignment, and basic tool use | Planned |
| **v3** | Reasoning and agents | Implement Thinking, Reasoning RL, inference budgets, tool use, and agent training | Planned |
| **v4+** | Future stages | Defined only after a new technical paradigm becomes stable | Reserved |

## 3. Relationship Between Versions

```text
v1: complete the foundation-language-model loop
 ↓ finish, document, and preserve
v2: independently design an instruction-aligned conversational system
 ↓ finish, document, and preserve
v3: independently design a reasoning and agent system
 ↓
v4: define only when the next stable paradigm emerges
```

The versions follow these rules:

1. **Stage independence:** each version can be understood, trained, evaluated, and run on its own.
2. **Historical fidelity:** later techniques are not forced into earlier versions.
3. **No mandatory backward compatibility:** later versions may replace the Tokenizer, architecture, data, objectives, and interfaces.
4. **No unequal ranking:** versions with different goals are not reduced to one overall score.
5. **Within-version experiments:** small, medium, and large are compared only inside the same version and under documented controls.
6. **Preservation after completion:** configurations, data provenance, results, failures, and capability boundaries are recorded.

## 4. Version Overview

### 4.1 v1: Foundation Language Model

v1 studies the complete path from raw text to a generative language model:

- corpus download, cleaning, and segmentation;
- Tokenizer training and validation;
- Decoder-only Transformer;
- Causal Language Modeling;
- pre-training and checkpoint recovery;
- Loss, Perplexity, and text-generation evaluation;
- basic zero-shot, few-shot, and ICL observations;
- small, medium, and large experiments.

Stable instruction following, Thinking, RLHF, DPO, and agents are outside v1's goals.

See [`v1/README.md`](v1/README.md) for the detailed v1 plan.

### 4.2 v2: Instruction Alignment and Modern Open Models

The planned topics include:

- RoPE, RMSNorm, SwiGLU, GQA, and related modern architecture choices;
- Chat Templates and multi-turn conversation;
- Supervised Fine-Tuning;
- preference data and DPO/RLHF;
- instruction following, format control, and alignment;
- basic function or tool calling.

v2 will be designed independently and does not have to inherit v1 weights, Tokenizer files, or implementation.

### 4.3 v3: Reasoning and Agents

The planned topics include:

- Thinking and Non-Thinking modes;
- long-CoT cold-start SFT;
- verifiable rewards and Reasoning RL;
- reasoning budgets and test-time compute;
- multi-turn tool use, planning, and recovery;
- agent environments and task rewards.

MoE, MLA, and ultra-long context may be studied as optional topics, but they are not automatic requirements for v3.

### 4.4 v4 and Later

v4 has no predetermined capability list. Its goals will be defined only after a new technical direction forms a clear and stable paradigm.

## 5. Repository Layout

```text
new/
├── README.md
├── README_en.md
├── v1/                     # Current: foundation language model
├── v2/                     # Later: instruction alignment and chat
└── v3/                     # Later: reasoning and agents
```

Each version owns its internal structure, dependencies, experiments, and evaluation. There is no mandatory shared model implementation or unified leaderboard.

## 6. Evaluation Principle

Each version is accepted against its own goals:

| Version | Main validation targets |
|---|---|
| v1 | Pre-training convergence, Loss/PPL, language generation, basic ICL, and scale observations |
| v2 | Instruction following, multi-turn dialogue, knowledge QA, format control, and preference alignment |
| v3 | Verifiable reasoning, Thinking control, tool use, planning, and agent task completion |

Running the same prompt across versions may illustrate historical progress, but it is not a strictly controlled or fair comparison.

## 7. Current Work

Only **v1** is active:

1. establish the directory structure and documentation;
2. define the v1 architecture;
3. define the small, medium, and large configurations;
4. define the data and Tokenizer plan;
5. begin implementation and experiments only after the design is reviewed.

v2 and v3 retain only their stage definitions for now.

## 8. Notes

- Time ranges are aids for understanding technical evolution, not strict academic boundaries.
- Multiple paradigms may coexist in the same year, and individual models may span stages.
- The objective is to deliver a complete architecture, training system, capability profile, and engineering boundary for each technical stage.
- Thinking does not guarantee correctness; reasoning outputs still require verification and tool support.
