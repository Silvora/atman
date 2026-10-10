# Atman v1：基础语言模型阶段

> 状态：正式语料过滤、划分、Tokenizer、token ID 编码、训练、评测和聊天入口已完成；正式预训练尚未启动。

## 1. 版本定位

v1 是 Atman LLM Evolution Lab 的第一个独立阶段产物，覆盖基础语言模型从数据到训练、评测和生成的完整工程流程。

本版本先训练自己的小规模基础模型，验证数据、Tokenizer、模型、训练和生成链路，再根据资源和实验结果扩展到 medium、large 规模。

这一版本保留早期 GPT 类 Decoder-only 模型的主要特征，重点建立并验证以下核心机制：

- 文本如何变成 token；
- token 如何组成训练样本；
- Decoder-only Transformer 如何预测下一个 token；
- 模型规模、数据规模和训练超参数如何影响结果；
- Loss 和 Perplexity 如何变化；
- 预训练模型能产生什么能力，又存在哪些限制。

v1 完成后将固化其架构、配置、训练结果和能力边界，形成可独立运行、验证与复现的正式版本。

## 2. 研发目标

v1 需要形成以下完整闭环：

```text
原始数据
  → 清洗与规范化
  → 去重与 train/validation 划分
  → Tokenizer 训练与验证
  → 数据编码与定长打包
  → 模型实现
  → 预训练
  → Checkpoint 与恢复
  → Loss/PPL 评测
  → 文本生成
  → 实验记录与阶段总结
```

核心交付内容：

1. 训练和验证自己的 Tokenizer；
2. 实现基础 Decoder-only Transformer；
3. 实现并验证因果注意力和 Next Token Prediction；
4. 完成单卡与多卡预训练实验；
5. 保存并恢复完整训练状态；
6. 分析 small、medium、large 的规模变化；
7. 使用固定提示词记录生成能力；
8. 记录成功实验、失败实验和能力边界。

## 3. 版本边界

v1 计划包含：

- 基础 GPT 类 Decoder-only Transformer
- Causal Language Modeling
- 基础 Tokenizer 训练
- 预训练数据处理
- small / medium / large 三种模型规模
- 单卡和数据并行训练
- Checkpoint、日志和断点恢复
- Loss、Perplexity、固定提示词生成评测
- 基础 zero-shot / few-shot / ICL 能力验证

本版本的交付边界止于预训练基座模型、文本生成、基础上下文能力验证及其完整工程体系。版本范围以本节列出的交付项为准。

## 4. 模型规模

模型部分固定保留三个配置文件：

| 配置 | 定位 | 主要用途 |
|---|---|---|
| `small.yaml` | 最小完整模型 | 验证模型、训练、保存和生成全流程 |
| `medium.yaml` | v1 主实验模型 | 完成主要预训练与能力验证 |
| `large.yaml` | v1 扩展实验模型 | 研究增加容量后的训练成本与效果变化 |

三个配置名称固定为 `small`、`medium`、`large`，不使用 `tiny_a`、`tiny_b`、`tiny_c`。

small、medium、large 的层数、隐藏维度、注意力头数、参数量和训练超参数已经写入各自配置，并在模型实现中进行结构校验。

公共数据与 Tokenizer 配置独立保存，避免在三个模型文件中重复维护。模型配置最终描述结构和训练参数：

```yaml
model:       # 模型结构
training:    # batch、学习率、训练步数、精度等
checkpoint:  # 保存频率和恢复设置
evaluation:  # PPL 与生成评测设置
```

数据处理参数位于 `data.yaml`，分词器参数位于 `tokenizer.yaml`。原始文件的来源、版本、许可、大小、行数和 SHA-256 位于 `data/raw/minimind/manifest.json`。

模型结构参数只在 `model` 部分定义一次，训练脚本不得在代码中重复硬编码。

## 5. 目录结构

```text
./
├── README.md                 # v1 说明、进度和最终结果摘要
├── configs/                  # 数据、Tokenizer 与模型配置
│   ├── data.yaml
│   ├── tokenizer.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── large.yaml
├── data/                     # v1 数据与 Tokenizer 产物
│   ├── raw/                  # 原始数据及可提交的来源清单
│   ├── processed/            # 清洗、去重和划分后的文本
│   ├── encoded/              # 定长 token ID 二进制数据
│   └── tokenizer/            # Tokenizer 配置、词表和训练产物
├── src/                      # 核心 Python 包
│   ├── model/                # 模型结构
│   ├── data/                 # 数据集和数据加载
│   ├── trainer/              # 训练循环、分布式和 Checkpoint
│   └── utils/                # 配置、日志、随机种子等通用能力
├── scripts/                  # 阶段用户入口
│   ├── download_data.sh
│   ├── train_tokenizer.py
│   ├── preprocess_data.py
│   ├── encode_data.py        # 文本编码和定长 token ID 打包
│   ├── train.py              # 预训练入口（单卡/DDP、验证、断点恢复）
│   ├── eval.py               # 模型评测入口
│   └── chat.py               # Checkpoint 交互式生成入口
├── outputs/                  # 训练生成的本地产物，不提交大型文件
│   ├── checkpoints/
│   ├── logs/
│   └── samples/
├── evals/                    # v1 自己的评测定义与结果
│   ├── prompts/              # 固定生成提示词
│   └── results/              # 小型 JSON/Markdown 结果
└── docs/                     # v1 设计和实验过程记录
    ├── design.md
    └── experiments.md
```

## 6. 目录职责

### `configs/`

保存三类相互独立的配置：

- `data.yaml`：原始输入、清洗、去重和数据划分；
- `tokenizer.yaml`：词表训练、评测与自动选择；
- `small.yaml`、`medium.yaml`、`large.yaml`：模型结构与对应训练参数。

共同参数只定义一次，命令行覆盖项仅用于临时验证或受控实验。

### `data/`

只服务于当前版本。原始数据、清洗规则、Tokenizer 和编码格式均按照当前版本的技术目标设计。

- `raw/`：下载后的原始语料；
- `processed/`：清洗、去重和切分后的文本数据；
- `encoded/`：使用正式 Tokenizer 生成的定长 token ID 二进制数据；
- `tokenizer/`：Tokenizer 模型、词表和必要配置。

大型语料和二进制数据不提交 Git，但必须在文档中记录来源、版本、文件校验值、处理参数和统计信息。

### `src/`

保存核心实现代码。核心逻辑与命令行入口分离，便于测试、验证和维护。

### `scripts/`

保存可直接执行的阶段入口。数据下载、预处理和 Tokenizer 入口已经实现；模型训练与评测入口将在模型结构确定后实现。脚本只负责读取配置并调用 `src/` 中的实现，不重复维护核心训练逻辑。

### `outputs/`

保存 Checkpoint、训练日志和生成样例。大型产物不提交 Git；重要指标和代表性样例整理后写入 `evals/results/` 或文档。

### `evals/`

只评测 v1 的阶段目标。固定提示词和轻量结果可以提交，用于复现同一版本内的实验。

### `docs/`

- `design.md`：记录架构、参数量计算、数据格式、训练策略和设计决策；
- `experiments.md`：按时间记录命令、环境、配置、结果、异常与结论。

## 7. 开发顺序

### 阶段 0：目录与设计

- [x] 建立 v1 目录骨架
- [x] 确定 small / medium / large 配置文件名称
- [x] 明确各目录职责
- [x] 完成数据与 Tokenizer 设计
- [ ] 确定三种模型的结构参数

### 阶段 1：数据方案

- [x] 确定并下载固定 revision 的语料
- [x] 记录数据仓库声明的许可
- [x] 确定清洗、去重和切分规则
- [x] 实现流式预处理与审计
- [x] 记录数据规模和处理统计
- [x] 生成正式训练集与验证集

### 阶段 2：Tokenizer

- [x] 确定 Byte-level BPE 算法与候选词表
- [x] 确定特殊 token
- [x] 实现候选训练、评测与自动选择
- [x] 训练并验证 Tokenizer
- [x] 记录压缩率、未知字符和中英文表现

### 阶段 3：模型与训练实现

- [x] 实现文本编码和定长 token ID 打包入口
- [x] 生成正式 token ID 数据
- [x] 实现并测试 Decoder-only Transformer
- [x] 验证参数量和张量形状
- [x] 完成训练、日志和 Checkpoint 入口
- [x] 完成断点恢复代码（待正式训练验证）

### 阶段 4：预训练实验

- [ ] small 流程验证
- [ ] medium 主实验
- [ ] large 扩展实验
- [ ] 记录训练成本、速度和稳定性

### 阶段 5：评测与封存

- [ ] 计算验证集 Loss 和 Perplexity
- [ ] 使用固定提示词生成样例
- [ ] 记录基础 ICL 验证结果
- [ ] 总结 v1 的能力和局限
- [ ] 固化最终配置、依赖和实验记录

## 8. v1 验收标准

只有同时满足以下条件，v1 才视为完成：

1. 数据来源和处理过程可追溯；
2. Tokenizer 可以独立加载并稳定编码、解码；
3. 三种配置均能准确计算模型参数量；
4. small 至少完成端到端训练、保存、重载和生成；
5. 正式实验可以从 Checkpoint 恢复；
6. 验证集 Loss/PPL 和固定提示词结果被记录；
7. 关键训练命令、环境和耗时可复现；
8. `docs/experiments.md` 记录成功与失败实验；
9. README 总结最终结果与 v1 能力边界；
10. 整个版本可以独立部署、运行、验证与复现。

## 9. 版本内实验原则

small、medium、large 用于分析规模变化，同时必须记录控制变量：

- 使用的数据版本；
- 训练 token 数；
- 序列长度；
- 优化器和学习率策略；
- 全局 batch size；
- 硬件和训练精度；
- 随机种子；
- 评测提示词版本。

如果这些条件不同，结果只能作为实验记录，不能解释为单一模型规模造成的变化。

## 10. 当前状态

截至当前阶段：

- v1 目录结构已经建立；
- `data.yaml` 和 `tokenizer.yaml` 已独立保存公共数据与分词参数；
- `small.yaml`、`medium.yaml`、`large.yaml` 只保留各自的模型与训练配置；
- 原始数据来源和完整性信息已记录在 `manifest.json`；
- 正式语料与 mini 语料已经下载；
- 流式校验、清洗、精确去重和稳定划分代码已经实现；
- 16K/32K Byte-level BPE 候选训练、评测和自动选择代码已经实现；
- 正式语料已完成过滤和稳定划分：8,423,079 条训练数据、41,999 条验证数据、3,749 条精确重复数据被删除、无无效 JSON；
- 正式 Tokenizer 已完成训练和选择，最终使用 16,384 词表；
- 16K 和 32K 候选在验证集上的往返一致率均为 100%，未知 token 数为 0；
- 文本编码和定长 token ID 打包入口已经实现，并已通过临时数据冒烟测试；
- 2,000 条数据的端到端冒烟测试已经通过；
- 正式 token ID 数据已经生成：训练集 837,900 条、验证集 4,192 条，每条保存 2,049 个 ID（输入长度 2,048）；
- Decoder-only Transformer 及 small/medium/large 结构和训练参数已经写入配置；
- `scripts/train.py` 已实现训练、验证、日志、checkpoint 和恢复入口；
- 尚未开始任何训练任务。

`data/` 已从已验收的 v1 数据产物复制到本目录，包含原始语料、处理后的 JSONL、Tokenizer 和 2,048 长度编码文件。

下一步依次完成：

1. 使用 `--dry-run` 检查配置、数据和一次前向/反向；
2. 先完成 small 端到端验证，再进行受控规模实验；
3. 根据训练日志记录 Loss、Perplexity、速度和显存占用。

上述正式数据和 Tokenizer 命令已经执行完成；如需重建，必须使用配置文件并显式确认 `--force` 的覆盖影响。

## 11. v1 实现关系

`v1-base` 是基础 Dense Decoder-only 基线。`v1-dense` 在独立目录中实现现代 Dense 结构和长上下文，`v1-moe` 在独立目录中实现 MoE、MLA 和长上下文。三个目录共享数据来源记录，模型代码、配置、编码长度和实验结果分别管理。

## 12. 平台运行

训练入口支持 `auto`、`cpu`、`cuda` 和 Apple Silicon 的 `mps`。macOS 可使用：

```bash
python scripts/train.py --config configs/small.yaml --device mps --dry-run
```

下载脚本使用 Bash；Windows 环境建议使用 WSL。数据处理、Tokenizer 和编码入口使用 Python 标准路径接口，可在 Linux、macOS 和 Windows 上运行。
