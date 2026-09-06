# Step 04: Pretraining

本步骤读取第 3 步固定的 train/validation token 分片，从随机权重开始训练 Atman decoder-only 因果语言模型。它复用 Transformers 的 GPT-2 模型实现，不复用 GPT-2 的 tokenizer 或预训练权重。

## 输入与输出

输入：

```text
02_tokenizer/output/atman_tokenizer/
03_encoding/output/manifest.json
03_encoding/output/*.npy
```

每个模型档位有独立输出：

```text
04_pretraining/output/{small|medium|large}/
├── checkpoint-step-2000/
├── final_model/
└── training_metadata.json
```

`final_model/` 同时保存模型权重、配置和 tokenizer，可直接使用 `AutoModelForCausalLM.from_pretrained()` 加载。

## 三档模型

以下参数量按 32K 词表估算，脚本启动时会打印精确值：

| 档位 | 层数 | 隐藏维度 | 注意力头 | 约参数量 | 用途 |
|---|---:|---:|---:|---:|---|
| `small` | 6 | 512 | 8 | 36M | 快速建立可靠基线 |
| `medium` | 8 | 768 | 12 | 82M | 默认推荐的质量/成本平衡 |
| `large` | 8 | 1024 | 16 | 134M | 充分利用扩充后的语料 |

词表、上下文长度和训练数据三档完全一致，因此结果可以直接比较。模型参数量与语料文件大小没有一一对应关系；更多语料主要增加训练步数，而不会自动放大模型。

## 训练改进

- NumPy mmap：只读取当前 shard，避免一次加载全部编码数据。
- DDP 多卡：`torchrun` 每张 GPU 启动一个进程，数据互不重复，梯度自动同步。
- 文档级验证集：训练过程和最终指标使用固定、未参与训练的数据。
- 梯度累积：单卡时三档默认有效 batch 都是 64 条序列。
- 自动混合精度：CUDA 优先 bf16，不支持时 fp16；Apple Silicon 使用 MPS fp16。
- Cosine 学习率：默认前 2% 优化步 warmup，之后余弦衰减。
- 梯度裁剪：默认最大范数 1.0，降低训练突然发散的概率。
- 自动续训：检测最新 `checkpoint-step-*` 并恢复模型、优化器和调度器。

## 正式运行

先训练推荐的 medium：

```bash
python 04_pretraining/pretrain_base_model.py --model-size medium
```

另外两档：

```bash
python 04_pretraining/pretrain_base_model.py --model-size small
python 04_pretraining/pretrain_base_model.py --model-size large
```

默认只完整遍历语料 1 次。对数十亿 token 来说，先完成 1 epoch 并查看验证 loss，再决定是否继续，比直接写死 3 或 5 epoch 更稳妥。

## 硬件调整

默认 batch 偏保守，可根据显存覆盖：

```bash
python 04_pretraining/pretrain_base_model.py \
  --model-size medium \
  --batch-size 16 \
  --gradient-accumulation-steps 4
```

两项乘积仍为 64，因此每次参数更新看到的序列数不变。出现显存不足时减小 `--batch-size`，并按相同比例增大梯度累积。

设备自动选择顺序是 CUDA、MPS、CPU，也可以明确指定：

```bash
python 04_pretraining/pretrain_base_model.py --model-size small --device mps
CUDA_VISIBLE_DEVICES=0 python 04_pretraining/pretrain_base_model.py --model-size medium --device cuda
```

CPU 可以用于功能测试，不建议承担正式预训练。

## CUDA 多卡训练

单机 4 卡训练 medium：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  04_pretraining/pretrain_base_model.py \
  --model-size medium \
  --device cuda
```

只使用指定的 4 张卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone --nnodes=1 --nproc-per-node=4 \
  04_pretraining/pretrain_base_model.py --model-size medium --device cuda
```

这里的 `--batch-size` 是**每张 GPU**的 batch。全局有效 batch 的计算公式是：

```text
global_batch_size = batch_size × gradient_accumulation_steps × GPU 数量
```

例如 medium 默认是 `8 × 8 × 4 = 256` 条序列，而不是单卡时的 64。若希望多卡
仍保持全局 batch=64，可在 4 卡时使用：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  04_pretraining/pretrain_base_model.py \
  --model-size medium \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --device cuda
```

脚本按每个 `.npy` 分片内部的序列行号划分数据，因此 GPU 数量可以多于数据分片
数量。所有 rank 执行相同的优化步数，只有 rank 0 显示进度、执行验证和保存文件。
梯度累积的非更新步使用 DDP `no_sync()`，避免每个微批次都通信。

DDP 是数据并行：每张卡各放一份完整模型，它能提高吞吐量，但不能让单卡原本放不
下的模型突然可训练。本项目最大模型约 134M，适合 DDP；未来模型大到单卡放不下时，
再考虑 FSDP。正式 CUDA 训练使用 NCCL，脚本中的 CPU/Gloo 分支仅用于验证分布式流程。

## 五步试跑

下面只进行 5 个优化步，用于检查前向、反向、验证和输出路径：

```bash
python 04_pretraining/pretrain_base_model.py \
  --model-size small \
  --max-steps 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --device cpu
```

试跑默认也会写到正式 small 目录。为了完全隔离，可以增加：

```text
--output-root 04_pretraining/output_smoke
```

## 常用参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--epochs` | 1 | 完整遍历训练集次数 |
| `--max-steps` | 0 | 0 表示按 epoch；正数用于限时或试跑 |
| `--learning-rate` | 3e-4 | AdamW 峰值学习率 |
| `--warmup-ratio` | 0.02 | 总优化步中的预热比例 |
| `--save-every` | 2000 | checkpoint 间隔 |
| `--eval-every` | 1000 | 训练中验证间隔 |
| `--mixed-precision` | auto | 自动、bf16、fp16 或 no |
| `--auto-resume` | 开启 | 使用 `--no-auto-resume` 可关闭 |

`training_metadata.json` 还会记录 `world_size`、`global_batch_size` 和
`distributed_backend`，便于复现实验。续训时建议使用与原任务相同的 GPU 数量和
batch 配置。

## 完成检查

训练完成后查看：

```text
04_pretraining/output/medium/training_metadata.json
```

重点检查 `status`、`num_parameters`、`completed_steps` 和 `final_validation`。第 5 步目前按要求保持清空，之后应基于这里保存的 `final_model/` 重新设计生成与评估。
