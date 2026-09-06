# Step 01: Public Corpus

本步骤从 5 个公开中文语料来源的指定文件中流式读取数据，实际有多少合格文本就输出多少，不要求清洗结果必须达到 18 GiB。它只负责下载、清洗、过滤、去重和记录来源，不训练 tokenizer，也不训练模型。

## 输出目录规则

每一个步骤都在自己的目录中创建 `output` 文件夹。本步骤的所有生成物写入：

```text
01_public_corpus/output/
```

后续步骤也使用相同规则：

```text
02_tokenizer/output/
03_encoding/output/
04_pretraining/output/
```

数据记录中的 `generated_by_step` 字段，以及清单中的 `generated_by_step` 和 `generated_by_step_name`，用于标记生成它的步骤。

## 为什么重新做第1步

原始语料主要是中文 Wikipedia，模型容易形成单一的百科续写风格。新版混合百科、教育网页、新闻、书籍、技术文章和合成教材，并在全部来源之间去重。

增加语料不会增加模型参数量。模型大小由层数、隐藏维度、注意力头数和词表大小决定；语料变大只会增加训练时间和模型学到的信息。

## 数据源配置

5 个数据集的下载地址和读取参数统一保存在：

```text
01_public_corpus/sources.json
```

每个数据源配置包含：

| 字段 | 作用 |
|---|---|
| `source_id` | 本项目内部使用的唯一来源名称 |
| `dataset` | 传给 Hugging Face `load_dataset` 的仓库名称 |
| `url` | 数据集页面和下载地址 |
| `license_note` | 许可与访问条件提示 |
| `text_fields` | 依次尝试读取的正文字段 |
| `title_fields` | 可选标题字段 |
| `config` | Hugging Face 数据集子配置，例如中文子集 |
| `split` | 读取的数据分片 |
| `data_files` | 指定仓库中的数据文件或匹配规则 |
| `download_bytes` | Hugging Face API 返回的所选文件精确字节数 |
| `requires_access` | 是否需要登录并提前接受访问条款 |

配置文件不保存固定的清洗结果大小或混合比例。它明确列出本次读取的文件，默认运行会处理完这些文件，最终产量由解压、清洗和过滤结果决定，并记录在 `output/manifest.json`。

当前选择如下。大小来自 Hugging Face 文件页面，仅用于估算网络流量：

| 来源 | 文件数量 | 预估下载大小 |
|---|---:|---:|
| Wikipedia 中文 | 1 | 2.394 GB |
| Chinese FineWeb Edu V2 | 5 | 4.858 GB |
| CCI2 | 2 | 3.836 GB |
| TeleChat-PTD | 2 | 5.215 GB |
| Chinese Cosmopedia | 2 | 2.429 GB |
| **合计** | **12** | **18.731 GB / 17.445 GiB** |

这里共有 12 个文件：Wikipedia 的一个 JSON 文件和其他来源的 11 个语料分片。18.731 GB 是 Hugging Face API 返回的远端文件大小之和，不是输出目标；Parquet/Gzip 解压后通常更大，过滤后又会缩小，所以最终 `totals.text_gib` 可以低于或高于这个数字。

标准 JSON 不能使用 `//` 或 `#` 行注释，因此配置采用 `_comments` 和 `comment` 字段保存中文说明。脚本只读取训练所需字段，会忽略这些说明字段；这样既能直接阅读注释，也能继续使用标准 JSON 工具检查格式。

数据源在配置文件中的排列顺序也是去重优先级。大型网页库之间存在重叠，脚本会在全部来源之间做统一内容指纹去重；相同文章优先保留靠前的数据源版本。

公开可下载不等于没有使用条件。运行前应打开每个数据集页面，检查当前许可证、用户协议、署名要求和商业用途限制。脚本会把地址和许可提示写入 `manifest.json`，但这些提示不替代原始许可文本。

## 环境准备

脚本需要 Python 3.10 或更高版本。先确认当前训练环境：

```bash
python --version
```

安装依赖：

```bash
pip install datasets tqdm
```

建议登录 Hugging Face，提高下载限额：

```bash
hf auth login
```

脚本会自动使用 Hugging Face 已保存的登录状态；也支持从环境变量读取 `HF_TOKEN`，不会把 token 写入输出。

## 查看来源

在项目根目录运行：

```bash
python 01_public_corpus/prepare_public_corpus.py --list-sources
```

使用另一份配置文件：

```bash
python 01_public_corpus/prepare_public_corpus.py \
  --sources-config 01_public_corpus/my_sources.json \
  --list-sources
```

## 小规模试跑

正式下载前，先用两个来源、每个最多 100 条检查环境和字段：

```bash
python 01_public_corpus/prepare_public_corpus.py \
  --sources wikipedia_zh,fineweb_edu_zh \
  --max-gb-per-source 0.01 \
  --max-samples-per-source 100 \
  --output-dir 01_public_corpus/output_smoke
```

试跑输出与正式输出分开，不会污染正式语料。

## 正式运行

处理配置文件中明确列出的全部文件：

```bash
python 01_public_corpus/prepare_public_corpus.py
```

默认输出位置由脚本自身决定，所以无论从项目根目录还是其他目录启动，都会写入：

```text
01_public_corpus/output/
```

脚本不会再读取这 5 个仓库中约 1.53 TB 的全部文件。正式运行前仍应预留明显高于 18.731 GB 的缓存和输出空间，并用上面的小规模命令确认字段和过滤效果。

如果希望每个来源最多保留 1 GiB 有效正文：

```bash
python 01_public_corpus/prepare_public_corpus.py --max-gb-per-source 1
```

该命令最多得到约 5 GiB 正文，但某个来源数据不足或大量内容未通过过滤时，实际值会更小。不传此参数时会处理完 `data_files` 中列出的文件。

## 中断与继续

脚本按来源写入独立分片。每完成一个来源，就更新一次 `manifest.json`。

- 已完成的来源再次运行时会跳过。
- 正在处理的来源如果中断，会留下 `.partial` 文件。
- 再次运行相同命令时，该来源会重新开始，已经完成的来源不会重做。
- 续跑时来源列表、可选的每来源上限和过滤参数必须保持一致。

不要同时启动两个进程写同一个输出目录。

## 输出结构

```text
01_public_corpus/output/
├── corpus_wikipedia_zh.jsonl
├── corpus_fineweb_edu_zh.jsonl
├── corpus_cci2_zh.jsonl
├── ...
└── manifest.json
```

每行格式如下：

```json
{"text": "清洗后的正文", "source": "wikipedia_zh", "generated_by_step": "01_public_corpus"}
```

`manifest.json` 记录：

- 每个来源的地址、许可提示和实际完成原因。
- 检查、保留、过滤和重复文档数量。
- 每个分片的正文大小和实际文件大小。
- 当前状态是 `running`、`interrupted`、`failed` 还是 `completed`。

每个来源的 `completion_reason` 用来说明停止原因：

- `dataset_exhausted`：该数据源已经读取完毕。
- `max_gb_reached`：达到了显式指定的每来源容量上限。
- `max_samples_reached`：达到了试跑时指定的样本数上限。

## 清洗与过滤规则

当前脚本会：

- 统一换行、空白和 HTML 实体。
- 去掉 HTML 标签和控制字符。
- 将 URL、邮箱和中国大陆手机号替换为占位符。
- 删除相邻重复行。
- 默认过滤少于 200 字符的短文本。
- 默认要求有效字符中的中文比例不低于 30%。
- 过滤明显乱码和单字异常重复文本。
- 将超过 100000 字符的文档分段。
- 使用忽略空白与标点的 BLAKE2 指纹做跨来源精确去重。

这里没有做语义级近重复判断。当前版本先用可解释、低依赖的规则建立可靠基线；语料规模很大时，可以再增加 MinHash 去重阶段。

## 完成标准

第1步完成后检查：

```bash
python -m json.tool 01_public_corpus/output/manifest.json
```

满足以下条件才进入第2步：

- `status` 为 `completed`。
- `totals.text_gib` 与磁盘上的实际分片规模一致，不要求接近某个固定值。
- 所有选定来源都有对应 JSONL 分片。
- 随机抽查文本没有大量广告、乱码、菜单和重复内容。

第2步将读取本步骤的 `manifest.json`，并从头训练新版 tokenizer。按照这次约定，后续不会跳过任何步骤。
