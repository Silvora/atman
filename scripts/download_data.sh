#!/usr/bin/env bash

set -Eeuo pipefail

# 所有路径都从脚本自身位置计算，因此可以从任意工作目录启动。
# 修改 TARGET_DIR 会改变原始数据落盘位置，并需要同步 YAML 的 raw_dir。
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly V1_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly RAW_DIR="${V1_ROOT}/data/raw"
readonly TARGET_DIR="${RAW_DIR}/minimind"

# 两个平台保存的是同一份发布数据，但 revision ID 属于各自仓库，不能混用。
# 固定 revision 可以避免远端 master 更新后，同一命令得到不同训练语料。
readonly MODELSCOPE_REPO_ID="gongjy/minimind_dataset"
readonly MODELSCOPE_BASE_URL="https://www.modelscope.cn/datasets/${MODELSCOPE_REPO_ID}/resolve"
readonly MODELSCOPE_DEFAULT_REVISION="74aad49fa4443e7ed640d44bc4e9c7d1fe71ada5"
readonly HUGGINGFACE_REPO_ID="jingyaogong/minimind_dataset"
readonly HUGGINGFACE_DEFAULT_REVISION="312afb4f76391145c6902f765bb51691c09a12f5"

# full 是正式语料，mini 只用于快速验证下载和处理流程。
# 环境变量允许在自动化环境中切换源或 revision，命令行参数拥有更高优先级。
variant="full"
source_name="${MINIMIND_DATA_SOURCE:-modelscope}"
revision="${MINIMIND_DATASET_REVISION:-}"

usage() {
    cat <<'EOF'
用法：
  ./scripts/download_data.sh [选项]

选项：
  --full              下载完整预训练语料 pretrain_t2t.jsonl（默认）
  --mini              下载较小语料 pretrain_t2t_mini.jsonl
  --both              同时下载完整语料和较小语料
  --source SOURCE     下载源：modelscope（默认）或 huggingface
  --revision REV      指定所选数据源的 revision
  -h, --help          显示帮助

环境变量：
  MINIMIND_DATA_SOURCE
                      覆盖默认下载源
  MINIMIND_DATASET_REVISION
                      覆盖默认的数据集 revision
  HF_ENDPOINT         使用 Hugging Face 备用源时的可选镜像地址

下载目录：
  data/raw/minimind/
EOF
}

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

while (( $# > 0 )); do
    # 每个选项都显式 shift，避免参数缺失时被误当成下一个选项。
    case "$1" in
        --full)
            variant="full"
            shift
            ;;
        --mini)
            variant="mini"
            shift
            ;;
        --both)
            variant="both"
            shift
            ;;
        --source)
            (( $# >= 2 )) || die "--source 缺少参数"
            source_name="$2"
            shift 2
            ;;
        --revision)
            (( $# >= 2 )) || die "--revision 缺少参数"
            revision="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "未知参数：$1。使用 --help 查看帮助"
            ;;
    esac
done

case "${variant}" in
    # 不把 full 和 mini 默认合并；两者可能存在样本重叠。
    full)
        dataset_files=("pretrain_t2t.jsonl")
        ;;
    mini)
        dataset_files=("pretrain_t2t_mini.jsonl")
        ;;
    both)
        dataset_files=("pretrain_t2t.jsonl" "pretrain_t2t_mini.jsonl")
        ;;
    *)
        die "不支持的数据集类型：${variant}"
        ;;
esac

mkdir -p "${TARGET_DIR}"

# 根据下载源选择对应的仓库和默认 revision。
# 自定义 --revision 时，调用方必须保证该 revision 属于所选平台。
case "${source_name}" in
    modelscope)
        revision="${revision:-${MODELSCOPE_DEFAULT_REVISION}}"
        repo_id="${MODELSCOPE_REPO_ID}"
        ;;
    huggingface)
        revision="${revision:-${HUGGINGFACE_DEFAULT_REVISION}}"
        repo_id="${HUGGINGFACE_REPO_ID}"
        ;;
    *)
        die "不支持的下载源：${source_name}。可选值：modelscope、huggingface"
        ;;
esac

printf '下载来源：%s\n' "${source_name}"
printf '数据仓库：%s\n' "${repo_id}"
printf '数据版本：%s\n' "${revision}"
printf '下载文件：%s README.md\n' "${dataset_files[*]}"
printf '保存目录：%s\n' "${TARGET_DIR}"

files_to_download=("${dataset_files[@]}" "README.md")

if [[ "${source_name}" == "modelscope" ]]; then
    # wget --continue 会复用未完成文件；重新运行脚本即可断点续传。
    command -v wget >/dev/null 2>&1 || die "未找到 wget，无法从 ModelScope 下载"

    for filename in "${files_to_download[@]}"; do
        url="${MODELSCOPE_BASE_URL}/${revision}/${filename}"
        filepath="${TARGET_DIR}/${filename}"
        printf '\n正在下载：%s\n' "${url}"
        wget_args=(
            "--continue"
            "--tries=10"
            "--timeout=60"
            "--output-document=${filepath}"
            "${url}"
        )
        wget "${wget_args[@]}"
    done
else
    # Hugging Face 作为备用源。优先使用新版 hf，兼容旧版 huggingface-cli。
    if command -v hf >/dev/null 2>&1; then
        downloader=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        downloader=(huggingface-cli download)
    else
        die "未找到 hf 命令。请先安装：python -m pip install -U huggingface_hub"
    fi

    download_args=(
        "${repo_id}"
        "${files_to_download[@]}"
        "--repo-type"
        "dataset"
        "--revision"
        "${revision}"
        "--local-dir"
        "${TARGET_DIR}"
    )

    "${downloader[@]}" "${download_args[@]}"
fi

# 下载命令成功并不代表文件一定有效；至少检查目标文件存在且非空。
# 更严格的字节数和行数校验由 preprocess_data.py 按 YAML 执行。
for filename in "${dataset_files[@]}"; do
    filepath="${TARGET_DIR}/${filename}"
    [[ -s "${filepath}" ]] || die "下载完成后未找到有效文件：${filepath}"
done

[[ -s "${TARGET_DIR}/README.md" ]] || die "下载完成后未找到数据集 README.md"

printf '\n下载完成。文件列表：\n'
for filename in "${dataset_files[@]}" README.md; do
    filepath="${TARGET_DIR}/${filename}"
    if command -v numfmt >/dev/null 2>&1; then
        size="$(stat -c '%s' "${filepath}" | numfmt --to=iec-i --suffix=B)"
    else
        size="$(stat -c '%s bytes' "${filepath}")"
    fi
    printf '  %s  %s\n' "${size}" "${filepath}"
done

printf '\n数据仅完成下载，尚未执行清洗、Tokenizer 训练或预处理。\n'
