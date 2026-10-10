#!/usr/bin/env python3
"""v1 基座模型交互式文本生成。

这个脚本提供一个持续交互的命令行入口，但当前模型只有预训练，没有经过
SFT/RLHF/DPO 对齐。因此用户输入会被转换成“问题/答案”文本提示，模型实际
执行的是续写，而不是可靠的指令问答。

特殊命令：

    /help       显示帮助
    /reset      清空当前对话上下文
    /info       显示模型和生成参数
    /quit       退出
    /exit       退出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# 复用评测脚本中已经验证过的设备、checkpoint、tokenizer 和生成逻辑。
V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(V1_ROOT / "src"))

from eval import choose_device, generate, load_checkpoint, load_tokenizer  # noqa: E402
from model import DecoderOnlyTransformer, ModelConfig  # noqa: E402
from train import load_training_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """读取交互式生成参数。"""

    parser = argparse.ArgumentParser(description="Interact with a v1 base model")
    parser.add_argument("--config", type=Path, required=True, help="模型配置，例如 configs/small.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型 checkpoint，例如 outputs/small/last.pt")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="运行设备；macOS Apple Silicon 可使用 mps",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="每次回答最多生成的 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度；0 表示贪心生成")
    parser.add_argument("--top-k", type=int, default=50, help="只从概率最高的 k 个 token 中采样；0 表示不限制")
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="每次只使用当前问题，不保留之前的对话上下文",
    )
    return parser.parse_args()


def print_help() -> None:
    """显示交互命令。"""

    print(
        "\n可用命令：\n"
        "  /help   显示帮助\n"
        "  /reset  清空对话上下文\n"
        "  /info   显示模型信息和生成参数\n"
        "  /quit   退出\n"
        "  /exit   退出\n"
    )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")

    device = choose_device(args.device)
    config, data_config = load_training_config(args.config)
    model_config = ModelConfig.from_mapping(config["model"])
    model = DecoderOnlyTransformer(model_config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    tokenizer = load_tokenizer(data_config)
    model.eval()

    model_name = str(config.get("profile", {}).get("name", "v1-base"))
    step = int(checkpoint.get("step", -1))
    print(f"已加载模型：{model_name}")
    print(f"参数量：{model.estimate_parameters():,} ({model.estimate_parameters() / 1e9:.4f}B)")
    print(f"checkpoint step：{step} | device：{device}")
    print("提示：这是预训练基座模型，当前功能是文本续写，不是经过对齐的聊天助手。")
    print("输入 /help 查看命令，输入 /quit 退出。\n")

    # history 保存纯文本提示和模型回答；每次生成时模型会自动截取最近上下文。
    history: list[str] = []
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            print("已退出。")
            break
        if question == "/help":
            print_help()
            continue
        if question == "/reset":
            history.clear()
            print("对话上下文已清空。")
            continue
        if question == "/info":
            print(
                f"模型：{model_name}\n"
                f"参数量：{model.estimate_parameters():,}\n"
                f"最大上下文：{model_config.max_seq_len} tokens\n"
                f"device：{device}\n"
                f"max_new_tokens：{args.max_new_tokens}\n"
                f"temperature：{args.temperature}\n"
                f"top_k：{args.top_k}\n"
                f"保留历史：{not args.no_history}"
            )
            continue

        # v1-base 没有 user/assistant 特殊 token，因此使用普通文本分隔符。
        turn = f"问题：{question}\n答案："
        prompt = turn if args.no_history else "".join(history) + turn
        try:
            generated = generate(
                model,
                tokenizer,
                prompt,
                device,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
        except RuntimeError as exc:
            # 常见原因是显存不足；给出可操作的提示后保留程序运行。
            print(f"生成失败：{exc}")
            print("可以降低 --max-new-tokens，或使用 /reset 清空上下文。")
            continue

        # generate 返回“提示+续写”，只显示本轮答案部分。
        answer = generated[len(prompt) :] if generated.startswith(prompt) else generated
        answer = answer.strip()
        print(f"模型：{answer}\n")
        if not args.no_history:
            history.append(turn + answer + "\n")


if __name__ == "__main__":
    main()
