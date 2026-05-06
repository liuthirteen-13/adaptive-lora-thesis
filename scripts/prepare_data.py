"""数据准备入口：把 GSM8K 或 instruction JSONL 转为统一 JSONL。"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_lora_thesis.data import (
    normalize_gsm8k_record,
    normalize_instruction_record,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 LoRA 微调数据")
    parser.add_argument("--input", type=str, help="本地 instruction JSONL 输入路径")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 路径")
    parser.add_argument("--format", choices=["instruction", "gsm8k"], default="instruction", help="输入数据格式")
    parser.add_argument("--dataset-name", type=str, help="Hugging Face datasets 名称，例如 gsm8k")
    parser.add_argument("--dataset-config", type=str, default=None, help="datasets 配置名，例如 main")
    parser.add_argument("--split", type=str, default="train", help="datasets split")
    parser.add_argument("--max-samples", type=int, default=None, help="最多保留多少条样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args)

    if args.format == "gsm8k":
        normalized = [normalize_gsm8k_record(record) for record in records]
    else:
        normalized = [normalize_instruction_record(record) for record in records]

    if args.max_samples is not None:
        normalized = normalized[: args.max_samples]

    write_jsonl(normalized, args.output)
    print(f"已写出 {len(normalized)} 条样本到 {Path(args.output)}")


def load_records(args: argparse.Namespace) -> list[dict]:
    """从本地 JSONL 或 Hugging Face datasets 读取原始样本。"""
    if args.input:
        return read_jsonl(args.input)

    if args.dataset_name:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        return [dict(item) for item in dataset]

    raise ValueError("必须提供 --input 或 --dataset-name")


if __name__ == "__main__":
    main()

