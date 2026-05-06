"""普通 LoRA 微调 baseline。

运行示例：
python src/train_lora.py --config configs/lora_r8.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from data import DataCollatorForCausalLM, load_examples, split_train_eval, tokenize_examples
from model import (
    add_lora_adapter,
    count_trainable_parameters,
    ensure_cuda_available,
    load_base_model,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="普通 LoRA 微调 baseline")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径，例如 configs/lora_r8.yaml")
    parser.add_argument("--train-file", type=str, help="覆盖配置中的训练 JSONL 路径")
    parser.add_argument("--eval-file", type=str, help="覆盖配置中的验证 JSONL 路径")
    parser.add_argument("--output-dir", type=str, help="覆盖配置中的输出目录")
    parser.add_argument("--model-name-or-path", type=str, help="覆盖基础模型路径")
    parser.add_argument("--max-samples", type=int, help="最多读取多少训练样本")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_cli_overrides(config, args)
    print_resolved_config(config)

    try:
        ensure_cuda_available()
        train(config)
    except RuntimeError as exc:
        print(f"\n[友好错误] {exc}")
        raise SystemExit(1) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """命令行参数覆盖 YAML，便于实验脚本复用同一配置。"""
    if args.train_file:
        config.setdefault("data", {})["train_file"] = args.train_file
    if args.eval_file:
        config.setdefault("data", {})["eval_file"] = args.eval_file
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
    if args.model_name_or_path:
        config.setdefault("model", {})["name_or_path"] = args.model_name_or_path
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples


def print_resolved_config(config: dict[str, Any]) -> None:
    """打印实际生效配置，方便论文实验记录。"""
    print("\n===== LoRA baseline 配置 =====")
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


def train(config: dict[str, Any]) -> None:
    """执行 LoRA 训练、保存 adapter 和指标。"""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed

    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    lora_config = config["lora"]

    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model = load_base_model(model_config)
    if bool(train_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = add_lora_adapter(model, lora_config)
    trainable_params, total_params, trainable_ratio = count_trainable_parameters(model)

    train_examples = load_examples(
        data_config["train_file"],
        data_format=str(data_config.get("format", "auto")),
        max_samples=data_config.get("max_samples"),
    )
    if data_config.get("eval_file"):
        eval_examples = load_examples(
            data_config["eval_file"],
            data_format=str(data_config.get("format", "auto")),
            max_samples=data_config.get("max_eval_samples"),
        )
    else:
        train_examples, eval_examples = split_train_eval(
            train_examples,
            eval_ratio=float(data_config.get("eval_ratio", 0.0)),
            seed=seed,
        )

    max_length = int(data_config.get("max_length", 1024))
    train_dataset = Dataset.from_list(
        tokenize_examples(
            train_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_prompt=bool(data_config.get("train_on_prompt", False)),
        )
    )
    eval_dataset = None
    if eval_examples:
        eval_dataset = Dataset.from_list(
            tokenize_examples(
                eval_examples,
                tokenizer=tokenizer,
                max_length=max_length,
                train_on_prompt=bool(data_config.get("train_on_prompt", False)),
            )
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_config.get("num_train_epochs", 1)),
        max_steps=int(train_config.get("max_steps", -1)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.03)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        logging_steps=int(train_config.get("logging_steps", 10)),
        save_steps=int(train_config.get("save_steps", 100)),
        eval_steps=int(train_config.get("eval_steps", 100)),
        save_total_limit=int(train_config.get("save_total_limit", 2)),
        fp16=bool(train_config.get("fp16", True)),
        bf16=bool(train_config.get("bf16", False)),
        optim=str(train_config.get("optim", "adamw_torch")),
        report_to=[] if train_config.get("report_to", "none") == "none" else train_config.get("report_to"),
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalLM(tokenizer),
        processing_class=tokenizer,
    )

    start_time = time.perf_counter()
    train_result = trainer.train()
    train_time_seconds = time.perf_counter() - start_time

    eval_metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()

    # PEFT 模型的 save_pretrained 只保存 adapter 权重与 adapter 配置。
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = collect_metrics(
        train_result_metrics=train_result.metrics,
        eval_metrics=eval_metrics,
        train_time_seconds=train_time_seconds,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_ratio=trainable_ratio,
    )
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("\n===== LoRA baseline 指标 =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nadapter 权重已保存到：{output_dir}")


def collect_metrics(
    train_result_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    train_time_seconds: float,
    trainable_params: int,
    total_params: int,
    trainable_ratio: float,
) -> dict[str, Any]:
    """汇总论文实验需要记录的训练指标。"""
    import torch

    max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    metrics = {
        "train_loss": train_result_metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "peak_gpu_memory_mb": round(max_memory_mb, 2),
        "train_time_seconds": round(train_time_seconds, 2),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
    }
    return metrics


if __name__ == "__main__":
    main()

