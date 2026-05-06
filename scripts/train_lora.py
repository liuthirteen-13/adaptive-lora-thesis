"""LoRA/QLoRA 训练入口，默认 dry-run，不会直接训练大模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_lora_thesis.config import deep_update, load_yaml, print_config, save_yaml
from adaptive_lora_thesis.data import (
    CausalLMPaddingCollator,
    normalize_instruction_record,
    read_jsonl,
    split_train_eval,
    tokenize_records,
)
from adaptive_lora_thesis.lora import (
    build_lora_config,
    build_quantization_config,
    load_rank_pattern,
    prepare_model_for_peft,
    resolve_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA 微调命令行入口")
    parser.add_argument("--config", type=str, default="configs/train_lora.yaml", help="YAML 配置文件")
    parser.add_argument("--model-name-or-path", type=str, help="基础模型名或本地路径")
    parser.add_argument("--data-path", type=str, help="训练 JSONL 路径")
    parser.add_argument("--eval-path", type=str, help="验证 JSONL 路径")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--rank-pattern", type=str, help="PEFT rank_pattern JSON 路径")
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=None, help="是否启用 QLoRA")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None, help="只检查配置，不训练")
    parser.add_argument("--max-samples", type=int, help="最多读取多少训练样本")
    parser.add_argument("--seed", type=int, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    print_config("训练配置", config)

    if config["run"].get("dry_run", True):
        print("当前为 dry-run：未加载模型，未启动训练。")
        return

    train(config)


def build_runtime_config(args: argparse.Namespace) -> dict:
    """合并 YAML 配置与命令行参数。"""
    config = load_yaml(args.config)
    override = {
        "model": {"name_or_path": args.model_name_or_path},
        "data": {
            "train_path": args.data_path,
            "eval_path": args.eval_path,
            "max_samples": args.max_samples,
        },
        "lora": {"rank_pattern": args.rank_pattern},
        "training": {"output_dir": args.output_dir, "seed": args.seed},
        "run": {"dry_run": args.dry_run, "qlora": args.qlora},
    }
    return deep_update(config, override)


def train(config: dict) -> None:
    """执行实际训练；只有 --no-dry-run 时才会进入。"""
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    training_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    qlora = bool(config["run"].get("qlora", False))

    set_seed(int(training_config.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_quantization_config(config.get("quantization"), enabled=qlora)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=resolve_torch_dtype(model_config.get("torch_dtype")),
        device_map=model_config.get("device_map", "auto"),
        quantization_config=quantization_config,
    )

    if bool(training_config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = prepare_model_for_peft(model, qlora=qlora)
    rank_pattern = load_rank_pattern(config["lora"].get("rank_pattern"))
    lora_config = build_lora_config(config["lora"], rank_pattern=rank_pattern)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records, eval_records = load_train_eval_records(config)
    train_dataset = Dataset.from_list(
        tokenize_records(
            train_records,
            tokenizer=tokenizer,
            max_seq_length=int(data_config.get("max_seq_length", 1024)),
            train_on_prompt=bool(data_config.get("train_on_prompt", False)),
        )
    )
    eval_dataset = None
    if eval_records:
        eval_dataset = Dataset.from_list(
            tokenize_records(
                eval_records,
                tokenizer=tokenizer,
                max_seq_length=int(data_config.get("max_seq_length", 1024)),
                train_on_prompt=bool(data_config.get("train_on_prompt", False)),
            )
        )

    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "resolved_config.yaml")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.03)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_steps=int(training_config.get("save_steps", 200)),
        eval_steps=int(training_config.get("eval_steps", 200)),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        bf16=bool(training_config.get("bf16", False)),
        fp16=bool(training_config.get("fp16", False)),
        optim=str(training_config.get("optim", "adamw_torch")),
        report_to=[] if training_config.get("report_to", "none") == "none" else training_config.get("report_to"),
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        remove_unused_columns=False,
        seed=int(training_config.get("seed", 42)),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMPaddingCollator(tokenizer),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    torch.cuda.empty_cache()


def load_train_eval_records(config: dict) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """读取并切分训练/验证数据。"""
    data_config = config["data"]
    train_records = [normalize_instruction_record(record) for record in read_jsonl(data_config["train_path"])]
    max_samples = data_config.get("max_samples")
    if max_samples is not None:
        train_records = train_records[: int(max_samples)]

    if data_config.get("eval_path"):
        eval_records = [normalize_instruction_record(record) for record in read_jsonl(data_config["eval_path"])]
        return train_records, eval_records

    return split_train_eval(
        train_records,
        eval_ratio=float(data_config.get("eval_ratio", 0.0) or 0.0),
        seed=int(config["training"].get("seed", 42)),
    )


if __name__ == "__main__":
    main()

