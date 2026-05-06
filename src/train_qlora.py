"""QLoRA 4-bit baseline 训练入口。

默认只执行 dry-run，真实训练需显式传入 --run：
python src/train_qlora.py --config configs/qlora_r8.yaml --run
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from data import DataCollatorForCausalLM, load_examples, split_train_eval, tokenize_examples
from model import (
    add_lora_adapter,
    count_trainable_parameters,
    ensure_bitsandbytes_available,
    load_4bit_base_model,
    load_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA 4-bit 微调 baseline")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径，例如 configs/qlora_r8.yaml")
    parser.add_argument("--train-file", type=str, help="覆盖配置中的训练 JSONL 路径")
    parser.add_argument("--eval-file", type=str, help="覆盖配置中的验证 JSONL 路径")
    parser.add_argument("--output-dir", type=str, help="覆盖配置中的 adapter 输出目录")
    parser.add_argument("--model-name-or-path", type=str, help="覆盖基座模型路径")
    parser.add_argument("--max-samples", type=int, help="最多读取多少训练样本，用于小样本调试")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置与路径，不加载模型、不启动训练")
    parser.add_argument("--run", action="store_true", help="显式关闭 dry-run 并启动 QLoRA 训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.run:
        raise SystemExit("--dry-run 与 --run 不能同时使用。")

    config = load_config(args.config)
    apply_cli_overrides(config, args)
    resolve_dry_run(config, args)
    print_resolved_config(config)

    if bool(config.setdefault("run", {}).get("dry_run", True)):
        dry_run(config)
        return

    try:
        ensure_bitsandbytes_available()
        train(config)
    except RuntimeError as exc:
        print(f"\n[友好错误] {exc}")
        raise SystemExit(1) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """命令行参数覆盖 YAML，便于同一份配置复用到不同数据或输出目录。"""
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


def resolve_dry_run(config: dict[str, Any], args: argparse.Namespace) -> None:
    """默认保护为 dry-run；只有 --run 会真正启动大模型训练。"""
    run_config = config.setdefault("run", {})
    dry_run = bool(run_config.get("dry_run", True))
    if args.dry_run:
        dry_run = True
    if args.run:
        dry_run = False
    run_config["dry_run"] = dry_run


def print_resolved_config(config: dict[str, Any]) -> None:
    """打印实际生效配置，方便论文实验记录。"""
    title = "QLoRA baseline 配置（dry-run）" if config.get("run", {}).get("dry_run", True) else "QLoRA baseline 配置"
    print(f"\n===== {title} =====")
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


def dry_run(config: dict[str, Any]) -> None:
    """只检查关键路径和配置，不导入 bitsandbytes、不加载模型。"""
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_config = config.get("training", {})
    quant_config = config.get("quantization", {})

    train_file = Path(data_config.get("train_file", ""))
    eval_file = data_config.get("eval_file")
    checks = {
        "model.name_or_path": model_config.get("name_or_path"),
        "data.train_file": str(train_file),
        "training.output_dir": train_config.get("output_dir"),
        "quantization.load_in_4bit": quant_config.get("load_in_4bit", True),
        "quantization.bnb_4bit_quant_type": quant_config.get("bnb_4bit_quant_type", "nf4"),
    }
    missing = [key for key, value in checks.items() if value in (None, "")]
    if missing:
        raise SystemExit(f"dry-run 配置检查失败，缺少字段：{', '.join(missing)}")
    if not train_file.exists():
        raise SystemExit(f"dry-run 配置检查失败，训练文件不存在：{train_file}")
    if eval_file and not Path(eval_file).exists():
        raise SystemExit(f"dry-run 配置检查失败，验证文件不存在：{eval_file}")

    print("dry-run 检查通过：不会加载模型，也不会启动训练。")
    print("真实启动命令：python src/train_qlora.py --config configs/qlora_r8.yaml --run")


def train(config: dict[str, Any]) -> None:
    """执行 4-bit QLoRA 训练，保存 adapter、指标和 results/logs 训练记录。"""
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed

    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    lora_config = config["lora"]
    quant_config = config.get("quantization", {})

    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()

    output_dir = resolve_output_dir(config)
    log_dir = Path(config.get("logging", {}).get("log_dir", "results/logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model = load_4bit_base_model(model_config, quant_config)

    gradient_checkpointing = bool(train_config.get("gradient_checkpointing", True))
    # prepare_model_for_kbit_training 会冻结量化基座，并为低比特反传准备输入梯度。
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model = add_lora_adapter(model, lora_config)
    assert_only_lora_adapter_trainable(model)
    trainable_params, total_params, trainable_ratio = count_trainable_parameters(model)

    train_dataset, eval_dataset = build_datasets(config, tokenizer, seed)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(log_dir),
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
        fp16=bool(train_config.get("fp16", False)),
        bf16=bool(train_config.get("bf16", True)),
        gradient_checkpointing=gradient_checkpointing,
        optim=str(train_config.get("optim", "paged_adamw_8bit")),
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

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    train_result = trainer.train()
    torch.cuda.synchronize()
    train_time_seconds = time.perf_counter() - start_time

    eval_metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()

    # PEFT save_pretrained 只保存 LoRA adapter 权重和配置，便于与 evaluate.py 共用。
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
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    record_path = write_training_record(config, metrics, trainer.state.log_history, log_dir)

    print("\n===== QLoRA baseline 指标 =====")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nadapter 权重已保存到：{output_dir}")
    print(f"训练记录已保存到：{record_path}")


def build_datasets(config: dict[str, Any], tokenizer: Any, seed: int) -> tuple[Any, Any | None]:
    """复用普通 LoRA 的数据读取与 tokenize 逻辑，保证数据和评估口径一致。"""
    from datasets import Dataset

    data_config = config["data"]
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
    return train_dataset, eval_dataset


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """避免覆盖已有实验目录，默认自动追加时间戳生成可区分输出目录。"""
    train_config = config["training"]
    output_dir = Path(train_config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(train_config.get("overwrite_output_dir", False)):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
        train_config["output_dir"] = str(output_dir)
        print(f"检测到输出目录已有内容，本次 QLoRA 结果将写入：{output_dir}")
    return output_dir


def assert_only_lora_adapter_trainable(model: Any) -> None:
    """确认只有 LoRA adapter 参数参与训练，避免量化基座被误解冻。"""
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected = [name for name in trainable_names if "lora_" not in name]
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise RuntimeError(f"检测到非 LoRA 参数仍可训练：{preview}。请检查 QLoRA 冻结逻辑。")


def collect_metrics(
    train_result_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    train_time_seconds: float,
    trainable_params: int,
    total_params: int,
    trainable_ratio: float,
) -> dict[str, Any]:
    """汇总论文对比需要的损失、显存峰值、训练耗时和可训练参数量。"""
    import torch

    peak_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
    return {
        "method": "qlora_4bit",
        "train_loss": train_result_metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "peak_gpu_memory_mb": round(peak_allocated_mb, 2),
        "peak_gpu_memory_reserved_mb": round(peak_reserved_mb, 2),
        "train_time_seconds": round(train_time_seconds, 2),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_ratio,
    }


def write_training_record(
    config: dict[str, Any],
    metrics: dict[str, Any],
    trainer_log_history: list[dict[str, Any]],
    log_dir: Path,
) -> Path:
    """将完整训练记录写入 results/logs，供论文表格和后处理脚本读取。"""
    run_name = safe_run_name(config.get("run", {}).get("name") or Path(config["training"]["output_dir"]).name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "method": "qlora_4bit",
        "run_name": run_name,
        "timestamp": timestamp,
        "output_dir": config["training"]["output_dir"],
        "metrics": metrics,
        "trainer_log_history": trainer_log_history,
        "resolved_config": config,
    }
    record_path = log_dir / f"{timestamp}_{run_name}_training_record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return record_path


def safe_run_name(value: str) -> str:
    """把实验名规整成适合文件名的形式。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "qlora_run"


if __name__ == "__main__":
    main()
