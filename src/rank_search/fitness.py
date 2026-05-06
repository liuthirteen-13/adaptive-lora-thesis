"""PSO rank_pattern 搜索的适应度函数。

score = eval_metric - lambda_param * param_ratio - lambda_time * time_ratio

默认支持两种模式：
- proxy：只用 importance 与参数量估计快速打分，适合 smoke test；
- train_eval：每个候选 rank_pattern 只训练少量 step，并在验证子集上评估 loss。
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import DataCollatorForCausalLM, load_examples, split_train_eval, tokenize_examples  # noqa: E402
from model import count_trainable_parameters, load_tokenizer  # noqa: E402
from rank_search.build_rank_pattern import (  # noqa: E402
    build_peft_patterns,
    estimate_lora_trainable_params,
    max_lora_trainable_params,
    pattern_rank_budget,
    rank_key_sort_value,
    save_pattern_files,
    write_json,
)
from rank_search.importance import cleanup_memory, has_device_map, load_model_with_lora  # noqa: E402


@dataclass
class FitnessResult:
    """单个 PSO trial 的适应度结果。"""

    trial_id: str
    status: str
    score: float
    eval_metric: float
    param_ratio: float
    time_ratio: float
    trainable_params_estimate: int
    rank_budget: int
    train_time_seconds: float
    eval_time_seconds: float
    peak_gpu_memory_mb: float
    output_dir: str
    failure_reason: str | None = None


def evaluate_trial(
    search_pattern: dict[str, int],
    config: dict[str, Any],
    trial_dir: str | Path,
    trial_id: str,
    seed: int,
    importance: dict[str, dict[str, float | int]] | None = None,
    force: bool = False,
) -> FitnessResult:
    """评估一个 rank_pattern；已完成 trial 会直接读取记录以支持断点恢复。"""
    trial_path = Path(trial_dir)
    trial_path.mkdir(parents=True, exist_ok=True)
    record_path = trial_path / "trial_record.json"
    if record_path.exists() and not force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return FitnessResult(**record["result"])

    peft_patterns = build_peft_patterns(
        search_pattern,
        alpha_multiplier=float(config.get("search", {}).get("alpha_multiplier", 2.0)),
    )
    pattern_files = save_pattern_files(peft_patterns, trial_path, prefix="trial")
    trial_config = build_trial_config(config, pattern_files, trial_path, seed)
    write_json(search_pattern, trial_path / "search_rank_pattern.json")
    write_json(peft_patterns, trial_path / "peft_patterns.json")
    (trial_path / "trial_config.yaml").write_text(
        yaml.safe_dump(trial_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    started_at = time.perf_counter()
    try:
        mode = str(config.get("fitness", {}).get("mode", "train_eval"))
        if mode == "proxy":
            result = evaluate_proxy(search_pattern, config, trial_path, trial_id, importance)
        elif mode == "train_eval":
            result = evaluate_train_eval(search_pattern, trial_config, trial_path, trial_id)
        else:
            raise ValueError(f"不支持的 fitness.mode：{mode}")
    except Exception as exc:  # noqa: BLE001 - 需要记录失败后继续搜索
        elapsed = time.perf_counter() - started_at
        result = failed_result(search_pattern, config, trial_path, trial_id, elapsed, exc)
        (trial_path / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        cleanup_memory()

    record = {
        "trial_id": trial_id,
        "seed": seed,
        "search_rank_pattern": search_pattern,
        "pattern_files": pattern_files,
        "trial_config": trial_config,
        "result": asdict(result),
    }
    write_json(record, record_path)
    return result


def build_trial_config(config: dict[str, Any], pattern_files: dict[str, str], trial_path: Path, seed: int) -> dict[str, Any]:
    """构造单个 trial 的 LoRA 训练配置，保证少量 step 和小子集约束显式落盘。"""
    trial_config = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True)) or {}
    trial_config.setdefault("lora", {})
    trial_config["lora"]["rank_pattern_path"] = pattern_files["rank_pattern_path"]
    trial_config["lora"]["alpha_pattern_path"] = pattern_files["alpha_pattern_path"]
    trial_config["lora"]["exclude_modules_path"] = pattern_files["exclude_modules_path"]

    fitness_config = trial_config.get("fitness", {})
    training_config = trial_config.setdefault("training", {})
    data_config = trial_config.setdefault("data", {})
    training_config["output_dir"] = str(trial_path / "adapter")
    training_config["seed"] = seed
    training_config["max_steps"] = int(fitness_config.get("max_train_steps", 2))
    training_config["num_train_epochs"] = 1
    training_config["save_steps"] = max(1, int(fitness_config.get("max_train_steps", 2)))
    training_config["eval_steps"] = max(1, int(fitness_config.get("max_train_steps", 2)))
    training_config["logging_steps"] = 1
    data_config["max_samples"] = int(fitness_config.get("train_subset_size", 8))
    data_config["max_eval_samples"] = int(fitness_config.get("eval_subset_size", 4))
    data_config["eval_ratio"] = float(fitness_config.get("eval_ratio", data_config.get("eval_ratio", 0.2)))
    return trial_config


def evaluate_proxy(
    search_pattern: dict[str, int],
    config: dict[str, Any],
    trial_path: Path,
    trial_id: str,
    importance: dict[str, dict[str, float | int]] | None,
) -> FitnessResult:
    """只用 importance 加权收益和参数量估计打分，便于快速验证 PSO 逻辑。"""
    started_at = time.perf_counter()
    metric = proxy_eval_metric(search_pattern, importance or {})
    train_time_seconds = time.perf_counter() - started_at
    estimates = estimate_ratios(search_pattern, config, train_time_seconds)
    score = compute_score(
        eval_metric=metric,
        param_ratio=estimates["param_ratio"],
        time_ratio=estimates["time_ratio"],
        config=config,
    )
    return FitnessResult(
        trial_id=trial_id,
        status="success",
        score=score,
        eval_metric=metric,
        param_ratio=estimates["param_ratio"],
        time_ratio=estimates["time_ratio"],
        trainable_params_estimate=estimates["trainable_params_estimate"],
        rank_budget=pattern_rank_budget(search_pattern),
        train_time_seconds=train_time_seconds,
        eval_time_seconds=0.0,
        peak_gpu_memory_mb=0.0,
        output_dir=str(trial_path),
    )


def evaluate_train_eval(
    search_pattern: dict[str, int],
    config: dict[str, Any],
    trial_path: Path,
    trial_id: str,
) -> FitnessResult:
    """真实少步训练和验证 loss 评估；失败由外层捕获并记录。"""
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed

    qlora = bool(config.get("run", {}).get("qlora", False))
    train_config = config["training"]
    data_config = config["data"]
    model_config = config["model"]
    fitness_config = config.get("fitness", {})

    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model = load_model_with_lora(config, qlora=qlora)
    if not qlora and not has_device_map(model):
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
            eval_ratio=float(data_config.get("eval_ratio", 0.2)),
            seed=seed,
        )
        if data_config.get("max_eval_samples") is not None:
            eval_examples = eval_examples[: int(data_config["max_eval_samples"])]

    max_length = int(data_config.get("max_length", 256))
    train_dataset = Dataset.from_list(
        tokenize_examples(
            train_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_prompt=bool(data_config.get("train_on_prompt", False)),
        )
    )
    eval_dataset = Dataset.from_list(
        tokenize_examples(
            eval_examples or train_examples[:1],
            tokenizer=tokenizer,
            max_length=max_length,
            train_on_prompt=bool(data_config.get("train_on_prompt", False)),
        )
    )

    training_args = TrainingArguments(
        output_dir=str(trial_path / "trainer"),
        num_train_epochs=1,
        max_steps=int(train_config.get("max_steps", 2)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(train_config.get("warmup_ratio", 0.0)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        fp16=bool(train_config.get("fp16", torch.cuda.is_available())),
        bf16=bool(train_config.get("bf16", False)),
        gradient_checkpointing=bool(train_config.get("gradient_checkpointing", True)),
        optim=str(train_config.get("optim", "adamw_torch")),
        report_to=[],
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

    train_started = time.perf_counter()
    trainer.train()
    train_time_seconds = time.perf_counter() - train_started

    eval_started = time.perf_counter()
    eval_metrics = trainer.evaluate()
    eval_time_seconds = time.perf_counter() - eval_started

    metric_name = str(fitness_config.get("metric_name", "neg_eval_loss"))
    eval_metric = extract_eval_metric(eval_metrics, metric_name)
    estimates = estimate_ratios(search_pattern, config, train_time_seconds)
    score = compute_score(
        eval_metric=eval_metric,
        param_ratio=estimates["param_ratio"],
        time_ratio=estimates["time_ratio"],
        config=config,
    )
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    write_json(
        {
            "trainer_eval_metrics": eval_metrics,
            "trainable_params_actual": trainable_params,
            "total_params_actual": total_params,
            "trainable_ratio_actual": trainable_ratio,
        },
        trial_path / "metrics_detail.json",
    )

    del trainer, model, tokenizer
    cleanup_memory()
    return FitnessResult(
        trial_id=trial_id,
        status="success",
        score=score,
        eval_metric=eval_metric,
        param_ratio=estimates["param_ratio"],
        time_ratio=estimates["time_ratio"],
        trainable_params_estimate=estimates["trainable_params_estimate"],
        rank_budget=pattern_rank_budget(search_pattern),
        train_time_seconds=train_time_seconds,
        eval_time_seconds=eval_time_seconds,
        peak_gpu_memory_mb=round(peak_memory, 2),
        output_dir=str(trial_path),
    )


def extract_eval_metric(eval_metrics: dict[str, Any], metric_name: str) -> float:
    """把 Trainer eval_loss 转成最大化指标；其余指标直接读取。"""
    if metric_name in {"neg_eval_loss", "eval_loss"}:
        return -float(eval_metrics["eval_loss"])
    if metric_name not in eval_metrics:
        raise KeyError(f"评估指标 {metric_name} 不存在，已有指标：{sorted(eval_metrics)}")
    return float(eval_metrics[metric_name])


def proxy_eval_metric(
    search_pattern: dict[str, int],
    importance: dict[str, dict[str, float | int]],
) -> float:
    """importance 加权的代理收益，rank 越接近高重要性层越高。"""
    if not search_pattern:
        return 0.0
    max_rank = max(max(search_pattern.values()), 1)
    norms = {key: float(value.get("grad_norm", 0.0)) for key, value in importance.items()}
    mean_norm = sum(norms.values()) / len(norms) if norms else 1.0
    if mean_norm <= 0:
        mean_norm = 1.0
    weighted_sum = 0.0
    weight_total = 0.0
    for key, rank in search_pattern.items():
        weight = norms.get(key, mean_norm) / mean_norm
        weighted_sum += weight * (max(0, int(rank)) / max_rank)
        weight_total += weight
    return weighted_sum / weight_total if weight_total else 0.0


def estimate_ratios(search_pattern: dict[str, int], config: dict[str, Any], train_time_seconds: float) -> dict[str, float | int]:
    """估算参数比例和时间比例，供统一 score 公式使用。"""
    model_config = config.get("model", {})
    fitness_config = config.get("fitness", {})
    hidden_size = int(model_config.get("hidden_size", 1536))
    intermediate_size = int(model_config.get("intermediate_size", 8960))
    trainable_params = estimate_lora_trainable_params(
        search_pattern,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    param_budget = int(fitness_config.get("param_budget", 0) or 0)
    if param_budget <= 0:
        max_rank = max((int(rank) for rank in config.get("search", {}).get("candidate_ranks", [32])), default=32)
        param_budget = max_lora_trainable_params(
            list(search_pattern),
            max_rank=max_rank,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    time_budget = float(fitness_config.get("time_budget_seconds", 1.0) or 1.0)
    return {
        "trainable_params_estimate": trainable_params,
        "param_ratio": trainable_params / param_budget if param_budget else 0.0,
        "time_ratio": train_time_seconds / time_budget,
    }


def compute_score(eval_metric: float, param_ratio: float, time_ratio: float, config: dict[str, Any]) -> float:
    """统一适应度公式。"""
    fitness_config = config.get("fitness", {})
    lambda_param = float(fitness_config.get("lambda_param", 0.1))
    lambda_time = float(fitness_config.get("lambda_time", 0.05))
    return float(eval_metric - lambda_param * param_ratio - lambda_time * time_ratio)


def failed_result(
    search_pattern: dict[str, int],
    config: dict[str, Any],
    trial_path: Path,
    trial_id: str,
    elapsed_seconds: float,
    exc: Exception,
) -> FitnessResult:
    """把失败 trial 记录成可恢复结果，避免一次失败中断整个 PSO。"""
    estimates = estimate_ratios(search_pattern, config, elapsed_seconds)
    return FitnessResult(
        trial_id=trial_id,
        status="failed",
        score=-1.0e30,
        eval_metric=-1.0e30,
        param_ratio=float(estimates["param_ratio"]),
        time_ratio=float(estimates["time_ratio"]),
        trainable_params_estimate=int(estimates["trainable_params_estimate"]),
        rank_budget=pattern_rank_budget(search_pattern),
        train_time_seconds=elapsed_seconds,
        eval_time_seconds=0.0,
        peak_gpu_memory_mb=current_peak_gpu_memory_mb(),
        output_dir=str(trial_path),
        failure_reason=f"{type(exc).__name__}: {exc}",
    )


def current_peak_gpu_memory_mb() -> float:
    """读取当前 CUDA 显存峰值；CPU 环境返回 0。"""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2)
    except Exception:
        pass
    return 0.0


def cleanup_trial_memory() -> None:
    """兼容旧调用名的显存清理入口。"""
    gc.collect()
    cleanup_memory()
