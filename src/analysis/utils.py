"""论文图表生成的实验记录读取与归一化工具。"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


METHOD_ORDER = ["Base", "Prompt", "LoRA", "QLoRA", "AdaLoRA", "Ours"]
SENTINEL_ABS_LIMIT = 1.0e20


@dataclass
class ExperimentRecord:
    """统一后的实验记录，便于表格和图共用。"""

    method: str
    run_name: str
    status: str
    metric_name: str
    eval_metric: float | None
    eval_loss: float | None
    train_loss: float | None
    score: float | None
    trainable_params: int | None
    total_params: int | None
    trainable_ratio: float | None
    param_ratio: float | None
    train_time_seconds: float | None
    eval_time_seconds: float | None
    peak_gpu_memory_mb: float | None
    rank_budget: int | None
    output_dir: str
    record_path: str
    loss_history: list[dict[str, float]]


def collect_experiment_records(
    logs_dir: str | Path = "results/logs",
    include_discovered: bool = True,
) -> list[ExperimentRecord]:
    """读取 results/logs，并兼容当前工程已有的 outputs/ 与 PSO trial 记录。"""
    records: list[ExperimentRecord] = []
    logs_path = Path(logs_dir)
    if logs_path.exists():
        for path in sorted(logs_path.rglob("*.json")):
            records.extend(records_from_json(path, source_hint="logs"))

    if include_discovered:
        metric_output_dirs = {
            path.parent.resolve() for path in sorted(Path("outputs").glob("*/training_metrics.json"))
        }
        for path in sorted(Path("outputs").glob("*/training_metrics.json")):
            records.extend(records_from_json(path, source_hint="training_metrics"))
        for path in sorted(Path("outputs").glob("*/checkpoint-*/trainer_state.json")):
            output_dir = path.parents[1].resolve() if len(path.parents) > 1 else path.parent.resolve()
            if output_dir in metric_output_dirs:
                continue
            records.extend(records_from_json(path, source_hint="trainer_state"))
        for path in sorted(Path("experiments/rank_search").glob("**/trial_record.json")):
            records.extend(records_from_json(path, source_hint="pso_trial"))

    return deduplicate_records(records)


def records_from_json(path: str | Path, source_hint: str = "") -> list[ExperimentRecord]:
    """从单个 JSON 文件中抽取 0 到多条统一记录。"""
    json_path = Path(path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if "result" in data and isinstance(data["result"], dict):
        return [record_from_pso_trial(data, json_path)]
    if "metrics" in data and isinstance(data["metrics"], dict):
        return [record_from_training_record(data, json_path)]
    if any(key in data for key in ("train_loss", "eval_loss", "peak_gpu_memory_mb", "train_time_seconds")):
        return [record_from_metrics(data, json_path)]
    if "log_history" in data and isinstance(data["log_history"], list):
        return [record_from_trainer_state(data, json_path)]
    return []


def record_from_training_record(data: dict[str, Any], path: Path) -> ExperimentRecord:
    """归一化 results/logs 下完整训练记录。"""
    metrics = data.get("metrics", {})
    method = normalize_method(data.get("method") or data.get("run_name") or path)
    run_name = str(data.get("run_name") or Path(str(data.get("output_dir", path.parent))).name)
    eval_loss = as_float(metrics.get("eval_loss"))
    eval_metric = extract_eval_metric(metrics, default_from_eval_loss=eval_loss)
    return ExperimentRecord(
        method=method,
        run_name=run_name,
        status=str(data.get("status", "success")),
        metric_name=infer_metric_name(metrics),
        eval_metric=eval_metric,
        eval_loss=eval_loss,
        train_loss=as_float(metrics.get("train_loss")),
        score=as_float(metrics.get("score")),
        trainable_params=as_int(metrics.get("trainable_params")),
        total_params=as_int(metrics.get("total_params")),
        trainable_ratio=as_float(metrics.get("trainable_ratio")),
        param_ratio=as_float(metrics.get("param_ratio")),
        train_time_seconds=as_float(metrics.get("train_time_seconds")),
        eval_time_seconds=as_float(metrics.get("eval_time_seconds")),
        peak_gpu_memory_mb=as_float(metrics.get("peak_gpu_memory_mb")),
        rank_budget=as_int(metrics.get("rank_budget")),
        output_dir=str(data.get("output_dir", "")),
        record_path=str(path),
        loss_history=normalize_loss_history(data.get("trainer_log_history", [])),
    )


def record_from_metrics(data: dict[str, Any], path: Path) -> ExperimentRecord:
    """归一化 training_metrics.json。"""
    method = normalize_method(data.get("method") or path.parent.name)
    run_name = path.parent.name
    eval_loss = as_float(data.get("eval_loss"))
    return ExperimentRecord(
        method=method,
        run_name=run_name,
        status=str(data.get("status", "success")),
        metric_name=infer_metric_name(data),
        eval_metric=extract_eval_metric(data, default_from_eval_loss=eval_loss),
        eval_loss=eval_loss,
        train_loss=as_float(data.get("train_loss")),
        score=as_float(data.get("score")),
        trainable_params=as_int(data.get("trainable_params")),
        total_params=as_int(data.get("total_params")),
        trainable_ratio=as_float(data.get("trainable_ratio")),
        param_ratio=as_float(data.get("param_ratio")),
        train_time_seconds=as_float(data.get("train_time_seconds")),
        eval_time_seconds=as_float(data.get("eval_time_seconds")),
        peak_gpu_memory_mb=as_float(data.get("peak_gpu_memory_mb") or data.get("peak_gpu_memory_reserved_mb")),
        rank_budget=as_int(data.get("rank_budget")),
        output_dir=str(path.parent),
        record_path=str(path),
        loss_history=[],
    )


def record_from_trainer_state(data: dict[str, Any], path: Path) -> ExperimentRecord:
    """归一化 Trainer state，用于补充 loss 曲线。"""
    history = normalize_loss_history(data.get("log_history", []))
    last_eval_loss = last_value(history, "eval_loss")
    last_train_loss = last_value(history, "loss")
    run_name = path.parents[1].name if len(path.parents) > 1 else path.parent.name
    return ExperimentRecord(
        method=normalize_method(run_name),
        run_name=f"{run_name}_trainer_state",
        status="success",
        metric_name="neg_eval_loss",
        eval_metric=-last_eval_loss if last_eval_loss is not None else None,
        eval_loss=last_eval_loss,
        train_loss=last_train_loss,
        score=None,
        trainable_params=None,
        total_params=None,
        trainable_ratio=None,
        param_ratio=None,
        train_time_seconds=None,
        eval_time_seconds=None,
        peak_gpu_memory_mb=None,
        rank_budget=None,
        output_dir=str(path.parent),
        record_path=str(path),
        loss_history=history,
    )


def record_from_pso_trial(data: dict[str, Any], path: Path) -> ExperimentRecord:
    """归一化 PSO trial_record.json。"""
    result = data.get("result", {})
    detail_path = path.parent / "metrics_detail.json"
    detail = {}
    if detail_path.exists():
        try:
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            detail = {}

    trainer_metrics = detail.get("trainer_eval_metrics", {})
    eval_loss = as_float(trainer_metrics.get("eval_loss"))
    eval_metric = as_float(result.get("eval_metric"))
    if eval_metric is None and eval_loss is not None:
        eval_metric = -eval_loss
    # 搜索失败时会写入极大惩罚值，这类占位值不能进入论文图表。
    inferred_eval_loss = eval_loss if eval_loss is not None else (-eval_metric if eval_metric is not None else None)
    return ExperimentRecord(
        method="Ours",
        run_name=str(data.get("trial_id") or path.parent.name),
        status=str(result.get("status", "unknown")),
        metric_name="neg_eval_loss",
        eval_metric=eval_metric,
        eval_loss=as_float(inferred_eval_loss),
        train_loss=None,
        score=as_float(result.get("score")),
        trainable_params=as_int(detail.get("trainable_params_actual") or result.get("trainable_params_estimate")),
        total_params=as_int(detail.get("total_params_actual")),
        trainable_ratio=as_float(detail.get("trainable_ratio_actual")),
        param_ratio=as_float(result.get("param_ratio")),
        train_time_seconds=as_float(result.get("train_time_seconds")),
        eval_time_seconds=as_float(result.get("eval_time_seconds")),
        peak_gpu_memory_mb=as_float(result.get("peak_gpu_memory_mb")),
        rank_budget=as_int(result.get("rank_budget")),
        output_dir=str(result.get("output_dir", path.parent)),
        record_path=str(path),
        loss_history=trial_loss_history(result, trainer_metrics),
    )


def normalize_method(value: Any) -> str:
    """把文件名、run_name 或 method 字段映射到论文中的方法名。"""
    text = str(value).lower()
    if "ours" in text or "pso" in text or "adaptive" in text:
        return "Ours"
    if "adalora" in text or "ada_lora" in text:
        return "AdaLoRA"
    if "qlora" in text or "4bit" in text or "4-bit" in text:
        return "QLoRA"
    if "lora" in text:
        return "LoRA"
    if "prompt" in text:
        return "Prompt"
    if "base" in text or "zero" in text:
        return "Base"
    return "LoRA"


def infer_metric_name(metrics: dict[str, Any]) -> str:
    """识别主要性能指标；没有显式指标时使用 -eval_loss。"""
    for name in ("eval_exact_match", "exact_match", "accuracy", "eval_accuracy", "eval_metric"):
        if name in metrics:
            return name
    return "neg_eval_loss"


def extract_eval_metric(metrics: dict[str, Any], default_from_eval_loss: float | None) -> float | None:
    """提取越大越好的性能指标。"""
    for name in ("eval_exact_match", "exact_match", "accuracy", "eval_accuracy", "eval_metric"):
        value = as_float(metrics.get(name))
        if value is not None:
            return value
    if default_from_eval_loss is not None:
        return -default_from_eval_loss
    return None


def normalize_loss_history(history: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    """抽取 step/loss/eval_loss，供 loss 曲线绘图。"""
    normalized: list[dict[str, float]] = []
    for idx, item in enumerate(history, start=1):
        row: dict[str, float] = {"step": as_float(item.get("step") or idx) or float(idx)}
        loss = as_float(item.get("loss") or item.get("train_loss"))
        eval_loss = as_float(item.get("eval_loss"))
        if loss is not None:
            row["loss"] = loss
        if eval_loss is not None:
            row["eval_loss"] = eval_loss
        if len(row) > 1:
            normalized.append(row)
    return normalized


def trial_loss_history(result: dict[str, Any], trainer_metrics: dict[str, Any]) -> list[dict[str, float]]:
    """PSO trial 通常只有最终 eval loss，这里构造成单点曲线。"""
    history = []
    eval_loss = as_float(trainer_metrics.get("eval_loss"))
    if eval_loss is not None:
        history.append({"step": 1.0, "eval_loss": eval_loss})
    return history


def best_records_by_method(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    """每个方法选一条最佳记录；性能越高越好，失败记录自动靠后。"""
    best: dict[str, ExperimentRecord] = {}
    for record in records:
        if record.status != "success":
            continue
        old = best.get(record.method)
        if old is None or sort_score(record) > sort_score(old):
            best[record.method] = record
    return [best[method] for method in METHOD_ORDER if method in best]


def method_sort_key(method: str) -> tuple[int, str]:
    """按论文方法顺序排序。"""
    return (METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER), method)


def sort_score(record: ExperimentRecord) -> float:
    """用于选 best 的分数。"""
    if record.eval_metric is not None and math.isfinite(record.eval_metric):
        return record.eval_metric
    if record.score is not None and math.isfinite(record.score):
        return record.score
    return -1.0e30


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """写 CSV，使用 utf-8-sig 便于 Excel 直接打开中文表头。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """读取 make_tables.py 输出的 CSV。"""
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_rank_pattern(path: str | Path | None = None) -> dict[str, int]:
    """读取 rank 分布；默认优先使用 PSO 最优搜索 pattern。"""
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("experiments/rank_search/ours_pso/best_peft_patterns.json"),
            Path("experiments/rank_search/ours_pso/best_summary.json"),
            Path("experiments/rank_search/ours_pso/best_rank_pattern.json"),
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "search_rank_pattern" in data:
            return {str(key): int(value) for key, value in data["search_rank_pattern"].items()}
        if isinstance(data, dict) and "rank_pattern" in data:
            return {str(key): int(value) for key, value in data["rank_pattern"].items()}
        if isinstance(data, dict):
            return {str(key): int(value) for key, value in data.items()}
    return {}


def deduplicate_records(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    """按 record_path/run_name 去重，避免 checkpoint 和主 metrics 重复污染主表。"""
    seen: set[tuple[str, str]] = set()
    unique: list[ExperimentRecord] = []
    for record in records:
        key = (record.method, record.record_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return sorted(unique, key=lambda record: (method_sort_key(record.method), record.run_name))


def as_float(value: Any) -> float | None:
    """安全转换浮点数。"""
    if value in (None, "", "nan"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    # PSO 搜索失败时使用 +/-1e30 作为惩罚占位，汇总图表应视为空值。
    if abs(number) >= SENTINEL_ABS_LIMIT:
        return None
    return number


def as_int(value: Any) -> int | None:
    """安全转换整数。"""
    number = as_float(value)
    return int(number) if number is not None else None


def format_number(value: float | int | None, digits: int = 4) -> str:
    """CSV 中统一数字格式，空值写为空字符串。"""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def last_value(history: list[dict[str, float]], key: str) -> float | None:
    """取曲线中某字段最后一次出现的值。"""
    for item in reversed(history):
        if key in item:
            return item[key]
    return None


def safe_filename(value: str) -> str:
    """把方法名转换成安全文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower() or "record"
