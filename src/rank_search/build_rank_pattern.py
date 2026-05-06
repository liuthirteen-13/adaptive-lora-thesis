"""LoRA rank_pattern / alpha_pattern 构造工具。

PSO 搜索空间允许 rank=0；PEFT 的 LoraConfig 不能直接把 0 写入 rank_pattern，
因此本模块把 0 rank 转换为 exclude_modules，其余正 rank 写入 rank_pattern。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


def load_importance(path: str | Path | None) -> dict[str, dict[str, float | int]]:
    """读取 importance JSON；为空时返回空字典，方便无 importance 的冷启动搜索。"""
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(key): dict(value) for key, value in data.items()}


def resolve_importance_path(pattern: str | None) -> Path | None:
    """支持从 results/importance/*.json 这类 glob 中选择最新 importance 文件。"""
    if not pattern:
        return None
    paths = sorted(Path().glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if paths:
        return paths[0]
    path = Path(pattern)
    return path if path.exists() else None


def build_rank_keys(
    num_hidden_layers: int,
    target_modules: list[str],
    importance: dict[str, dict[str, float | int]] | None = None,
) -> list[str]:
    """生成 layer-wise rank key；有 importance 时优先沿用其中的真实 PEFT key。"""
    if importance:
        return sorted(importance.keys(), key=rank_key_sort_value)

    keys: list[str] = []
    for layer_idx in range(num_hidden_layers):
        for module_name in target_modules:
            keys.append(f"model.layers.{layer_idx}.{normalize_target_module(module_name)}")
    return keys


def normalize_target_module(module_name: str) -> str:
    """把 q_proj/v_proj 这类短名展开为 Qwen 常用层级名。"""
    if "." in module_name:
        return module_name
    if module_name in ATTENTION_MODULES:
        return f"self_attn.{module_name}"
    if module_name in MLP_MODULES:
        return f"mlp.{module_name}"
    return module_name


def rank_key_sort_value(key: str) -> tuple[int, str]:
    """按 layer index 再按模块名排序，使 JSON 输出稳定。"""
    match = re.search(r"layers\.(\d+)\.", key)
    layer_idx = int(match.group(1)) if match else -1
    return layer_idx, key


def nearest_candidate_rank(value: int | float, candidate_ranks: list[int]) -> int:
    """把 suggested_rank 或 PSO 连续位置投影到候选 rank 集合。"""
    if not candidate_ranks:
        raise ValueError("candidate_ranks 不能为空")
    return min(candidate_ranks, key=lambda rank: (abs(rank - value), rank))


def importance_seed_pattern(
    keys: list[str],
    importance: dict[str, dict[str, float | int]],
    candidate_ranks: list[int],
    default_rank: int,
) -> dict[str, int]:
    """根据 importance 中的 suggested_rank 生成一个确定性初始粒子。"""
    pattern: dict[str, int] = {}
    for key in keys:
        suggested_rank = int(importance.get(key, {}).get("suggested_rank", default_rank))
        pattern[key] = nearest_candidate_rank(suggested_rank, candidate_ranks)
    return pattern


def build_peft_patterns(
    search_pattern: dict[str, int],
    alpha_multiplier: float,
    min_positive_rank: int = 1,
) -> dict[str, Any]:
    """把搜索 pattern 转换为 PEFT LoraConfig 可用的 rank/alpha/exclude 三件套。"""
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    zero_rank_modules: list[str] = []

    for key, rank in sorted(search_pattern.items(), key=lambda item: rank_key_sort_value(item[0])):
        rank = int(rank)
        if rank <= 0:
            zero_rank_modules.append(key)
            continue
        peft_rank = max(rank, min_positive_rank)
        rank_pattern[key] = peft_rank
        alpha_pattern[key] = max(1, int(round(peft_rank * alpha_multiplier)))

    return {
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "exclude_modules": zero_rank_modules,
        "search_rank_pattern": {key: int(value) for key, value in search_pattern.items()},
    }


def save_pattern_files(patterns: dict[str, Any], output_dir: str | Path, prefix: str = "best") -> dict[str, str]:
    """分别写出 rank_pattern、alpha_pattern、exclude_modules 和完整 manifest。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = {
        "rank_pattern_path": output_path / f"{prefix}_rank_pattern.json",
        "alpha_pattern_path": output_path / f"{prefix}_alpha_pattern.json",
        "exclude_modules_path": output_path / f"{prefix}_exclude_modules.json",
        "manifest_path": output_path / f"{prefix}_peft_patterns.json",
    }
    write_json(patterns["rank_pattern"], files["rank_pattern_path"])
    write_json(patterns["alpha_pattern"], files["alpha_pattern_path"])
    write_json(patterns["exclude_modules"], files["exclude_modules_path"])
    write_json(patterns, files["manifest_path"])
    return {key: str(value) for key, value in files.items()}


def write_json(data: Any, path: str | Path) -> None:
    """以稳定格式写 JSON，便于实验记录 diff 和论文后处理。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def estimate_lora_trainable_params(
    search_pattern: dict[str, int],
    hidden_size: int,
    intermediate_size: int,
) -> int:
    """估算 LoRA 可训练参数量，作为 PSO 参数预算惩罚项。"""
    total = 0
    for key, rank in search_pattern.items():
        rank = int(rank)
        if rank <= 0:
            continue
        in_features, out_features = infer_linear_shape(key, hidden_size, intermediate_size)
        total += rank * (in_features + out_features)
    return total


def infer_linear_shape(key: str, hidden_size: int, intermediate_size: int) -> tuple[int, int]:
    """按 Qwen/LLaMA 常见线性层形状估算 LoRA 参数量。"""
    module_name = key.rsplit(".", 1)[-1]
    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return hidden_size, hidden_size
    if module_name in {"gate_proj", "up_proj"}:
        return hidden_size, intermediate_size
    if module_name == "down_proj":
        return intermediate_size, hidden_size
    return hidden_size, hidden_size


def max_lora_trainable_params(
    keys: list[str],
    max_rank: int,
    hidden_size: int,
    intermediate_size: int,
) -> int:
    """估算所有层取最大 rank 时的 LoRA 参数量，用于归一化。"""
    return estimate_lora_trainable_params(
        {key: max_rank for key in keys},
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def pattern_rank_budget(search_pattern: dict[str, int]) -> int:
    """返回 rank 总预算，便于快速筛选粒子。"""
    return sum(max(0, int(rank)) for rank in search_pattern.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 importance 构造 PEFT rank_pattern/alpha_pattern")
    parser.add_argument("--config", type=str, required=True, help="PSO/LoRA YAML 配置")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    search_config = config.get("search", {})
    lora_config = config.get("lora", {})
    model_config = config.get("model", {})
    output_dir = args.output_dir or config.get("output", {}).get("run_dir", "experiments/rank_search/ours_pso")

    importance_path = resolve_importance_path(search_config.get("importance_path"))
    importance = load_importance(importance_path) if importance_path else {}
    candidate_ranks = [int(rank) for rank in search_config.get("candidate_ranks", [0, 2, 4, 8, 16, 32])]
    keys = build_rank_keys(
        num_hidden_layers=int(model_config.get("num_hidden_layers", 28)),
        target_modules=list(lora_config.get("target_modules", ["q_proj", "v_proj"])),
        importance=importance,
    )
    pattern = importance_seed_pattern(
        keys=keys,
        importance=importance,
        candidate_ranks=candidate_ranks,
        default_rank=int(lora_config.get("r", 8)),
    )
    peft_patterns = build_peft_patterns(
        pattern,
        alpha_multiplier=float(search_config.get("alpha_multiplier", 2.0)),
    )
    files = save_pattern_files(peft_patterns, output_dir, prefix="importance_seed")
    print(json.dumps(files, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

