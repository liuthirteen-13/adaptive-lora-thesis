"""粒子群搜索 layer-wise rank_pattern 的基础结构。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Particle:
    """一个粒子表示一组 layer-wise rank 取值。"""

    position: list[int]
    velocity: list[float]
    best_position: list[int] = field(default_factory=list)
    best_score: float = float("-inf")

    def __post_init__(self) -> None:
        if not self.best_position:
            self.best_position = list(self.position)


def build_qwen_rank_keys(num_hidden_layers: int, target_modules: list[str]) -> list[str]:
    """生成 Qwen 类模型常见的 PEFT rank_pattern key。"""
    keys: list[str] = []
    for layer_idx in range(num_hidden_layers):
        for module_name in target_modules:
            keys.append(f"model.layers.{layer_idx}.{module_name}")
    return keys


def initialize_particles(
    num_particles: int,
    dimension: int,
    min_rank: int,
    max_rank: int,
    rank_step: int,
    seed: int,
) -> list[Particle]:
    """初始化粒子位置和速度，rank 会落在离散候选集合中。"""
    rng = random.Random(seed)
    ranks = list(range(min_rank, max_rank + 1, rank_step))
    particles: list[Particle] = []
    for _ in range(num_particles):
        position = [rng.choice(ranks) for _ in range(dimension)]
        velocity = [rng.uniform(-rank_step, rank_step) for _ in range(dimension)]
        particles.append(Particle(position=position, velocity=velocity))
    return particles


def position_to_rank_pattern(keys: list[str], position: list[int]) -> dict[str, int]:
    """把粒子位置转换成 PEFT rank_pattern 字典。"""
    if len(keys) != len(position):
        raise ValueError("keys 与 position 维度不一致")
    return dict(zip(keys, position, strict=True))


def save_rank_pattern(pattern: dict[str, int], path: str | Path) -> None:
    """保存 rank_pattern，供 train_lora.py 通过 --rank-pattern 使用。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(pattern, file, ensure_ascii=False, indent=2)
        file.write("\n")


def estimate_rank_budget(pattern: dict[str, int]) -> int:
    """简单估计 rank 总预算，后续可替换为真实可训练参数量估计。"""
    return sum(pattern.values())

