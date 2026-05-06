"""基于粒子群优化的 LoRA layer-wise rank_pattern 搜索入口。

完成一次搜索后会在 output.run_dir 下生成：
- best_rank_pattern.json：PEFT LoraConfig 可用的正 rank pattern；
- best_alpha_pattern.json：与 rank_pattern 对应的 alpha pattern；
- best_exclude_modules.json：搜索中 rank=0 的层，用 PEFT exclude_modules 表达；
- best_lora_config.yaml：最终训练可直接使用的 LoRA 配置。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rank_search.build_rank_pattern import (  # noqa: E402
    build_peft_patterns,
    build_rank_keys,
    importance_seed_pattern,
    load_importance,
    nearest_candidate_rank,
    resolve_importance_path,
    save_pattern_files,
    write_json,
)
from rank_search.fitness import FitnessResult, evaluate_trial  # noqa: E402


@dataclass
class Particle:
    """一个粒子表示一组 layer-wise rank 候选索引。"""

    position: list[int]
    velocity: list[float]
    best_position: list[int] = field(default_factory=list)
    best_score: float = -1.0e30

    def __post_init__(self) -> None:
        if not self.best_position:
            self.best_position = list(self.position)


@dataclass
class SearchState:
    """可恢复 PSO 搜索状态。"""

    seed: int
    iteration: int
    particle_index: int
    particles: list[Particle]
    global_best_position: list[int]
    global_best_score: float
    completed_trials: list[str] = field(default_factory=list)
    failed_trials: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSO 搜索 LoRA layer-wise rank_pattern")
    parser.add_argument("--config", type=str, required=True, help="PSO YAML 配置文件")
    parser.add_argument("--run", action="store_true", help="显式关闭 dry-run 并启动真实搜索")
    parser.add_argument("--dry-run", action="store_true", help="只检查搜索空间，不训练/评估 trial")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="是否从 state.json 恢复")
    parser.add_argument("--force-trial", action="store_true", help="即使 trial_record 已存在也重新评估")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolve_run_mode(config, args)
    output_dir = Path(config.get("output", {}).get("run_dir", "experiments/rank_search/ours_pso"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_search_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    importance_path = resolve_importance_path(config.get("search", {}).get("importance_path"))
    importance = load_importance(importance_path) if importance_path else {}
    keys = build_search_keys(config, importance)
    candidate_ranks = [int(rank) for rank in config.get("search", {}).get("candidate_ranks", [0, 2, 4, 8, 16, 32])]
    validate_candidate_ranks(candidate_ranks)

    print_search_summary(config, output_dir, importance_path, keys, candidate_ranks)
    if config.get("run", {}).get("dry_run", True):
        dry_run_preview(config, output_dir, keys, candidate_ranks, importance)
        return

    state_path = output_dir / "state.json"
    state = load_or_initialize_state(
        state_path=state_path,
        config=config,
        keys=keys,
        candidate_ranks=candidate_ranks,
        importance=importance,
        resume=bool(args.resume),
    )
    run_search(
        config=config,
        output_dir=output_dir,
        state_path=state_path,
        state=state,
        keys=keys,
        candidate_ranks=candidate_ranks,
        importance=importance,
        force_trial=bool(args.force_trial),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    """读取搜索配置。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_run_mode(config: dict[str, Any], args: argparse.Namespace) -> None:
    """搜索脚本默认受 dry-run 保护，只有 --run 会启动 trial 训练/评估。"""
    run_config = config.setdefault("run", {})
    if args.dry_run:
        run_config["dry_run"] = True
    elif args.run:
        run_config["dry_run"] = False
    else:
        run_config["dry_run"] = bool(run_config.get("dry_run", True))


def build_search_keys(config: dict[str, Any], importance: dict[str, dict[str, float | int]]) -> list[str]:
    """从 importance 或模型结构配置生成搜索维度。"""
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    return build_rank_keys(
        num_hidden_layers=int(model_config.get("num_hidden_layers", 28)),
        target_modules=list(lora_config.get("target_modules", ["q_proj", "v_proj"])),
        importance=importance,
    )


def validate_candidate_ranks(candidate_ranks: list[int]) -> None:
    """检查候选 rank 集合，确保 0 rank 和正 rank 都可被表达。"""
    if not candidate_ranks:
        raise ValueError("search.candidate_ranks 不能为空")
    if any(rank < 0 for rank in candidate_ranks):
        raise ValueError("search.candidate_ranks 不能包含负数")
    if not any(rank > 0 for rank in candidate_ranks):
        raise ValueError("search.candidate_ranks 至少需要一个正 rank，供 PEFT LoraConfig 使用")


def print_search_summary(
    config: dict[str, Any],
    output_dir: Path,
    importance_path: Path | None,
    keys: list[str],
    candidate_ranks: list[int],
) -> None:
    """打印搜索空间摘要。"""
    print("\n===== PSO LoRA rank 搜索配置 =====")
    print(f"输出目录：{output_dir}")
    print(f"importance：{importance_path or '未使用'}")
    print(f"搜索维度：{len(keys)}")
    print(f"候选 rank：{candidate_ranks}")
    print(f"fitness.mode：{config.get('fitness', {}).get('mode', 'train_eval')}")
    print(f"dry_run：{config.get('run', {}).get('dry_run', True)}")


def dry_run_preview(
    config: dict[str, Any],
    output_dir: Path,
    keys: list[str],
    candidate_ranks: list[int],
    importance: dict[str, dict[str, float | int]],
) -> None:
    """dry-run 只生成一个 importance seed pattern 预览，不启动训练。"""
    lora_config = config.get("lora", {})
    pattern = importance_seed_pattern(
        keys=keys,
        importance=importance,
        candidate_ranks=candidate_ranks,
        default_rank=int(lora_config.get("r", 8)),
    )
    peft_patterns = build_peft_patterns(
        pattern,
        alpha_multiplier=float(config.get("search", {}).get("alpha_multiplier", 2.0)),
    )
    preview_dir = output_dir / "dry_run_preview"
    files = save_pattern_files(peft_patterns, preview_dir, prefix="importance_seed")
    print("dry-run 完成：未训练模型，已写出 importance seed 预览。")
    print(json.dumps(files, ensure_ascii=False, indent=2))
    print("真实启动命令：python src/rank_search/pso.py --config configs/ours_pso_rank_lora.yaml --run")


def load_or_initialize_state(
    state_path: Path,
    config: dict[str, Any],
    keys: list[str],
    candidate_ranks: list[int],
    importance: dict[str, dict[str, float | int]],
    resume: bool,
) -> SearchState:
    """从 state.json 恢复，或初始化新的粒子群。"""
    if resume and state_path.exists():
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return SearchState(
            seed=int(data["seed"]),
            iteration=int(data["iteration"]),
            particle_index=int(data["particle_index"]),
            particles=[Particle(**particle) for particle in data["particles"]],
            global_best_position=list(data["global_best_position"]),
            global_best_score=float(data["global_best_score"]),
            completed_trials=list(data.get("completed_trials", [])),
            failed_trials=list(data.get("failed_trials", [])),
        )

    search_config = config.get("search", {})
    seed = int(search_config.get("seed", 42))
    particles = initialize_particles(
        num_particles=int(search_config.get("particles", 4)),
        dimension=len(keys),
        candidate_ranks=candidate_ranks,
        seed=seed,
        keys=keys,
        importance=importance,
        default_rank=int(config.get("lora", {}).get("r", 8)),
    )
    return SearchState(
        seed=seed,
        iteration=0,
        particle_index=0,
        particles=particles,
        global_best_position=list(particles[0].position),
        global_best_score=-1.0e30,
    )


def initialize_particles(
    num_particles: int,
    dimension: int,
    candidate_ranks: list[int],
    seed: int,
    keys: list[str],
    importance: dict[str, dict[str, float | int]],
    default_rank: int,
) -> list[Particle]:
    """初始化粒子群；第一个粒子使用 importance suggested_rank，其余粒子随机扰动。"""
    rng = random.Random(seed)
    candidate_ranks = sorted(candidate_ranks)
    rank_to_index = {rank: idx for idx, rank in enumerate(candidate_ranks)}
    particles: list[Particle] = []

    seed_pattern = importance_seed_pattern(keys, importance, candidate_ranks, default_rank=default_rank)
    seed_position = [rank_to_index[seed_pattern[key]] for key in keys]
    particles.append(
        Particle(
            position=seed_position,
            velocity=[0.0 for _ in range(dimension)],
        )
    )

    for _ in range(max(0, num_particles - 1)):
        position = []
        velocity = []
        for key in keys:
            suggested = int(importance.get(key, {}).get("suggested_rank", default_rank))
            center_rank = nearest_candidate_rank(suggested, candidate_ranks)
            center_idx = rank_to_index[center_rank]
            offset = rng.choice([-2, -1, 0, 1, 2])
            idx = min(max(center_idx + offset, 0), len(candidate_ranks) - 1)
            position.append(idx)
            velocity.append(rng.uniform(-1.0, 1.0))
        particles.append(Particle(position=position, velocity=velocity))
    return particles


def run_search(
    config: dict[str, Any],
    output_dir: Path,
    state_path: Path,
    state: SearchState,
    keys: list[str],
    candidate_ranks: list[int],
    importance: dict[str, dict[str, float | int]],
    force_trial: bool,
) -> None:
    """执行可恢复 PSO 搜索。"""
    search_config = config.get("search", {})
    iterations = int(search_config.get("iterations", 2))
    max_trials = int(search_config.get("max_trials", 0) or 0)
    trial_counter = len(state.completed_trials) + len(state.failed_trials)
    rng = random.Random(state.seed + 1009)

    while state.iteration < iterations:
        while state.particle_index < len(state.particles):
            if max_trials and trial_counter >= max_trials:
                save_state(state_path, state)
                write_best_outputs(config, output_dir, state, keys, candidate_ranks)
                print(f"达到 max_trials={max_trials}，提前停止搜索。")
                return

            particle = state.particles[state.particle_index]
            search_pattern = position_to_pattern(keys, particle.position, candidate_ranks)
            trial_id = build_trial_id(state.iteration, state.particle_index, search_pattern)
            trial_dir = output_dir / "trials" / trial_id
            result = evaluate_trial(
                search_pattern=search_pattern,
                config=config,
                trial_dir=trial_dir,
                trial_id=trial_id,
                seed=int(search_config.get("seed", 42)) + trial_counter,
                importance=importance,
                force=force_trial,
            )
            update_bests(state, particle, result)
            if result.status == "success":
                state.completed_trials.append(trial_id)
            else:
                state.failed_trials.append(trial_id)
            write_best_outputs(config, output_dir, state, keys, candidate_ranks)

            update_particle_position(
                particle=particle,
                global_best_position=state.global_best_position,
                candidate_count=len(candidate_ranks),
                rng=rng,
                inertia=float(search_config.get("inertia", 0.7)),
                cognitive=float(search_config.get("cognitive", 1.4)),
                social=float(search_config.get("social", 1.4)),
            )

            state.particle_index += 1
            trial_counter += 1
            save_state(state_path, state)
            print(
                f"trial={trial_id} status={result.status} score={result.score:.6f} "
                f"best={state.global_best_score:.6f}"
            )

        state.iteration += 1
        state.particle_index = 0
        save_state(state_path, state)

    write_best_outputs(config, output_dir, state, keys, candidate_ranks)
    print(f"搜索结束，best_rank_pattern.json 已写入：{output_dir / 'best_rank_pattern.json'}")


def position_to_pattern(keys: list[str], position: list[int], candidate_ranks: list[int]) -> dict[str, int]:
    """把粒子索引位置转换成 rank_pattern。"""
    if len(keys) != len(position):
        raise ValueError("keys 与 particle.position 维度不一致")
    return {key: int(candidate_ranks[index]) for key, index in zip(keys, position, strict=True)}


def build_trial_id(iteration: int, particle_index: int, search_pattern: dict[str, int]) -> str:
    """用 pattern hash 构造稳定 trial id，避免恢复时目录冲突。"""
    payload = json.dumps(search_pattern, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"iter{iteration:03d}_particle{particle_index:03d}_{digest}"


def update_bests(state: SearchState, particle: Particle, result: FitnessResult) -> None:
    """更新粒子历史最好和全局最好。"""
    if result.status != "success":
        return
    if result.score > particle.best_score:
        particle.best_score = result.score
        particle.best_position = list(particle.position)
    if result.score > state.global_best_score:
        state.global_best_score = result.score
        state.global_best_position = list(particle.position)


def update_particle_position(
    particle: Particle,
    global_best_position: list[int],
    candidate_count: int,
    rng: random.Random,
    inertia: float,
    cognitive: float,
    social: float,
) -> None:
    """执行离散 PSO 速度与位置更新。"""
    for dim, current_idx in enumerate(particle.position):
        r1 = rng.random()
        r2 = rng.random()
        personal_delta = particle.best_position[dim] - current_idx
        global_delta = global_best_position[dim] - current_idx
        velocity = inertia * particle.velocity[dim] + cognitive * r1 * personal_delta + social * r2 * global_delta
        next_idx = round(current_idx + velocity)
        next_idx = min(max(next_idx, 0), candidate_count - 1)
        particle.velocity[dim] = velocity
        particle.position[dim] = next_idx


def write_best_outputs(
    config: dict[str, Any],
    output_dir: Path,
    state: SearchState,
    keys: list[str],
    candidate_ranks: list[int],
) -> None:
    """把当前全局最优写成 PEFT 可用文件，便于中断后也能拿到 best。"""
    if not state.global_best_position:
        return
    best_pattern = position_to_pattern(keys, state.global_best_position, candidate_ranks)
    peft_patterns = build_peft_patterns(
        best_pattern,
        alpha_multiplier=float(config.get("search", {}).get("alpha_multiplier", 2.0)),
    )
    files = save_pattern_files(peft_patterns, output_dir, prefix="best")
    write_json(
        {
            "global_best_score": state.global_best_score,
            "pattern_files": files,
            "search_rank_pattern": best_pattern,
        },
        output_dir / "best_summary.json",
    )
    write_best_lora_config(config, output_dir, files)


def write_best_lora_config(config: dict[str, Any], output_dir: Path, pattern_files: dict[str, str]) -> None:
    """生成最终训练可直接使用的 YAML 配置。"""
    best_config = yaml.safe_load(yaml.safe_dump(config, allow_unicode=True)) or {}
    best_config.setdefault("lora", {})
    best_config["lora"]["rank_pattern_path"] = pattern_files["rank_pattern_path"]
    best_config["lora"]["alpha_pattern_path"] = pattern_files["alpha_pattern_path"]
    best_config["lora"]["exclude_modules_path"] = pattern_files["exclude_modules_path"]
    best_config.setdefault("training", {})
    best_config["training"]["output_dir"] = str(output_dir / "final_lora_adapter")
    best_config["training"]["max_steps"] = -1
    best_config.setdefault("data", {})
    # 搜索阶段使用小子集；最终训练配置默认恢复为完整训练数据，可再通过 CLI 覆盖。
    best_config["data"]["max_samples"] = None
    best_config["data"]["max_eval_samples"] = None
    final_overrides = config.get("final_training", {})
    for section_name in ("data", "training"):
        if isinstance(final_overrides.get(section_name), dict):
            best_config.setdefault(section_name, {})
            best_config[section_name].update(final_overrides[section_name])
    best_config.pop("fitness", None)
    best_config.pop("search", None)
    best_config.pop("output", None)
    best_config.pop("run", None)
    best_config.pop("final_training", None)
    (output_dir / "best_lora_config.yaml").write_text(
        yaml.safe_dump(best_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def save_state(state_path: Path, state: SearchState) -> None:
    """保存可恢复状态。"""
    data = asdict(state)
    write_json(data, state_path)


if __name__ == "__main__":
    main()
