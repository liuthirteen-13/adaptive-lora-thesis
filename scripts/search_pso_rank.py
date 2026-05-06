"""PSO layer-wise rank_pattern 搜索入口。

当前实现用于生成和记录候选 rank_pattern，真实训练/评估目标函数后续接入。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_lora_thesis.config import deep_update, load_yaml, print_config
from adaptive_lora_thesis.pso import (
    build_qwen_rank_keys,
    estimate_rank_budget,
    initialize_particles,
    position_to_rank_pattern,
    save_rank_pattern,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="粒子群搜索 LoRA layer-wise rank_pattern")
    parser.add_argument("--config", type=str, default="configs/pso_search.yaml", help="YAML 配置文件")
    parser.add_argument("--output-dir", type=str, help="rank_pattern 输出目录")
    parser.add_argument("--particles", type=int, help="粒子数量")
    parser.add_argument("--iterations", type=int, help="迭代轮数")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--emit-initial-patterns", action="store_true", help="写出初始粒子的 rank_pattern")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None, help="只检查搜索空间")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    print_config("PSO 搜索配置", config)

    search_config = config["search"]
    keys = build_qwen_rank_keys(
        num_hidden_layers=int(config["model"].get("num_hidden_layers", 28)),
        target_modules=list(search_config["target_modules"]),
    )
    particles = initialize_particles(
        num_particles=int(search_config.get("particles", 8)),
        dimension=len(keys),
        min_rank=int(search_config.get("min_rank", 2)),
        max_rank=int(search_config.get("max_rank", 32)),
        rank_step=int(search_config.get("rank_step", 2)),
        seed=int(search_config.get("seed", 42)),
    )
    print(f"搜索维度：{len(keys)}，初始粒子数：{len(particles)}")

    if config["run"].get("dry_run", True):
        first_pattern = position_to_rank_pattern(keys, particles[0].position)
        preview = dict(list(first_pattern.items())[:8])
        print("当前为 dry-run：未启动训练/评估，仅展示首个粒子前 8 个 rank。")
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    if args.emit_initial_patterns:
        output_dir = Path(config["output"]["rank_pattern_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for idx, particle in enumerate(particles):
            pattern = position_to_rank_pattern(keys, particle.position)
            pattern_path = output_dir / f"candidate_{idx:03d}.json"
            save_rank_pattern(pattern, pattern_path)
            manifest.append(
                {
                    "candidate": idx,
                    "path": str(pattern_path),
                    "rank_budget": estimate_rank_budget(pattern),
                }
            )
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"已写出 {len(manifest)} 个初始 rank_pattern 到 {output_dir}")
        print(f"清单文件：{manifest_path}")
    else:
        print("未指定 --emit-initial-patterns，当前版本不会自动启动真实 PSO 训练。")


def build_runtime_config(args: argparse.Namespace) -> dict:
    """合并 YAML 配置与命令行覆盖。"""
    config = load_yaml(args.config)
    override = {
        "search": {
            "particles": args.particles,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "output": {"rank_pattern_dir": args.output_dir},
        "run": {"dry_run": args.dry_run},
    }
    return deep_update(config, override)


if __name__ == "__main__":
    main()

