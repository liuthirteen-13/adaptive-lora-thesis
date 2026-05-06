from __future__ import annotations

import shutil
from pathlib import Path

from rank_search.build_rank_pattern import (
    build_peft_patterns,
    build_rank_keys,
    estimate_lora_trainable_params,
    importance_seed_pattern,
)
from rank_search.fitness import evaluate_trial
from rank_search.pso import initialize_particles, position_to_pattern


def test_build_peft_patterns_excludes_zero_rank() -> None:
    search_pattern = {
        "model.layers.0.self_attn.q_proj": 0,
        "model.layers.0.self_attn.v_proj": 8,
    }

    patterns = build_peft_patterns(search_pattern, alpha_multiplier=2.0)

    assert "model.layers.0.self_attn.q_proj" not in patterns["rank_pattern"]
    assert patterns["exclude_modules"] == ["model.layers.0.self_attn.q_proj"]
    assert patterns["rank_pattern"]["model.layers.0.self_attn.v_proj"] == 8
    assert patterns["alpha_pattern"]["model.layers.0.self_attn.v_proj"] == 16


def test_importance_seed_and_param_estimate() -> None:
    importance = {
        "model.layers.0.self_attn.q_proj": {"grad_norm": 0.1, "suggested_rank": 6},
        "model.layers.0.self_attn.v_proj": {"grad_norm": 0.2, "suggested_rank": 15},
    }
    keys = build_rank_keys(1, ["q_proj", "v_proj"], importance)
    pattern = importance_seed_pattern(keys, importance, [0, 2, 4, 8, 16, 32], default_rank=8)

    assert pattern["model.layers.0.self_attn.q_proj"] == 4
    assert pattern["model.layers.0.self_attn.v_proj"] == 16
    assert estimate_lora_trainable_params(pattern, hidden_size=4, intermediate_size=8) == 160


def test_proxy_fitness_records_trial() -> None:
    output_dir = Path(".pytest_cache") / "rank_search_proxy"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    config = {
        "model": {"hidden_size": 4, "intermediate_size": 8},
        "lora": {"r": 8, "target_modules": ["q_proj", "v_proj"]},
        "search": {"candidate_ranks": [0, 2, 4, 8], "alpha_multiplier": 2.0},
        "fitness": {"mode": "proxy", "lambda_param": 0.1, "lambda_time": 0.0, "param_budget": 128},
        "training": {"seed": 42},
        "data": {"train_file": "data/processed/instruction_train.jsonl"},
    }
    pattern = {
        "model.layers.0.self_attn.q_proj": 0,
        "model.layers.0.self_attn.v_proj": 8,
    }

    result = evaluate_trial(
        search_pattern=pattern,
        config=config,
        trial_dir=output_dir,
        trial_id="trial_proxy",
        seed=42,
        importance={"model.layers.0.self_attn.v_proj": {"grad_norm": 1.0}},
    )

    assert result.status == "success"
    assert (output_dir / "trial_record.json").exists()
    assert (output_dir / "trial_rank_pattern.json").exists()


def test_initialize_particles_is_deterministic() -> None:
    keys = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"]
    importance = {keys[0]: {"suggested_rank": 8}, keys[1]: {"suggested_rank": 2}}
    particles_a = initialize_particles(2, 2, [0, 2, 4, 8], 42, keys, importance, default_rank=4)
    particles_b = initialize_particles(2, 2, [0, 2, 4, 8], 42, keys, importance, default_rank=4)

    assert particles_a[0].position == particles_b[0].position
    assert position_to_pattern(keys, particles_a[0].position, [0, 2, 4, 8]) == {
        keys[0]: 8,
        keys[1]: 2,
    }

