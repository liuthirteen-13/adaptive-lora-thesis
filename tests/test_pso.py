from adaptive_lora_thesis.pso import build_qwen_rank_keys, estimate_rank_budget, position_to_rank_pattern


def test_rank_pattern_keys() -> None:
    keys = build_qwen_rank_keys(2, ["self_attn.q_proj", "self_attn.v_proj"])
    pattern = position_to_rank_pattern(keys, [4, 8, 12, 16])
    assert pattern["model.layers.0.self_attn.q_proj"] == 4
    assert pattern["model.layers.1.self_attn.v_proj"] == 16
    assert estimate_rank_budget(pattern) == 40

