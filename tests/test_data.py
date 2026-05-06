from adaptive_lora_thesis.data import normalize_instruction_record, split_train_eval


def test_normalize_instruction_record() -> None:
    record = normalize_instruction_record({"instruction": "问题", "output": "答案"})
    assert record["instruction"] == "问题"
    assert record["output"] == "答案"
    assert "system" in record


def test_split_train_eval_is_deterministic() -> None:
    records = [{"instruction": str(i), "input": "", "output": str(i)} for i in range(10)]
    train_a, eval_a = split_train_eval(records, eval_ratio=0.2, seed=42)
    train_b, eval_b = split_train_eval(records, eval_ratio=0.2, seed=42)
    assert train_a == train_b
    assert eval_a == eval_b
    assert len(eval_a) == 2

