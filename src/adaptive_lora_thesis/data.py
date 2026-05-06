"""数据读取、prompt 渲染与 causal LM 批处理工具。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable


PROMPT_TEMPLATE = """{system_block}### 指令
{instruction}

### 输入
{input_text}

### 回答
{output_text}"""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，每行是一个样本。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """写出统一格式 JSONL。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_instruction_record(record: dict[str, Any]) -> dict[str, str]:
    """把常见 instruction 字段归一化，便于训练脚本统一处理。"""
    instruction = record.get("instruction") or record.get("prompt") or record.get("question") or ""
    input_text = record.get("input") or record.get("context") or ""
    output_text = record.get("output") or record.get("answer") or record.get("response") or ""
    system = record.get("system") or "你是一个严谨、可靠的中文助手。"
    if not instruction and input_text:
        instruction, input_text = input_text, ""
    if not instruction or not output_text:
        raise ValueError(f"样本缺少 instruction/output 字段：{record}")
    return {
        "system": str(system),
        "instruction": str(instruction),
        "input": str(input_text),
        "output": str(output_text),
    }


def normalize_gsm8k_record(record: dict[str, Any]) -> dict[str, str]:
    """将 GSM8K 的 question/answer 转为 instruction JSONL。"""
    question = str(record["question"])
    answer = str(record["answer"])
    return {
        "system": "你是一个擅长数学推理的助手。",
        "instruction": "请解答下面的数学题，并给出必要推理过程。",
        "input": question,
        "output": answer,
    }


def render_record(record: dict[str, str], tokenizer: Any | None = None, include_answer: bool = True) -> str:
    """优先使用 tokenizer chat_template，否则回退到固定 instruction 模板。"""
    output_text = record["output"] if include_answer else ""
    messages = [
        {"role": "system", "content": record.get("system", "")},
        {"role": "user", "content": _join_instruction_and_input(record)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": output_text})

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        # Qwen Instruct 模型通常提供 chat_template，用它可减少训练/推理格式偏差。
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )

    system_block = f"### 系统\n{record.get('system', '')}\n\n" if record.get("system") else ""
    return PROMPT_TEMPLATE.format(
        system_block=system_block,
        instruction=record["instruction"],
        input_text=record.get("input", ""),
        output_text=output_text,
    )


def split_train_eval(
    records: list[dict[str, str]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """按比例切分训练/验证集，保证随机种子可复现。"""
    if not 0 <= eval_ratio < 1:
        raise ValueError("eval_ratio 必须位于 [0, 1) 区间")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    eval_size = int(len(shuffled) * eval_ratio)
    if eval_size == 0:
        return shuffled, []
    return shuffled[eval_size:], shuffled[:eval_size]


def tokenize_records(
    records: list[dict[str, str]],
    tokenizer: Any,
    max_seq_length: int,
    train_on_prompt: bool,
) -> list[dict[str, list[int]]]:
    """把文本样本转成 Trainer 可用的 input_ids/attention_mask/labels。"""
    tokenized: list[dict[str, list[int]]] = []
    eos_token = tokenizer.eos_token or ""
    for record in records:
        full_text = render_record(record, tokenizer=tokenizer, include_answer=True) + eos_token
        encoded = tokenizer(full_text, truncation=True, max_length=max_seq_length, padding=False)
        labels = list(encoded["input_ids"])

        if not train_on_prompt:
            # 只在回答部分计算 loss，避免模型过度学习模板和题面复述。
            prompt_text = render_record(record, tokenizer=tokenizer, include_answer=False)
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_seq_length, padding=False)["input_ids"]
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

        encoded["labels"] = labels
        tokenized.append(encoded)
    return tokenized


class CausalLMPaddingCollator:
    """为 causal LM 动态 padding，同时保持 labels 的 -100 掩码。"""

    def __init__(self, tokenizer: Any, label_pad_token_id: int = -100) -> None:
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padding = [self.label_pad_token_id] * (max_length - len(label))
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                padded_labels.append(padding + label)
            else:
                padded_labels.append(label + padding)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def _join_instruction_and_input(record: dict[str, str]) -> str:
    """合并 instruction 和 input，作为 chat_template 的 user 内容。"""
    if record.get("input"):
        return f"{record['instruction']}\n\n{record['input']}"
    return record["instruction"]
