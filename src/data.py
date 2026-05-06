"""LoRA baseline 的数据读取与监督微调样本构造。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SYSTEM_PROMPT = "你是一个严谨、可靠的中文助手。"


@dataclass
class SupervisedExample:
    """统一后的监督微调样本格式。"""

    instruction: str
    input: str
    output: str
    system: str = DEFAULT_SYSTEM_PROMPT


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，每行一个 JSON 对象。"""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON。") from exc
    return records


def normalize_record(record: dict[str, Any], data_format: str = "auto") -> SupervisedExample:
    """支持 instruction JSONL 与 GSM8K question-answer 两类常见格式。"""
    if data_format == "auto":
        data_format = "gsm8k" if "question" in record and "answer" in record else "instruction"

    if data_format == "gsm8k":
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"GSM8K 样本缺少 question/answer：{record}")
        return SupervisedExample(
            system="你是一个擅长数学推理的助手。",
            instruction="请解答下面的数学题，并给出必要的推理过程。",
            input=question,
            output=answer,
        )

    instruction = str(record.get("instruction") or record.get("prompt") or record.get("question") or "").strip()
    input_text = str(record.get("input") or record.get("context") or "").strip()
    output_text = str(record.get("output") or record.get("response") or record.get("answer") or "").strip()
    system = str(record.get("system") or DEFAULT_SYSTEM_PROMPT).strip()
    if not instruction or not output_text:
        raise ValueError(f"instruction 样本缺少 instruction/output：{record}")
    return SupervisedExample(instruction=instruction, input=input_text, output=output_text, system=system)


def load_examples(
    path: str | Path,
    data_format: str = "auto",
    max_samples: int | None = None,
) -> list[SupervisedExample]:
    """从 JSONL 加载并归一化样本，可限制样本数用于快速调试。"""
    records = read_jsonl(path)
    examples = [normalize_record(record, data_format=data_format) for record in records]
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def split_train_eval(
    examples: list[SupervisedExample],
    eval_ratio: float,
    seed: int,
) -> tuple[list[SupervisedExample], list[SupervisedExample]]:
    """按比例切分训练集和验证集，随机种子保证实验可复现。"""
    if eval_ratio <= 0:
        return examples, []
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio 必须位于 (0, 1) 区间。")
    if len(examples) < 2:
        return examples, []
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_size = min(max(1, int(len(shuffled) * eval_ratio)), len(shuffled) - 1)
    return shuffled[eval_size:], shuffled[:eval_size]


def render_prompt(example: SupervisedExample, tokenizer: Any | None = None, with_answer: bool = True) -> str:
    """优先使用模型 chat_template，确保 Qwen Instruct 的训练格式贴近推理格式。"""
    user_content = example.instruction if not example.input else f"{example.instruction}\n\n{example.input}"
    messages = [
        {"role": "system", "content": example.system},
        {"role": "user", "content": user_content},
    ]
    if with_answer:
        messages.append({"role": "assistant", "content": example.output})

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not with_answer,
        )

    answer = example.output if with_answer else ""
    return (
        f"### 系统\n{example.system}\n\n"
        f"### 指令\n{example.instruction}\n\n"
        f"### 输入\n{example.input}\n\n"
        f"### 回答\n{answer}"
    )


def tokenize_examples(
    examples: Iterable[SupervisedExample],
    tokenizer: Any,
    max_length: int,
    train_on_prompt: bool,
) -> list[dict[str, list[int]]]:
    """构造 causal LM 训练特征；默认只对回答部分计算 loss。"""
    tokenized: list[dict[str, list[int]]] = []
    eos = tokenizer.eos_token or ""

    for example in examples:
        full_text = render_prompt(example, tokenizer=tokenizer, with_answer=True) + eos
        encoded = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
        labels = list(encoded["input_ids"])

        if not train_on_prompt:
            prompt_text = render_prompt(example, tokenizer=tokenizer, with_answer=False)
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len

        encoded["labels"] = labels
        tokenized.append(encoded)

    return tokenized


class DataCollatorForCausalLM:
    """动态 padding，并保持 labels 中 -100 的忽略标记。"""

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
