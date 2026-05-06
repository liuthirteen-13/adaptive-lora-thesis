"""LoRA/QLoRA 配置构造工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rank_pattern(path: str | Path | None) -> dict[str, int] | None:
    """读取 PEFT LoraConfig 可用的 rank_pattern JSON。"""
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as file:
        pattern = json.load(file)
    return {str(key): int(value) for key, value in pattern.items()}


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    """把配置中的 dtype 字符串转换为 torch dtype。"""
    if dtype_name in (None, "auto"):
        return "auto"
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"不支持的 torch_dtype：{dtype_name}")
    return mapping[dtype_name]


def build_quantization_config(config: dict[str, Any] | None, enabled: bool) -> Any | None:
    """构造 bitsandbytes 4-bit 量化配置，仅在 QLoRA 启用时导入依赖。"""
    if not enabled:
        return None
    if not config or not config.get("load_in_4bit", True):
        return None

    import torch
    from transformers import BitsAndBytesConfig

    dtype_name = config.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype_name]
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
    )


def build_lora_config(config: dict[str, Any], rank_pattern: dict[str, int] | None = None) -> Any:
    """根据 YAML 配置构造 PEFT LoraConfig。"""
    from peft import LoraConfig, TaskType

    target_modules = config.get("target_modules", ["q_proj", "v_proj"])
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config.get("r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("bias", "none")),
        target_modules=target_modules,
        rank_pattern=rank_pattern,
    )


def prepare_model_for_peft(model: Any, qlora: bool) -> Any:
    """在 QLoRA 模式下开启 k-bit 训练准备，然后注入 LoRA adapter。"""
    if qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return model

