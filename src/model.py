"""Qwen2.5 LoRA baseline 的模型与 PEFT 构造函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_cuda_available() -> None:
    """普通 LoRA baseline 默认面向 GPU；没有 GPU 时给出友好提示。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("未安装 PyTorch。请先执行 pip install -r requirements.txt。") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到可用 GPU，已停止训练。Qwen2.5-1.5B LoRA baseline 建议在 CUDA GPU 上运行；"
            "请确认已安装支持 CUDA 的 PyTorch，并在有 GPU 的环境中重新执行。"
        )


def ensure_bitsandbytes_available() -> None:
    """检查 QLoRA 所需的 CUDA 与 bitsandbytes 环境，并给出可操作的切换提示。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "未安装 PyTorch。请先执行 pip install -r requirements.txt；"
            "如果当前环境不能运行 QLoRA，可切换普通 LoRA：python src/train_lora.py --config configs/lora_r8.yaml。"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA baseline 需要 CUDA GPU 与 bitsandbytes 4-bit kernel，当前未检测到可用 GPU。"
            "如需继续实验，请切换到支持 CUDA 的 Linux/Windows 环境并安装 requirements-qlora.txt；"
            "如果当前机器只适合普通微调，请改用：python src/train_lora.py --config configs/lora_r8.yaml。"
        )

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "未能导入 bitsandbytes，无法进行 4-bit QLoRA 训练。"
            "请先执行 pip install -r requirements-qlora.txt，并确认 CUDA、PyTorch 与 bitsandbytes wheel 版本匹配；"
            "若当前系统暂不支持 bitsandbytes，请切换普通 LoRA：python src/train_lora.py --config configs/lora_r8.yaml。"
        ) from exc

    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "当前 transformers 版本缺少 BitsAndBytesConfig，无法构造 4-bit 量化配置。"
            "请升级 transformers，或切换普通 LoRA：python src/train_lora.py --config configs/lora_r8.yaml。"
        ) from exc


def resolve_torch_dtype(dtype_name: str | None) -> Any:
    """将配置中的 dtype 字符串转换为 torch dtype。"""
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


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = True) -> Any:
    """加载 tokenizer，并补齐 pad_token，避免 batch padding 报错。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_config: dict[str, Any], for_training: bool = True) -> Any:
    """加载 causal LM 基座模型。

    普通 LoRA 训练不使用 device_map="auto"，因为 auto 在小显存机器上会把部分层
    offload 到 CPU/meta device，反向传播时容易触发 meta/cuda 梯度设备不一致。
    """
    from transformers import AutoModelForCausalLM

    device_map = model_config.get("device_map")
    load_kwargs = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "dtype": resolve_torch_dtype(model_config.get("torch_dtype", "auto")),
    }
    if device_map not in (None, "", "none", "null"):
        if for_training and str(device_map).lower() == "auto":
            print("检测到 device_map=auto；普通 LoRA 训练将改为单卡加载，避免 CPU/meta offload。")
        else:
            load_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_config["name_or_path"], **load_kwargs)
    assert_no_meta_parameters(model)
    return model


def build_4bit_quantization_config(quantization_config: dict[str, Any] | None) -> Any:
    """按 QLoRA baseline 需要构造 bitsandbytes 4-bit 量化配置。"""
    from transformers import BitsAndBytesConfig

    quantization_config = quantization_config or {}
    compute_dtype = resolve_torch_dtype(quantization_config.get("bnb_4bit_compute_dtype", "bfloat16"))
    if compute_dtype == "auto":
        compute_dtype = resolve_torch_dtype("bfloat16")

    return BitsAndBytesConfig(
        load_in_4bit=bool(quantization_config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(quantization_config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization_config.get("bnb_4bit_use_double_quant", True)),
    )


def load_4bit_base_model(model_config: dict[str, Any], quantization_config: dict[str, Any] | None) -> Any:
    """使用 bitsandbytes 4-bit 配置加载 causal LM，供 QLoRA 只训练 adapter。"""
    from transformers import AutoModelForCausalLM

    device_map = model_config.get("device_map", "auto")
    load_kwargs = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "quantization_config": build_4bit_quantization_config(quantization_config),
        "low_cpu_mem_usage": bool(model_config.get("low_cpu_mem_usage", True)),
    }
    torch_dtype = resolve_torch_dtype(model_config.get("torch_dtype", "auto"))
    if torch_dtype != "auto":
        load_kwargs["torch_dtype"] = torch_dtype
    if device_map not in (None, "", "none", "null"):
        load_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_config["name_or_path"], **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "4-bit 量化加载基座模型失败。请确认已安装可用的 bitsandbytes、CUDA 版 PyTorch，"
            "且当前显卡/系统支持 4-bit 量化；如果该环境不支持 QLoRA，请切换普通 LoRA："
            "python src/train_lora.py --config configs/lora_r8.yaml。"
        ) from exc

    assert_no_meta_parameters(model)
    return model


def assert_no_meta_parameters(model: Any) -> None:
    """训练前禁止 meta 参数残留，提前给出可读错误。"""
    meta_names = [name for name, parameter in model.named_parameters() if parameter.device.type == "meta"]
    if meta_names:
        preview = ", ".join(meta_names[:5])
        raise RuntimeError(
            "模型中仍存在 meta device 参数，普通 LoRA 训练无法反向传播。"
            f"示例参数：{preview}。请关闭 device_map=auto/offload，或改用 QLoRA/更小模型。"
        )


def build_lora_config(lora_config: dict[str, Any]) -> Any:
    """按默认 r=8/alpha=16/dropout=0.05 构造 PEFT LoRA 配置。"""
    from peft import LoraConfig, TaskType

    kwargs = {
        "task_type": TaskType.CAUSAL_LM,
        "r": int(lora_config.get("r", 8)),
        "lora_alpha": int(lora_config.get("lora_alpha", 16)),
        "lora_dropout": float(lora_config.get("lora_dropout", 0.05)),
        "target_modules": lora_config.get("target_modules", ["q_proj", "v_proj"]),
        "bias": str(lora_config.get("bias", "none")),
    }
    rank_pattern = lora_config.get("rank_pattern") or load_json_mapping(lora_config.get("rank_pattern_path"))
    if rank_pattern:
        kwargs["rank_pattern"] = rank_pattern
    alpha_pattern = lora_config.get("alpha_pattern") or load_json_mapping(lora_config.get("alpha_pattern_path"))
    if alpha_pattern:
        kwargs["alpha_pattern"] = alpha_pattern
    exclude_modules = lora_config.get("exclude_modules") or load_json_list(lora_config.get("exclude_modules_path"))
    if exclude_modules:
        # rank=0 的层通过 PEFT exclude_modules 跳过注入，避免给 LoraConfig 写入非法 0 rank。
        kwargs["exclude_modules"] = exclude_modules
    return LoraConfig(**kwargs)


def load_json_mapping(path: str | Path | None) -> dict[str, int] | None:
    """读取 rank_pattern/alpha_pattern JSON，保持 PEFT 可直接消费的 dict 结构。"""
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {str(key): int(value) for key, value in data.items()}


def load_json_list(path: str | Path | None) -> list[str] | None:
    """读取 exclude_modules JSON 列表，用于表达搜索空间中的 rank=0 层。"""
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return [str(item) for item in data]


def add_lora_adapter(model: Any, lora_config: dict[str, Any]) -> Any:
    """给基座模型注入 LoRA adapter，并打印可训练参数。"""
    from peft import get_peft_model

    peft_model = get_peft_model(model, build_lora_config(lora_config))
    peft_model.print_trainable_parameters()
    return peft_model


def count_trainable_parameters(model: Any) -> tuple[int, int, float]:
    """统计可训练参数量、总参数量与比例，写入实验指标。"""
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = trainable / total if total else 0.0
    return trainable, total, ratio
