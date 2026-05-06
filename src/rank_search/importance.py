"""Layer-wise LoRA importance 计算入口。

该模块用于先用小 rank adapter 做少量梯度采样，再把每个 LoRA target module
的梯度范数写成 JSON，供后续 PSO 等智能优化算法构造 rank_pattern 的初始化先验。

运行示例：
python src/rank_search/importance.py --config configs/lora_r8.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import DataCollatorForCausalLM, load_examples, tokenize_examples  # noqa: E402
from model import (  # noqa: E402
    add_lora_adapter,
    load_4bit_base_model,
    load_base_model,
    load_tokenizer,
)


def collect_lora_importance(model: Any, dataloader: Any, num_batches: int) -> dict[str, dict[str, float | int]]:
    """对 LoRA target module 做前向/反向采样，返回平均梯度范数。

    只调用 backward 收集梯度，不创建 optimizer，也不执行 step，因此不会更新模型参数。
    `suggested_rank` 在这里先填为当前 adapter rank，CLI 会按梯度相对强度再做启发式调整。
    """
    import torch

    if num_batches <= 0:
        raise ValueError("num_batches 必须大于 0")

    lora_modules = find_lora_target_modules(model)
    if not lora_modules:
        raise ValueError("未在模型中发现 LoRA target module，请确认已注入 PEFT LoRA adapter")

    input_device = infer_input_device(model)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    grad_norm_sums = {name: 0.0 for name in lora_modules}
    observed_batches = 0

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            batch = move_batch_to_device(batch, input_device)
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = extract_loss(outputs)
            loss.backward()

            for module_name, module in lora_modules.items():
                grad_norm_sums[module_name] += compute_lora_module_grad_norm(module)
            observed_batches += 1

        if observed_batches == 0:
            raise ValueError("dataloader 未产生任何 batch，无法计算 LoRA importance")

        importance: dict[str, dict[str, float | int]] = {}
        for module_name, module in lora_modules.items():
            importance[module_name] = {
                "grad_norm": float(grad_norm_sums[module_name] / observed_batches),
                "suggested_rank": infer_lora_rank(module),
            }
        return importance
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        # 及时释放反向图残留和 CUDA cache，避免 importance 之后接训练时显存被占住。
        if "loss" in locals():
            del loss
        cleanup_memory()


def find_lora_target_modules(model: Any) -> dict[str, Any]:
    """定位带 lora_A/lora_B 的 PEFT 注入层，并规整成 rank_pattern 常用 key。"""
    modules: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            clean_name = normalize_lora_module_name(module_name)
            modules[clean_name] = module
    return dict(sorted(modules.items()))


def normalize_lora_module_name(module_name: str) -> str:
    """去掉 PEFT 包装前缀，使 key 接近 `model.layers.0.self_attn.q_proj`。"""
    prefixes = ("base_model.model.", "base_model.")
    clean_name = module_name
    for prefix in prefixes:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix) :]
            break
    if clean_name.startswith("model.model.layers."):
        clean_name = clean_name[len("model.") :]
    return clean_name


def compute_lora_module_grad_norm(module: Any) -> float:
    """统计一个 target module 内所有 LoRA 参数梯度的 L2 范数。"""
    total_sq_norm = 0.0
    for param_name, parameter in module.named_parameters(recurse=True):
        if "lora_" not in param_name or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_sq_norm += float(grad.norm(2).item() ** 2)
    return math.sqrt(total_sq_norm)


def infer_lora_rank(module: Any) -> int:
    """从 PEFT LoRA A 矩阵形状推断当前 adapter rank。"""
    ranks: list[int] = []
    lora_a = getattr(module, "lora_A", None)
    values = lora_a.values() if hasattr(lora_a, "values") else []
    for adapter_module in values:
        if hasattr(adapter_module, "out_features"):
            ranks.append(int(adapter_module.out_features))
        elif hasattr(adapter_module, "weight"):
            ranks.append(int(adapter_module.weight.shape[0]))
    return max(ranks) if ranks else 0


def assign_suggested_ranks(
    importance: dict[str, dict[str, float | int]],
    min_rank: int,
    max_rank: int,
    rank_step: int,
) -> dict[str, dict[str, float | int]]:
    """按梯度相对强度生成启发式 rank，后续 PSO 可把它作为初始化先验。

    使用 sqrt 缩放避免个别层梯度过大时直接把 rank 拉到上限；最终仍需由搜索目标函数验证。
    """
    if min_rank <= 0 or max_rank < min_rank or rank_step <= 0:
        raise ValueError("min_rank/max_rank/rank_step 配置不合法")

    positive_norms = [float(item["grad_norm"]) for item in importance.values() if float(item["grad_norm"]) > 0]
    mean_norm = sum(positive_norms) / len(positive_norms) if positive_norms else 0.0

    adjusted: dict[str, dict[str, float | int]] = {}
    for module_name, item in importance.items():
        grad_norm = float(item["grad_norm"])
        current_rank = int(item.get("suggested_rank", min_rank) or min_rank)
        if mean_norm <= 0 or grad_norm <= 0:
            raw_rank = float(min_rank)
        else:
            raw_rank = float(current_rank) * math.sqrt(grad_norm / mean_norm)

        adjusted[module_name] = {
            "grad_norm": grad_norm,
            "suggested_rank": snap_rank(raw_rank, min_rank=min_rank, max_rank=max_rank, rank_step=rank_step),
        }
    return adjusted


def snap_rank(raw_rank: float, min_rank: int, max_rank: int, rank_step: int) -> int:
    """把连续 rank 建议投影到 PEFT rank_pattern 可用的离散 rank 集合。"""
    clipped = min(max(raw_rank, float(min_rank)), float(max_rank))
    steps = round((clipped - min_rank) / rank_step)
    snapped = int(min_rank + steps * rank_step)
    return min(max(snapped, min_rank), max_rank)


def save_importance_json(importance: dict[str, dict[str, float | int]], output_file: str | Path) -> Path:
    """写出 importance JSON，便于后处理脚本和论文表格直接读取。"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(importance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算 layer-wise LoRA importance JSON")
    parser.add_argument("--config", type=str, required=True, help="LoRA/QLoRA YAML 配置文件")
    parser.add_argument("--train-file", type=str, help="覆盖配置中的训练 JSONL 路径")
    parser.add_argument("--output-file", type=str, help="importance JSON 输出路径")
    parser.add_argument("--model-name-or-path", type=str, help="覆盖基础模型路径")
    parser.add_argument("--num-batches", type=int, help="用于估计 importance 的 batch 数")
    parser.add_argument("--max-samples", type=int, help="最多读取多少条样本用于 importance 估计")
    parser.add_argument("--min-rank", type=int, help="suggested_rank 下限")
    parser.add_argument("--max-rank", type=int, help="suggested_rank 上限")
    parser.add_argument("--rank-step", type=int, help="suggested_rank 离散步长")
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=None, help="是否按 QLoRA 方式加载")
    parser.add_argument("--allow-cpu", action="store_true", help="允许在 CPU 上调试小模型")
    parser.add_argument("--overwrite-output-file", action="store_true", help="允许覆盖已有 importance JSON")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和路径，不加载模型、不反向传播")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)

    output_file = resolve_output_file(config, overwrite=args.overwrite_output_file)
    print_resolved_config(config, output_file=output_file)

    if args.dry_run:
        dry_run(config, output_file)
        return

    try:
        run_importance(config, output_file=output_file, allow_cpu=args.allow_cpu)
    except RuntimeError as exc:
        print(f"\n[友好错误] {exc}")
        raise SystemExit(1) from exc
    finally:
        cleanup_memory()


def build_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    """合并 YAML 配置和命令行覆盖，保证实验参数可追踪。"""
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if args.train_file:
        config.setdefault("data", {})["train_file"] = args.train_file
    if args.model_name_or_path:
        config.setdefault("model", {})["name_or_path"] = args.model_name_or_path
    if args.output_file:
        config.setdefault("importance", {})["output_file"] = args.output_file
    if args.num_batches is not None:
        config.setdefault("importance", {})["num_batches"] = args.num_batches
    if args.max_samples is not None:
        config.setdefault("importance", {})["max_samples"] = args.max_samples
    if args.min_rank is not None:
        config.setdefault("importance", {})["min_rank"] = args.min_rank
    if args.max_rank is not None:
        config.setdefault("importance", {})["max_rank"] = args.max_rank
    if args.rank_step is not None:
        config.setdefault("importance", {})["rank_step"] = args.rank_step
    if args.qlora is not None:
        config.setdefault("run", {})["qlora"] = args.qlora
    return config


def print_resolved_config(config: dict[str, Any], output_file: Path) -> None:
    """打印实际生效配置，便于把 importance 过程写进论文实验记录。"""
    preview = dict(config)
    preview.setdefault("importance", {})["resolved_output_file"] = str(output_file)
    print("\n===== LoRA importance 配置 =====")
    print(yaml.safe_dump(preview, allow_unicode=True, sort_keys=False))


def dry_run(config: dict[str, Any], output_file: Path) -> None:
    """检查路径和关键字段，避免误以为已启动真实 importance 计算。"""
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    train_file = Path(data_config.get("train_file", ""))
    missing = [
        key
        for key, value in {
            "model.name_or_path": model_config.get("name_or_path"),
            "data.train_file": data_config.get("train_file"),
            "importance.output_file": str(output_file),
        }.items()
        if value in (None, "")
    ]
    if missing:
        raise SystemExit(f"dry-run 配置检查失败，缺少字段：{', '.join(missing)}")
    if not train_file.exists():
        raise SystemExit(f"dry-run 配置检查失败，训练文件不存在：{train_file}")

    print("dry-run 检查通过：不会加载模型，也不会执行前向/反向。")
    print(f"真实 importance JSON 将写入：{output_file}")


def run_importance(config: dict[str, Any], output_file: Path, allow_cpu: bool) -> Path:
    """加载模型与数据，执行 LoRA 梯度范数采样并写出 JSON。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("未安装 PyTorch，请先执行 pip install -r requirements.txt") from exc
    from torch.utils.data import DataLoader

    qlora = is_qlora_config(config)
    ensure_runtime_available(qlora=qlora, allow_cpu=allow_cpu)

    model_config = config["model"]
    data_config = config["data"]
    train_config = config.get("training", {})
    importance_config = config.setdefault("importance", {})

    seed = int(train_config.get("seed", importance_config.get("seed", 42)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model = load_model_with_lora(config, qlora=qlora)

    if not qlora and not has_device_map(model):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)

    max_samples = importance_config.get("max_samples", data_config.get("max_samples"))
    examples = load_examples(
        data_config["train_file"],
        data_format=str(data_config.get("format", "auto")),
        max_samples=max_samples,
    )
    tokenized = tokenize_examples(
        examples,
        tokenizer=tokenizer,
        max_length=int(data_config.get("max_length", 1024)),
        train_on_prompt=bool(data_config.get("train_on_prompt", False)),
    )
    # train_on_prompt=false 且序列被截断时，个别样本可能没有有效 label；这些 batch 对梯度重要性无贡献。
    tokenized = [item for item in tokenized if any(label != -100 for label in item["labels"])]
    if not tokenized:
        raise RuntimeError("tokenize 后没有包含有效 labels 的样本，请调大 data.max_length 或开启 train_on_prompt")
    dataloader = DataLoader(
        tokenized,
        batch_size=int(train_config.get("per_device_train_batch_size", 1)),
        shuffle=False,
        collate_fn=DataCollatorForCausalLM(tokenizer),
    )

    raw_importance = collect_lora_importance(
        model,
        dataloader,
        num_batches=int(importance_config.get("num_batches", 4)),
    )
    importance = assign_suggested_ranks(
        raw_importance,
        min_rank=int(importance_config.get("min_rank", 2)),
        max_rank=int(importance_config.get("max_rank", max(2, int(config["lora"].get("r", 8)) * 4))),
        rank_step=int(importance_config.get("rank_step", 2)),
    )
    saved_path = save_importance_json(importance, output_file)

    print("\n===== LoRA importance 概览 =====")
    print(json.dumps(dict(list(importance.items())[:8]), ensure_ascii=False, indent=2))
    print(f"\nimportance JSON 已保存到：{saved_path}")

    del model, tokenizer, dataloader
    cleanup_memory()
    return saved_path


def load_model_with_lora(config: dict[str, Any], qlora: bool) -> Any:
    """按普通 LoRA 或 QLoRA 加载基座模型并注入 adapter。"""
    model_config = config["model"]
    train_config = config.get("training", {})
    gradient_checkpointing = bool(train_config.get("gradient_checkpointing", True))

    if qlora:
        from peft import prepare_model_for_kbit_training

        model = load_4bit_base_model(model_config, config.get("quantization", {}))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        model = load_base_model(model_config)
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            # 手动 backward 时需要让输入 embedding 保留梯度路径，否则部分模型的 checkpointing 会断开 LoRA 梯度。
            model.enable_input_require_grads()

    if gradient_checkpointing and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return add_lora_adapter(model, config["lora"])


def is_qlora_config(config: dict[str, Any]) -> bool:
    """根据 run.qlora 或 quantization.load_in_4bit 判断是否走 QLoRA 路径。"""
    if "qlora" in config.get("run", {}):
        return bool(config["run"]["qlora"])
    return bool(config.get("quantization", {}).get("load_in_4bit", False))


def ensure_runtime_available(qlora: bool, allow_cpu: bool) -> None:
    """给大模型 importance 计算做运行环境保护，CPU 仅用于小模型调试。"""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("未安装 PyTorch，请先执行 pip install -r requirements.txt") from exc

    if torch.cuda.is_available():
        if qlora:
            try:
                import bitsandbytes  # noqa: F401
            except Exception as exc:
                raise RuntimeError("QLoRA importance 需要可用的 bitsandbytes，请先安装 requirements-qlora.txt") from exc
        return

    if qlora:
        raise RuntimeError("QLoRA importance 需要 CUDA GPU 与 bitsandbytes 4-bit kernel，当前未检测到 GPU")
    if not allow_cpu:
        raise RuntimeError("未检测到 CUDA GPU；如仅调试小模型，请显式追加 --allow-cpu")


def resolve_output_file(config: dict[str, Any], overwrite: bool) -> Path:
    """解析输出路径；默认不覆盖已有 importance 结果。"""
    output_file = config.get("importance", {}).get("output_file")
    if not output_file:
        output_file = default_importance_output_file(config)
        config.setdefault("importance", {})["output_file"] = output_file

    output_path = Path(output_file)
    if output_path.exists() and not overwrite:
        output_path = output_path.with_name(f"{output_path.stem}_{timestamp_suffix()}{output_path.suffix}")
        config["importance"]["output_file"] = str(output_path)
        print(f"检测到 importance JSON 已存在，本次将写入：{output_path}")
    return output_path


def default_importance_output_file(config: dict[str, Any]) -> str:
    """基于模型名和数据文件名生成可区分的默认结果路径。"""
    model_name = str(config.get("model", {}).get("name_or_path", "model")).split("/")[-1]
    data_name = Path(str(config.get("data", {}).get("train_file", "train"))).stem
    return str(Path("results") / "importance" / f"{safe_name(model_name)}_{safe_name(data_name)}_importance.json")


def safe_name(value: str) -> str:
    """把模型名/数据名规整为稳定文件名片段。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower() or "run"


def timestamp_suffix() -> str:
    """生成避免覆盖已有结果的短时间戳。"""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_loss(outputs: Any) -> Any:
    """兼容 transformers ModelOutput、dict 和测试用简单对象。"""
    if isinstance(outputs, dict):
        loss = outputs.get("loss")
    else:
        loss = getattr(outputs, "loss", None)
    if loss is None:
        raise ValueError("模型前向输出不包含 loss，请确认 batch 中提供了 labels")
    return loss


def infer_input_device(model: Any) -> Any:
    """找到 input_ids 应放置的设备；device_map 场景下优先使用 embedding 设备。"""
    import torch

    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            weight = getattr(embeddings, "weight", None)
            if weight is not None and weight.device.type != "meta":
                return weight.device
            for parameter in embeddings.parameters():
                if parameter.device.type != "meta":
                    return parameter.device

    # PEFT 包装或部分 device_map 场景下 get_input_embeddings 可能拿不到真实权重，
    # 这里按常见 embedding 参数名再找一次，保证 input_ids 跟 embedding lookup 同设备。
    embedding_tokens = ("embed_tokens", "word_embeddings", ".wte.", ".embed.")
    for name, parameter in model.named_parameters():
        if any(token in name for token in embedding_tokens) and parameter.device.type != "meta":
            return parameter.device

    for parameter in model.parameters():
        if parameter.device.type == "cuda":
            return parameter.device

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def has_device_map(model: Any) -> bool:
    """判断模型是否已由 transformers/accelerate 设备映射管理。"""
    return bool(getattr(model, "hf_device_map", None) or getattr(getattr(model, "base_model", None), "hf_device_map", None))


def move_batch_to_device(batch: Any, device: Any) -> Any:
    """递归移动 batch tensor，兼容 dict/list/tuple 嵌套结构。"""
    import torch

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if hasattr(batch, "to") and callable(batch.to):
        try:
            return batch.to(device)
        except TypeError:
            pass
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def cleanup_memory() -> None:
    """清理 Python 和 CUDA 显存缓存。"""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


if __name__ == "__main__":
    main()
