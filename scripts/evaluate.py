"""生成式评估入口，支持基础模型与 PEFT adapter。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_lora_thesis.config import deep_update, load_yaml, print_config
from adaptive_lora_thesis.data import normalize_instruction_record, read_jsonl, render_record
from adaptive_lora_thesis.lora import resolve_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 LoRA/QLoRA adapter")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="YAML 配置文件")
    parser.add_argument("--model-name-or-path", type=str, help="基础模型名或本地路径")
    parser.add_argument("--adapter-path", type=str, help="PEFT adapter 路径")
    parser.add_argument("--data-path", type=str, help="评估 JSONL 路径")
    parser.add_argument("--output-file", type=str, help="评估结果 JSON 输出路径")
    parser.add_argument("--max-samples", type=int, help="最多评估多少条样本")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None, help="只检查配置，不加载模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    print_config("评估配置", config)

    if config["run"].get("dry_run", True):
        print("当前为 dry-run：未加载模型，未执行生成评估。")
        return

    evaluate(config)


def build_runtime_config(args: argparse.Namespace) -> dict:
    """合并 YAML 配置与命令行覆盖。"""
    config = load_yaml(args.config)
    override = {
        "model": {
            "name_or_path": args.model_name_or_path,
            "adapter_path": args.adapter_path,
        },
        "data": {
            "path": args.data_path,
            "max_samples": args.max_samples,
        },
        "output": {"file": args.output_file},
        "run": {"dry_run": args.dry_run},
    }
    return deep_update(config, override)


def evaluate(config: dict) -> None:
    """执行简单 exact-match/contains 评估，后续可扩展 GSM8K 数值抽取。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    generation_config = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
        torch_dtype=resolve_torch_dtype(model_config.get("torch_dtype")),
        device_map=model_config.get("device_map", "auto"),
    )
    if model_config.get("adapter_path"):
        model = PeftModel.from_pretrained(model, model_config["adapter_path"])
    model.eval()

    records = [normalize_instruction_record(record) for record in read_jsonl(config["data"]["path"])]
    max_samples = config["data"].get("max_samples")
    if max_samples is not None:
        records = records[: int(max_samples)]

    results = []
    exact_match = 0
    contains = 0
    for record in records:
        prompt = render_record(record, tokenizer=tokenizer, include_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(generation_config.get("max_new_tokens", 256)),
                temperature=float(generation_config.get("temperature", 0.0)),
                top_p=float(generation_config.get("top_p", 1.0)),
                do_sample=bool(generation_config.get("do_sample", False)),
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        gold = record["output"].strip()
        exact = prediction == gold
        contain = gold in prediction or prediction in gold
        exact_match += int(exact)
        contains += int(contain)
        results.append(
            {
                "instruction": record["instruction"],
                "input": record.get("input", ""),
                "gold": gold,
                "prediction": prediction,
                "exact_match": exact,
                "contains": contain,
            }
        )

    metrics = {
        "num_samples": len(records),
        "exact_match": exact_match / max(len(records), 1),
        "contains": contains / max(len(records), 1),
        "items": results,
    }
    output_path = Path(config["output"]["file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"评估完成：{output_path}")


if __name__ == "__main__":
    main()

