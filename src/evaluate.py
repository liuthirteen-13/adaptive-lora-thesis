"""LoRA adapter 的简单生成式评估脚本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from data import load_examples, render_prompt
from model import ensure_cuda_available, load_base_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 LoRA adapter")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--adapter-path", type=str, help="adapter 路径")
    parser.add_argument("--data-file", type=str, help="评估 JSONL 路径")
    parser.add_argument("--output-file", type=str, help="评估结果输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.adapter_path:
        config.setdefault("model", {})["adapter_path"] = args.adapter_path
    if args.data_file:
        config.setdefault("data", {})["eval_file"] = args.data_file
    if args.output_file:
        config.setdefault("evaluation", {})["output_file"] = args.output_file

    try:
        ensure_cuda_available()
        evaluate(config)
    except RuntimeError as exc:
        print(f"\n[友好错误] {exc}")
        raise SystemExit(1) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def evaluate(config: dict[str, Any]) -> None:
    """加载基座模型和 adapter，生成回答并记录 exact/contains 指标。"""
    import torch
    from peft import PeftModel

    model_config = config["model"]
    data_config = config["data"]
    eval_config = config.get("evaluation", {})

    tokenizer = load_tokenizer(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", True)),
    )
    model = load_base_model(model_config)
    adapter_path = model_config.get("adapter_path") or config.get("training", {}).get("output_dir")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    eval_file = data_config.get("eval_file") or data_config.get("train_file")
    examples = load_examples(
        eval_file,
        data_format=str(data_config.get("format", "auto")),
        max_samples=eval_config.get("max_samples", data_config.get("max_eval_samples")),
    )

    results = []
    exact = 0
    contains = 0
    for example in examples:
        prompt = render_prompt(example, tokenizer=tokenizer, with_answer=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(eval_config.get("max_new_tokens", 256)),
                do_sample=bool(eval_config.get("do_sample", False)),
                temperature=float(eval_config.get("temperature", 0.0)),
                top_p=float(eval_config.get("top_p", 1.0)),
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        gold = example.output.strip()
        is_exact = prediction == gold
        is_contains = gold in prediction or prediction in gold
        exact += int(is_exact)
        contains += int(is_contains)
        results.append(
            {
                "instruction": example.instruction,
                "input": example.input,
                "gold": gold,
                "prediction": prediction,
                "exact_match": is_exact,
                "contains": is_contains,
            }
        )

    metrics = {
        "num_samples": len(examples),
        "exact_match": exact / max(len(examples), 1),
        "contains": contains / max(len(examples), 1),
        "items": results,
    }
    output_file = Path(eval_config.get("output_file", "experiments/runs/lora_r8_eval.json"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"评估结果已保存到：{output_file}")


if __name__ == "__main__":
    main()

