"""配置读取与命令行覆盖工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件，空文件返回空字典。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """保存 YAML，主要用于后续记录实际实验配置。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置，命令行参数覆盖 YAML 默认值。"""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def print_config(title: str, config: dict[str, Any]) -> None:
    """以 YAML 形式打印配置，方便 dry-run 时检查。"""
    print(f"\n===== {title} =====")
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

