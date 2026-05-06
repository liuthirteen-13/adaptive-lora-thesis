#!/usr/bin/env bash
set -euo pipefail

# 普通 LoRA r=8 baseline。运行前请先激活 conda 环境：
# conda activate adaptive-lora
python src/train_lora.py --config configs/lora_r8.yaml "$@"

