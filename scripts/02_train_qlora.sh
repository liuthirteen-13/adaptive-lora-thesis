#!/usr/bin/env bash
set -euo pipefail

# QLoRA r=8 baseline。默认 dry-run；真实训练请追加 --run。
# conda activate adaptive-lora
# pip install -r requirements-qlora.txt
python src/train_qlora.py --config configs/qlora_r8.yaml "$@"
