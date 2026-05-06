# adaptive-lora-thesis

本项目服务于硕士论文题目：**基于智能优化的自适应低秩大模型微调方法研究**。

当前工程包含普通 LoRA baseline、QLoRA 4-bit baseline、PSO 后续扩展骨架、JSONL 数据准备与评估入口。普通 LoRA 与 QLoRA baseline 面向 `Qwen/Qwen2.5-1.5B-Instruct`，使用 Transformers + PEFT 实现。

## 1. 环境安装

```bash
conda create -n adaptive-lora python=3.10 -y
conda activate adaptive-lora

pip install -r requirements.txt
pip install -e .
```

普通 LoRA baseline 需要 CUDA GPU。如果在无 GPU 环境运行，脚本会在加载模型前给出友好报错。

QLoRA 需要额外安装 bitsandbytes：

```bash
pip install -r requirements-qlora.txt
```

注意：bitsandbytes 的 4-bit 训练依赖 CUDA 环境，Windows 可用性取决于当前 wheel、CUDA 与 PyTorch 版本。如果 QLoRA 环境检查失败，可先切换普通 LoRA 跑通流程。

注意：普通 LoRA 训练不建议使用 `device_map: auto`。`auto` 在 4GB 显存等小显存环境下会把部分层 offload 到 CPU/meta device，训练反传可能报 `expected device meta but got cuda:0`。本项目的 `configs/lora_r8.yaml` 默认使用单卡加载；如果显存不足，请优先降低 `data.max_length`，或改用 QLoRA/更小模型。

## 2. 数据格式

支持 instruction JSONL：

```json
{"instruction": "请解答下面的问题。", "input": "1+1 等于多少？", "output": "2"}
```

也支持 GSM8K 风格 question-answer JSONL：

```json
{"question": "Natalia sold clips to 48 of her friends...", "answer": "Natalia sold 48/2 = 24 clips... #### 72"}
```

使用项目自带样例数据：

```bash
python scripts/prepare_data.py \
  --input data/samples/instruction_sample.jsonl \
  --output data/processed/instruction_train.jsonl \
  --format instruction
```

准备 GSM8K：

```bash
python scripts/prepare_data.py \
  --dataset-name gsm8k \
  --dataset-config main \
  --split train \
  --output data/processed/gsm8k_train.jsonl \
  --format gsm8k \
  --max-samples 1000
```

## 3. 普通 LoRA Baseline

默认配置见 `configs/lora_r8.yaml`：

- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.05`
- `target_modules = ["q_proj", "v_proj"]`
- `task_type = CAUSAL_LM`
- `device_map = null`，避免训练时 CPU/meta offload
- `max_length = 256`，方便 RTX 3050 Ti 4GB 先跑通流程

启动训练：

```bash
python src/train_lora.py --config configs/lora_r8.yaml
```

或使用 shell 脚本：

```bash
bash scripts/02_train_lora.sh
```

覆盖数据和输出目录：

```bash
python src/train_lora.py \
  --config configs/lora_r8.yaml \
  --train-file data/processed/gsm8k_train.jsonl \
  --output-dir outputs/lora_r8_qwen25_gsm8k
```

训练完成后，adapter 权重、tokenizer、实际配置和指标会保存到 `training.output_dir`。指标文件为：

```text
outputs/lora_r8_qwen25/training_metrics.json
```

其中包含：

- `train_loss`
- `eval_loss`
- `peak_gpu_memory_mb`
- `train_time_seconds`
- `trainable_params`
- `total_params`
- `trainable_ratio`

代码会在训练前调用 `model.print_trainable_parameters()` 打印 PEFT 可训练参数。

## 4. QLoRA 4-bit Baseline

默认配置见 `configs/qlora_r8.yaml`：

- `r = 8`
- `target_modules = "all-linear"`，用于标准 QLoRA baseline；如需和普通 LoRA 完全同模块对比，可改为 `["q_proj", "v_proj"]`
- `load_in_4bit = true`
- `bnb_4bit_quant_type = "nf4"`
- `bnb_4bit_compute_dtype = "float16"`
- `bnb_4bit_use_double_quant = true`
- `optim = "paged_adamw_8bit"`
- 默认 `run.dry_run = true`，不会自动启动大模型训练

先执行 dry-run 检查配置和数据路径：

```bash
python src/train_qlora.py --config configs/qlora_r8.yaml --dry-run
```

或使用脚本，默认同样是 dry-run：

```bash
bash scripts/02_train_qlora.sh
```

确认 bitsandbytes 与 CUDA 环境可用后，显式关闭 dry-run 启动训练：

```bash
python src/train_qlora.py --config configs/qlora_r8.yaml --run
```

或：

```bash
bash scripts/02_train_qlora.sh --run
```

覆盖数据和输出目录：

```bash
python src/train_qlora.py \
  --config configs/qlora_r8.yaml \
  --train-file data/processed/gsm8k_train.jsonl \
  --output-dir outputs/qlora_r8_qwen25_gsm8k \
  --run
```

如果当前系统不支持 bitsandbytes，脚本会停止并提示切换普通 LoRA：

```bash
python src/train_lora.py --config configs/lora_r8.yaml
```

训练完成后，QLoRA adapter、tokenizer、实际配置和指标会保存到 `training.output_dir`。指标文件为：

```text
outputs/qlora_r8_qwen25/training_metrics.json
```

其中包含 `peak_gpu_memory_mb`、`peak_gpu_memory_reserved_mb`、`train_time_seconds`、`trainable_params`、`total_params`、`train_loss` 和 `eval_loss`。完整训练记录和 Trainer 日志历史会写入：

```text
results/logs/
```

## 5. Baseline 评估

```bash
python src/evaluate.py \
  --config configs/lora_r8.yaml \
  --adapter-path outputs/lora_r8_qwen25 \
  --data-file data/processed/instruction_train.jsonl \
  --output-file experiments/runs/lora_r8_eval.json
```

QLoRA 与普通 LoRA 共用同一个评估脚本，只需替换配置和 adapter 路径：

```bash
python src/evaluate.py \
  --config configs/qlora_r8.yaml \
  --adapter-path outputs/qlora_r8_qwen25 \
  --data-file data/processed/instruction_train.jsonl \
  --output-file experiments/runs/qlora_r8_eval.json
```

## 6. Layer-wise LoRA Importance

importance 模块用于自适应 rank 分配前的轻量梯度采样：加载 LoRA/QLoRA adapter 结构，对少量 batch 执行前向和反向，只统计各 LoRA target module 的梯度范数，不执行 optimizer step，也不会更新模型参数。输出 JSON 可作为后续 PSO rank_pattern 搜索的初始化先验或约束参考。

先执行 dry-run 检查配置与输出路径：

```bash
python src/rank_search/importance.py --config configs/lora_r8.yaml --dry-run
```

确认 CUDA 环境和数据路径可用后，生成 importance JSON：

```bash
python src/rank_search/importance.py --config configs/lora_r8.yaml
```

默认输出到：

```text
results/importance/qwen25_1_5b_instruction_importance.json
```

也可以覆盖数据、采样 batch 数和输出文件，例如 GSM8K：

```bash
python src/rank_search/importance.py \
  --config configs/lora_r8.yaml \
  --train-file data/processed/gsm8k_train.jsonl \
  --output-file results/importance/qwen15b_gsm8k_importance.json \
  --num-batches 8 \
  --max-samples 32
```

QLoRA 使用同一入口，配置中 `run.qlora: true` 时会按 4-bit 路径加载；也可显式指定：

```bash
python src/rank_search/importance.py --config configs/qlora_r8.yaml --qlora
```

输出格式示例：

```json
{
  "model.layers.0.self_attn.q_proj": {
    "grad_norm": 0.0123,
    "suggested_rank": 8
  }
}
```

后续 rank 搜索可读取该文件，把 `suggested_rank` 作为初始候选，或把 `grad_norm` 归一化后作为 PSO 粒子初始化和 rank budget 分配的先验。

## 7. PSO 自适应 Rank 搜索

PSO 搜索入口会读取 `results/importance/*.json`，每个粒子表示一组 layer-wise rank 分配。默认候选 rank 集合为 `[0, 2, 4, 8, 16, 32]`，其中 `rank=0` 会转换为 PEFT `exclude_modules`，正 rank 会写入 `rank_pattern` 和 `alpha_pattern`，保证最终 `LoraConfig` 可直接加载。

先执行 dry-run 检查搜索空间，不训练模型：

```bash
python src/rank_search/pso.py --config configs/ours_pso_rank_lora.yaml --dry-run
```

显式关闭 dry-run 后启动真实搜索。配置默认只用小训练子集、少量 step 和验证子集，避免完整训练：

```bash
python src/rank_search/pso.py --config configs/ours_pso_rank_lora.yaml --run
```

搜索支持断点恢复：同一命令会从 `experiments/rank_search/ours_pso/state.json` 继续；如需重新开始，追加 `--no-resume`。每个 trial 的配置、rank pattern、指标、耗时、显存和失败原因会写入：

```text
experiments/rank_search/ours_pso/trials/
```

搜索结束后会生成：

```text
experiments/rank_search/ours_pso/best_rank_pattern.json
experiments/rank_search/ours_pso/best_alpha_pattern.json
experiments/rank_search/ours_pso/best_exclude_modules.json
experiments/rank_search/ours_pso/best_lora_config.yaml
```

最终训练建议直接使用搜索生成的配置，它已经引用了 `best_rank_pattern.json`、`best_alpha_pattern.json` 和 `best_exclude_modules.json`：

```bash
python src/train_lora.py \
  --config experiments/rank_search/ours_pso/best_lora_config.yaml \
  --output-dir outputs/ours_pso_rank_lora_qwen25
```

如果要手动接入其他训练配置，只需在 `lora` 段加入：

```yaml
rank_pattern_path: experiments/rank_search/ours_pso/best_rank_pattern.json
alpha_pattern_path: experiments/rank_search/ours_pso/best_alpha_pattern.json
exclude_modules_path: experiments/rank_search/ours_pso/best_exclude_modules.json
```

## 8. 论文图表生成

实验记录优先从 `results/logs/` 读取；脚本也兼容当前工程已有的 `outputs/*/training_metrics.json` 和 `experiments/rank_search/**/trial_record.json`，便于把 LoRA baseline 与 PSO trial 先汇总起来。支持方法名包括 Base、Prompt、LoRA、QLoRA、AdaLoRA 和 Ours。

生成主结果表和效率表：

```bash
python src/analysis/make_tables.py
```

输出文件：

```text
results/tables/main_results.csv
results/tables/efficiency_results.csv
```

生成论文图，包括 loss 曲线、参数量 vs 性能、rank 分布柱状图和显存占用对比图：

```bash
python src/analysis/plot_results.py
```

所有图片会同时保存为 PNG 和 PDF：

```text
results/figures/loss_curves.png
results/figures/loss_curves.pdf
results/figures/params_vs_performance.png
results/figures/params_vs_performance.pdf
results/figures/rank_distribution.png
results/figures/rank_distribution.pdf
results/figures/memory_comparison.png
results/figures/memory_comparison.pdf
```

绘图只使用 matplotlib，不依赖 seaborn。若新增 Base、Prompt、AdaLoRA 等实验，只需把同结构 JSON 记录放入 `results/logs/`，重新运行上述两个脚本即可更新表格和图片。

## 9. 旧版 Dry-run 入口

这些脚本仍保留，用于后续 QLoRA 和 PSO rank_pattern 搜索扩展：

```bash
python scripts/train_lora.py --config configs/train_lora.yaml --dry-run
python scripts/train_lora.py --config configs/train_qlora.yaml --dry-run
python scripts/search_pso_rank.py --config configs/pso_search.yaml --dry-run
python scripts/evaluate.py --config configs/eval.yaml --dry-run
```

## 10. 文件用途说明

| 路径 | 用途 |
| --- | --- |
| `src/data.py` | 普通 LoRA baseline 的 JSONL/GSM8K 数据读取、prompt 渲染、tokenize 与 collator。 |
| `src/model.py` | Qwen2.5 基座模型加载、4-bit 量化配置、LoRA 配置构造、adapter 注入、可训练参数统计。 |
| `src/train_lora.py` | 普通 LoRA baseline 训练入口，保存 adapter 和训练指标。 |
| `src/train_qlora.py` | QLoRA 4-bit baseline 训练入口，默认 dry-run，保存 adapter、显存峰值、训练耗时和日志记录。 |
| `src/evaluate.py` | LoRA adapter 生成式评估入口。 |
| `src/rank_search/importance.py` | layer-wise LoRA importance 计算入口，输出梯度范数和 suggested_rank JSON。 |
| `src/rank_search/build_rank_pattern.py` | 将搜索 rank 分配转换为 PEFT 可用的 rank_pattern、alpha_pattern 和 exclude_modules。 |
| `src/rank_search/fitness.py` | PSO trial 适应度函数，记录少步训练/验证指标、参数量、耗时和失败原因。 |
| `src/rank_search/pso.py` | 可恢复的 PSO rank_pattern 搜索入口。 |
| `src/analysis/make_tables.py` | 从实验记录生成 `main_results.csv` 和 `efficiency_results.csv`。 |
| `src/analysis/plot_results.py` | 生成论文用 loss、参数-性能、rank 分布和显存对比图。 |
| `configs/lora_r8.yaml` | 普通 LoRA r=8 baseline 配置。 |
| `configs/qlora_r8.yaml` | QLoRA r=8 baseline 配置。 |
| `configs/ours_pso_rank_lora.yaml` | 本文方法的 PSO 自适应 rank 搜索配置，默认 dry-run。 |
| `scripts/02_train_lora.sh` | 一键启动普通 LoRA baseline 的 shell 脚本。 |
| `scripts/02_train_qlora.sh` | QLoRA baseline 脚本，默认 dry-run，真实训练需追加 `--run`。 |
| `results/logs/` | QLoRA 训练记录和 Trainer 日志历史。 |
| `results/importance/` | LoRA importance JSON 输出目录，默认不进入版本管理。 |
| `results/tables/` | 论文 CSV 表格输出目录，默认不进入版本管理。 |
| `results/figures/` | 论文图片输出目录，默认不进入版本管理。 |
| `experiments/rank_search/` | PSO trial 记录、恢复状态和 best rank/alpha pattern 输出目录，默认不进入版本管理。 |
| `requirements.txt` | 核心依赖。 |
| `requirements-qlora.txt` | QLoRA 额外依赖。 |
| `AGENTS.md` | 后续 Codex 修改项目时必须遵守的规则。 |
| `src/adaptive_lora_thesis/` | 后续 QLoRA、PSO 搜索和通用工具的包代码。 |
