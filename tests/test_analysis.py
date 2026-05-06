from __future__ import annotations

import json
import shutil
from pathlib import Path

from analysis.make_tables import build_efficiency_rows, build_main_rows
from analysis.utils import collect_experiment_records


def test_collect_records_and_build_tables() -> None:
    work_dir = Path(".pytest_cache") / "analysis_logs"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    record_path = work_dir / "lora_record.json"
    record_path.write_text(
        json.dumps(
            {
                "method": "lora",
                "run_name": "lora_smoke",
                "output_dir": "outputs/lora_smoke",
                "metrics": {
                    "train_loss": 1.2,
                    "eval_loss": 2.3,
                    "trainable_params": 128,
                    "total_params": 1024,
                    "trainable_ratio": 0.125,
                    "peak_gpu_memory_mb": 512,
                    "train_time_seconds": 3.4,
                },
                "trainer_log_history": [{"step": 1, "loss": 1.2, "eval_loss": 2.3}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = collect_experiment_records(work_dir, include_discovered=False)
    assert len(records) == 1
    assert records[0].method == "LoRA"
    assert records[0].eval_metric == -2.3
    assert records[0].loss_history[0]["eval_loss"] == 2.3

    main_rows = build_main_rows(records)
    efficiency_rows = build_efficiency_rows(records)
    assert any(row["方法"] == "LoRA" and row["验证损失"] == "2.3000" for row in main_rows)
    assert any(row["方法"] == "LoRA" and row["峰值显存(MB)"] == "512.00" for row in efficiency_rows)

