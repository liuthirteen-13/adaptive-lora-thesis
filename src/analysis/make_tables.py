"""生成论文实验结果 CSV 表格。

运行示例：
python src/analysis/make_tables.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.utils import (  # noqa: E402
    METHOD_ORDER,
    ExperimentRecord,
    best_records_by_method,
    collect_experiment_records,
    format_number,
    method_sort_key,
    write_csv,
)


MAIN_FIELDS = [
    "方法",
    "运行名",
    "状态",
    "指标名称",
    "验证性能",
    "验证损失",
    "训练损失",
    "综合得分",
    "记录文件",
]

EFFICIENCY_FIELDS = [
    "方法",
    "运行名",
    "可训练参数量",
    "总参数量",
    "可训练参数比例",
    "参数预算比例",
    "训练时间(s)",
    "评估时间(s)",
    "峰值显存(MB)",
    "rank预算",
    "输出目录",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总论文实验记录并生成 CSV 表格")
    parser.add_argument("--logs-dir", type=str, default="results/logs", help="实验记录目录")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="CSV 输出目录")
    parser.add_argument(
        "--include-discovered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="同时读取 outputs/ 和 experiments/rank_search/ 中已有记录",
    )
    parser.add_argument("--all-trials", action="store_true", help="输出所有 trial；默认每个方法只保留最佳记录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_experiment_records(args.logs_dir, include_discovered=bool(args.include_discovered))
    selected = records if args.all_trials else best_records_by_method(records)

    tables_dir = Path(args.tables_dir)
    main_path = tables_dir / "main_results.csv"
    efficiency_path = tables_dir / "efficiency_results.csv"
    write_csv(main_path, build_main_rows(selected), MAIN_FIELDS)
    write_csv(efficiency_path, build_efficiency_rows(selected), EFFICIENCY_FIELDS)

    print(f"读取记录数：{len(records)}")
    print(f"主结果表：{main_path}")
    print(f"效率结果表：{efficiency_path}")


def build_main_rows(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    """构造主结果表，重点呈现验证性能与 loss。"""
    rows = []
    for record in sorted(records, key=lambda item: (method_sort_key(item.method), item.run_name)):
        rows.append(
            {
                "方法": record.method,
                "运行名": record.run_name,
                "状态": record.status,
                "指标名称": record.metric_name,
                "验证性能": format_number(record.eval_metric),
                "验证损失": format_number(record.eval_loss),
                "训练损失": format_number(record.train_loss),
                "综合得分": format_number(record.score),
                "记录文件": record.record_path,
            }
        )
    return add_missing_method_rows(rows, table_type="main")


def build_efficiency_rows(records: list[ExperimentRecord]) -> list[dict[str, Any]]:
    """构造效率表，重点呈现参数量、耗时和显存。"""
    rows = []
    for record in sorted(records, key=lambda item: (method_sort_key(item.method), item.run_name)):
        rows.append(
            {
                "方法": record.method,
                "运行名": record.run_name,
                "可训练参数量": record.trainable_params or "",
                "总参数量": record.total_params or "",
                "可训练参数比例": format_number(record.trainable_ratio, digits=6),
                "参数预算比例": format_number(record.param_ratio),
                "训练时间(s)": format_number(record.train_time_seconds, digits=2),
                "评估时间(s)": format_number(record.eval_time_seconds, digits=2),
                "峰值显存(MB)": format_number(record.peak_gpu_memory_mb, digits=2),
                "rank预算": record.rank_budget or "",
                "输出目录": record.output_dir,
            }
        )
    return add_missing_method_rows(rows, table_type="efficiency")


def add_missing_method_rows(rows: list[dict[str, Any]], table_type: str) -> list[dict[str, Any]]:
    """为暂未完成的实验方法保留空行，方便论文表格后续补数。"""
    existing = {str(row.get("方法")) for row in rows}
    for method in METHOD_ORDER:
        if method in existing:
            continue
        if table_type == "main":
            rows.append({"方法": method, "状态": "missing"})
        else:
            rows.append({"方法": method, "运行名": "", "输出目录": ""})
    return sorted(rows, key=lambda row: method_sort_key(str(row.get("方法", ""))))


if __name__ == "__main__":
    main()

