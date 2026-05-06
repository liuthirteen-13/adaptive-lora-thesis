"""生成论文实验结果图。

运行示例：
python src/analysis/plot_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.make_tables import build_efficiency_rows, build_main_rows  # noqa: E402
from analysis.utils import (  # noqa: E402
    METHOD_ORDER,
    ExperimentRecord,
    collect_experiment_records,
    load_rank_pattern,
    read_csv_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制论文实验图")
    parser.add_argument("--logs-dir", type=str, default="results/logs", help="实验记录目录")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="CSV 表格目录")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="图片输出目录")
    parser.add_argument("--rank-pattern", type=str, help="rank_pattern 或 best_peft_patterns JSON 路径")
    parser.add_argument(
        "--include-discovered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="同时读取 outputs/ 和 experiments/rank_search/ 中已有记录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_experiment_records(args.logs_dir, include_discovered=bool(args.include_discovered))
    tables_dir = Path(args.tables_dir)
    ensure_tables(records, tables_dir)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    main_rows = read_csv_rows(tables_dir / "main_results.csv")
    efficiency_rows = read_csv_rows(tables_dir / "efficiency_results.csv")
    rank_pattern = load_rank_pattern(args.rank_pattern)

    plot_loss_curves(records, figures_dir)
    plot_param_vs_performance(main_rows, efficiency_rows, figures_dir)
    plot_rank_distribution(rank_pattern, figures_dir)
    plot_memory_comparison(efficiency_rows, figures_dir)

    print(f"图片已输出到：{figures_dir}")


def setup_matplotlib() -> None:
    """设置适合论文的 matplotlib 中文显示与基础样式。"""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def ensure_tables(records: list[ExperimentRecord], tables_dir: Path) -> None:
    """plot_results.py 可单独运行；若 CSV 不存在则先生成。"""
    main_path = tables_dir / "main_results.csv"
    efficiency_path = tables_dir / "efficiency_results.csv"
    if main_path.exists() and efficiency_path.exists():
        return
    write_csv(main_path, build_main_rows(records), ["方法", "运行名", "状态", "指标名称", "验证性能", "验证损失", "训练损失", "综合得分", "记录文件"])
    write_csv(
        efficiency_path,
        build_efficiency_rows(records),
        ["方法", "运行名", "可训练参数量", "总参数量", "可训练参数比例", "参数预算比例", "训练时间(s)", "评估时间(s)", "峰值显存(MB)", "rank预算", "输出目录"],
    )


def plot_loss_curves(records: list[ExperimentRecord], figures_dir: Path) -> None:
    """绘制训练/验证 loss 曲线。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plotted = False
    for record in sorted(records, key=lambda item: (method_index(item.method), item.run_name)):
        history = record.loss_history or single_point_history(record)
        if not history:
            continue
        steps = [item["step"] for item in history]
        if any("loss" in item for item in history):
            values = [item.get("loss") for item in history]
            ax.plot(steps, values, marker="o", linewidth=1.6, label=f"{record.method}-训练")
            plotted = True
        if any("eval_loss" in item for item in history):
            values = [item.get("eval_loss") for item in history]
            ax.plot(steps, values, marker="s", linestyle="--", linewidth=1.6, label=f"{record.method}-验证")
            plotted = True

    if not plotted:
        draw_empty_message(ax, "暂无 loss 曲线数据")
    ax.set_title("训练与验证损失曲线")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=False)
    save_figure(fig, figures_dir / "loss_curves")


def plot_param_vs_performance(main_rows: list[dict[str, str]], efficiency_rows: list[dict[str, str]], figures_dir: Path) -> None:
    """绘制参数量 vs 性能散点图。"""
    import matplotlib.pyplot as plt

    perf_by_key = {(row.get("方法", ""), row.get("运行名", "")): parse_float(row.get("验证性能")) for row in main_rows}
    points = []
    for row in efficiency_rows:
        method = row.get("方法", "")
        run_name = row.get("运行名", "")
        params = parse_float(row.get("可训练参数量"))
        perf = perf_by_key.get((method, run_name))
        if params is None or perf is None:
            continue
        points.append((method, run_name, params, perf))

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if points:
        for method, run_name, params, perf in points:
            ax.scatter(params / 1e6, perf, s=70, label=method, alpha=0.85)
            ax.annotate(method, (params / 1e6, perf), xytext=(5, 4), textcoords="offset points", fontsize=9)
    else:
        draw_empty_message(ax, "暂无参数量与性能数据")
    ax.set_title("参数量与验证性能关系")
    ax.set_xlabel("可训练参数量（百万）")
    ax.set_ylabel("验证性能（越高越好）")
    ax.grid(True, linestyle="--", alpha=0.35)
    deduplicate_legend(ax)
    save_figure(fig, figures_dir / "params_vs_performance")


def plot_rank_distribution(rank_pattern: dict[str, int], figures_dir: Path) -> None:
    """绘制自适应 rank 分布柱状图。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if rank_pattern:
        counts: dict[int, int] = {}
        for rank in rank_pattern.values():
            counts[int(rank)] = counts.get(int(rank), 0) + 1
        ranks = sorted(counts)
        values = [counts[rank] for rank in ranks]
        ax.bar([str(rank) for rank in ranks], values, color="#4C78A8", width=0.65)
        for idx, value in enumerate(values):
            ax.text(idx, value, str(value), ha="center", va="bottom", fontsize=9)
    else:
        draw_empty_message(ax, "暂无 rank_pattern 数据")
    ax.set_title("自适应 LoRA Rank 分布")
    ax.set_xlabel("Rank")
    ax.set_ylabel("层数")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(fig, figures_dir / "rank_distribution")


def plot_memory_comparison(efficiency_rows: list[dict[str, str]], figures_dir: Path) -> None:
    """绘制不同方法峰值显存占用对比图。"""
    import matplotlib.pyplot as plt

    rows = []
    for row in efficiency_rows:
        memory = parse_float(row.get("峰值显存(MB)"))
        if memory is not None:
            rows.append((row.get("方法", ""), memory))
    rows.sort(key=lambda item: method_index(item[0]))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if rows:
        methods = [item[0] for item in rows]
        values = [item[1] for item in rows]
        colors = ["#72B7B2", "#F58518", "#54A24B", "#B279A2", "#E45756", "#4C78A8"][: len(rows)]
        ax.bar(methods, values, color=colors, width=0.62)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    else:
        draw_empty_message(ax, "暂无显存占用数据")
    ax.set_title("不同方法峰值显存占用对比")
    ax.set_xlabel("方法")
    ax.set_ylabel("峰值显存（MB）")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    save_figure(fig, figures_dir / "memory_comparison")


def single_point_history(record: ExperimentRecord) -> list[dict[str, float]]:
    """缺少完整曲线时，用最终 train/eval loss 形成单点图。"""
    point: dict[str, float] = {"step": 1.0}
    if record.train_loss is not None:
        point["loss"] = record.train_loss
    if record.eval_loss is not None:
        point["eval_loss"] = record.eval_loss
    return [point] if len(point) > 1 else []


def draw_empty_message(ax: Any, message: str) -> None:
    """无数据时仍生成可追踪的空图。"""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def deduplicate_legend(ax: Any) -> None:
    """去重图例，避免多 trial 同方法重复。"""
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels, strict=False):
        unique.setdefault(label, handle)
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", frameon=False)


def save_figure(fig: Any, stem: Path) -> None:
    """同时保存 png 和 pdf。"""
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def parse_float(value: Any) -> float | None:
    """CSV 字符串转浮点数。"""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def method_index(method: str) -> int:
    """论文方法顺序。"""
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


if __name__ == "__main__":
    main()

