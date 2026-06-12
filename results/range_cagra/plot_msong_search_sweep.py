#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#9C755F",
    "#76B7B2",
    "#EDC948",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SINGLE_COL = (3.25, 2.25)
DOUBLE_COL = (6.8, 2.35)


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "results/range_cagra/msong_gpu_benchmark/"
            "search_sweep_20260603_225019/sweep_summary.csv"
        ),
        help="Input sweep_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/range_cagra/msong_gpu_benchmark/"
            "search_sweep_20260603_225019/figures"
        ),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validate_data(df: pd.DataFrame) -> None:
    required = {
        "search_config_id",
        "ef",
        "graph_iterations",
        "graph_search_concurrency",
        "entry_count",
        "best_search_seconds",
        "best_qps",
        "recall_at_k",
        "filter_violations",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Sweep input is empty.")
    if df["search_config_id"].duplicated().any():
        dupes = df.loc[df["search_config_id"].duplicated(), "search_config_id"].tolist()
        raise ValueError(f"Duplicate search_config_id values: {dupes}")
    if not df["recall_at_k"].between(0, 1).all():
        raise ValueError("recall_at_k must be in [0, 1].")
    if (df["best_qps"] <= 0).any():
        raise ValueError("best_qps must be positive.")
    if (df["best_search_seconds"] <= 0).any():
        raise ValueError("best_search_seconds must be positive.")
    bad = df[df["filter_violations"] != 0]
    if not bad.empty:
        raise ValueError(
            "Filter violations present in configs: "
            + ",".join(map(str, bad["search_config_id"].tolist()))
        )


def save_figure(fig, output_dir: Path, stem: str, formats) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")


def plot_tradeoff(df: pd.DataFrame, output_dir: Path, formats) -> None:
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for i, row in df.iterrows():
        ax.scatter(
            row["best_qps"],
            row["recall_at_k"],
            color=PALETTE[i % len(PALETTE)],
            marker=MARKERS[i % len(MARKERS)],
            s=30,
            label=f"cfg {int(row.search_config_id)}",
            zorder=3,
        )
        ax.annotate(
            str(int(row["search_config_id"])),
            (row["best_qps"], row["recall_at_k"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("QPS")
    ax.set_ylabel("Recall@10")
    ax.set_title("msong search sweep trade-off")
    ax.grid(True, which="both", axis="both", linewidth=0.3, alpha=0.45)
    ax.set_ylim(max(0.94, df["recall_at_k"].min() - 0.003), min(1.0, df["recall_at_k"].max() + 0.004))
    save_figure(fig, output_dir, "fig_msong_search_sweep_tradeoff", formats)
    plt.close(fig)


def plot_recall_latency(df: pd.DataFrame, output_dir: Path, formats) -> None:
    x = df["search_config_id"].astype(int)

    fig, (ax_recall, ax_time) = plt.subplots(1, 2, figsize=DOUBLE_COL)
    ax_recall.plot(
        x,
        df["recall_at_k"],
        color=PALETTE[0],
        marker=MARKERS[0],
    )
    ax_recall.set_xlabel("Search config")
    ax_recall.set_ylabel("Recall@10")
    ax_recall.set_xticks(x)
    ax_recall.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    ax_recall.set_ylim(max(0.94, df["recall_at_k"].min() - 0.003), min(1.0, df["recall_at_k"].max() + 0.004))

    ax_time.plot(
        x,
        df["best_search_seconds"],
        color=PALETTE[1],
        marker=MARKERS[1],
    )
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Search config")
    ax_time.set_ylabel("Search time (s)")
    ax_time.set_xticks(x)
    ax_time.grid(True, which="both", axis="y", linewidth=0.3, alpha=0.45)

    for ax in (ax_recall, ax_time):
        ax.set_xticklabels(x)

    fig.suptitle("msong range-CAGRA search sensitivity", y=1.02)
    save_figure(fig, output_dir, "fig_msong_search_sweep_recall_latency", formats)
    plt.close(fig)


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    df = load_data(input_path)
    validate_data(df)
    df = df.sort_values("search_config_id").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print("No aggregation applied: one completed result row per search_config_id.")
    print(
        "Best recall config: "
        f"{int(df.loc[df['recall_at_k'].idxmax(), 'search_config_id'])}"
    )
    plot_tradeoff(df, output_dir, args.format)
    plot_recall_latency(df, output_dir, args.format)


if __name__ == "__main__":
    main()
