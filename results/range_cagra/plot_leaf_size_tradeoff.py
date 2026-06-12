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
SINGLE_COL = (3.25, 2.35)
DOUBLE_COL = (6.8, 2.55)


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
        "--run-dir",
        default="results/range_cagra/segment_tree_param_sweep/msong_leaf_coarse_20260605",
        help="Leaf sweep directory containing aggregate CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures. Defaults to <run-dir>/figures.",
    )
    parser.add_argument(
        "--derived-dir",
        default=None,
        help="Output directory for derived CSVs. Defaults to <run-dir>/derived.",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png", "svg"],
    )
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validate(summary: pd.DataFrame, sweep: pd.DataFrame, phase: pd.DataFrame) -> None:
    required_summary = {
        "dataset",
        "config_label",
        "leaf_size",
        "leaf_blocks",
        "graph_count_est",
        "edge_gib",
        "build_seconds",
        "nq",
        "recall_at_k",
        "best_qps",
        "filter_violations",
    }
    required_sweep = {
        "leaf_size",
        "search_config_id",
        "ef",
        "graph_iterations",
        "graph_search_concurrency",
        "entry_count",
        "best_search_seconds",
        "best_qps",
        "recall_at_k",
        "filter_violations",
        "exact_seconds",
        "graph_seconds",
        "merge_seconds",
        "exact_vectors_scanned",
        "graph_node_tasks",
    }
    required_phase = {"leaf_size", "phase", "peak_memory_used_mb", "avg_memory_used_mb"}
    missing_summary = required_summary - set(summary.columns)
    missing_sweep = required_sweep - set(sweep.columns)
    missing_phase = required_phase - set(phase.columns)
    if missing_summary:
        raise ValueError(f"Missing summary columns: {sorted(missing_summary)}")
    if missing_sweep:
        raise ValueError(f"Missing sweep columns: {sorted(missing_sweep)}")
    if missing_phase:
        raise ValueError(f"Missing phase columns: {sorted(missing_phase)}")
    if summary.empty or sweep.empty or phase.empty:
        raise ValueError("Input aggregate CSVs must be non-empty.")
    if summary["leaf_size"].duplicated().any():
        dupes = summary.loc[summary["leaf_size"].duplicated(), "leaf_size"].tolist()
        raise ValueError(f"Duplicate build summary leaf_size rows: {dupes}")
    if sweep.duplicated(["leaf_size", "search_config_id"]).any():
        raise ValueError("Duplicate sweep rows by leaf_size/search_config_id.")
    if not summary["recall_at_k"].between(0, 1).all():
        raise ValueError("summary recall_at_k must be in [0, 1].")
    if not sweep["recall_at_k"].between(0, 1).all():
        raise ValueError("sweep recall_at_k must be in [0, 1].")
    if (summary["best_qps"] <= 0).any() or (sweep["best_qps"] <= 0).any():
        raise ValueError("QPS values must be positive.")
    if (summary["build_seconds"] <= 0).any():
        raise ValueError("Build time must be positive.")
    if summary["filter_violations"].sum() != 0 or sweep["filter_violations"].sum() != 0:
        raise ValueError("Filter violations are present; inspect correctness before plotting.")


def save_figure(fig, output_dir: Path, stem: str, formats) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"Saved {out}")


def cfg_label(row: pd.Series) -> str:
    return (
        f"cfg {int(row.search_config_id)} "
        f"(ef={int(row.ef)}, it={int(row.graph_iterations)}, "
        f"c={int(row.graph_search_concurrency)})"
    )


def prepare(summary: pd.DataFrame, sweep: pd.DataFrame, phase: pd.DataFrame):
    summary = summary.copy()
    sweep = sweep.copy()
    phase = phase.copy()
    for df in (summary, sweep, phase):
        df["leaf_size"] = pd.to_numeric(df["leaf_size"])

    build_phase = phase[phase["phase"] == "build"].copy()
    build = summary.merge(
        build_phase[
            [
                "leaf_size",
                "peak_memory_used_mb",
                "avg_memory_used_mb",
                "peak_gpu_util_pct",
                "avg_gpu_util_pct",
            ]
        ],
        on="leaf_size",
        how="left",
    )
    if build["peak_memory_used_mb"].isna().any():
        missing = build.loc[build["peak_memory_used_mb"].isna(), "leaf_size"].tolist()
        raise ValueError(f"Missing build phase GPU rows for leaf sizes: {missing}")

    build = build.sort_values("leaf_size").reset_index(drop=True)
    sweep = sweep.sort_values(["search_config_id", "leaf_size"]).reset_index(drop=True)

    build["build_peak_memory_gib"] = build["peak_memory_used_mb"] / 1024.0
    build["build_avg_memory_gib"] = build["avg_memory_used_mb"] / 1024.0
    build["edge_gib"] = pd.to_numeric(build["edge_gib"])
    build["build_seconds"] = pd.to_numeric(build["build_seconds"])
    build["nq"] = pd.to_numeric(build["nq"])

    for col in [
        "best_qps",
        "recall_at_k",
        "best_search_seconds",
        "exact_seconds",
        "graph_seconds",
        "merge_seconds",
        "exact_vectors_scanned",
        "graph_node_tasks",
    ]:
        sweep[col] = pd.to_numeric(sweep[col])
    sweep["latency_ms_per_query"] = sweep["best_search_seconds"] * 1000.0 / build["nq"].iloc[0]
    sweep["exact_ms_per_query"] = sweep["exact_seconds"] * 1000.0 / build["nq"].iloc[0]
    sweep["graph_ms_per_query"] = sweep["graph_seconds"] * 1000.0 / build["nq"].iloc[0]
    sweep["merge_ms_per_query"] = sweep["merge_seconds"] * 1000.0 / build["nq"].iloc[0]
    sweep["cfg_label"] = sweep.apply(cfg_label, axis=1)
    return build, sweep


def plot_build(build: pd.DataFrame, output_dir: Path, formats) -> None:
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COL)
    x = build["leaf_size"]

    axes[0].plot(x, build["build_seconds"], color=PALETTE[0], marker=MARKERS[0])
    axes[0].set_ylabel("Build Time (s)")
    axes[0].set_xlabel("Leaf Size")
    axes[0].grid(True, axis="y", linewidth=0.3, alpha=0.45)

    axes[1].plot(x, build["build_peak_memory_gib"], color=PALETTE[1], marker=MARKERS[1])
    axes[1].set_ylabel("Peak Build Memory (GiB)")
    axes[1].set_xlabel("Leaf Size")
    axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.45)

    axes[2].plot(x, build["edge_gib"], color=PALETTE[2], marker=MARKERS[2])
    axes[2].set_ylabel("Graph Edges (GiB)")
    axes[2].set_xlabel("Leaf Size")
    axes[2].grid(True, axis="y", linewidth=0.3, alpha=0.45)

    for ax in axes:
        ax.set_xticks(x)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Leaf size vs build resource cost", y=1.03)
    save_figure(fig, output_dir, "fig_leaf_size_build_resource", formats)
    plt.close(fig)


def plot_search(sweep: pd.DataFrame, output_dir: Path, formats) -> None:
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    for i, (label, group) in enumerate(sweep.groupby("cfg_label", sort=False)):
        axes[0].plot(
            group["leaf_size"],
            group["best_qps"],
            color=PALETTE[i % len(PALETTE)],
            marker=MARKERS[i % len(MARKERS)],
            label=label,
        )
        axes[1].plot(
            group["leaf_size"],
            group["recall_at_k"],
            color=PALETTE[i % len(PALETTE)],
            marker=MARKERS[i % len(MARKERS)],
            label=label,
        )

    axes[0].set_ylabel("QPS")
    axes[0].set_xlabel("Leaf Size")
    axes[0].grid(True, axis="y", linewidth=0.3, alpha=0.45)

    axes[1].set_ylabel("Recall@10")
    axes[1].set_xlabel("Leaf Size")
    axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.45)
    axes[1].set_ylim(max(0.98, sweep["recall_at_k"].min() - 0.002), 1.0)

    for ax in axes:
        ax.set_xticks(sorted(sweep["leaf_size"].unique()))
        ax.tick_params(axis="x", rotation=35)
    axes[0].legend(frameon=False)
    fig.suptitle("Leaf size vs search throughput and quality", y=1.03)
    save_figure(fig, output_dir, "fig_leaf_size_search_quality", formats)
    plt.close(fig)


def plot_breakdown(sweep: pd.DataFrame, output_dir: Path, formats) -> None:
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    metrics = [
        ("exact_ms_per_query", "Exact Scan (ms/query)", PALETTE[0], MARKERS[0]),
        ("graph_ms_per_query", "Graph Search (ms/query)", PALETTE[1], MARKERS[1]),
    ]
    for ax, (label, group) in zip(axes, sweep.groupby("cfg_label", sort=False)):
        for metric, metric_label, color, marker in metrics:
            ax.plot(group["leaf_size"], group[metric], color=color, marker=marker, label=metric_label)
        ax.set_title(label)
        ax.set_xlabel("Leaf Size")
        ax.set_ylabel("Time (ms/query)")
        ax.set_xticks(sorted(sweep["leaf_size"].unique()))
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    axes[0].legend(frameon=False)
    fig.suptitle("Leaf size search-time components", y=1.03)
    save_figure(fig, output_dir, "fig_leaf_size_search_breakdown", formats)
    plt.close(fig)


def write_report(run_dir: Path, build: pd.DataFrame, sweep: pd.DataFrame) -> None:
    cfg0 = sweep[sweep["search_config_id"] == sweep["search_config_id"].min()].copy()
    cfg1 = sweep[sweep["search_config_id"] == sweep["search_config_id"].max()].copy()
    merged = build[
        [
            "leaf_size",
            "leaf_blocks",
            "graph_count_est",
            "edge_gib",
            "build_seconds",
            "build_peak_memory_gib",
        ]
    ].merge(
        cfg0[["leaf_size", "best_qps", "recall_at_k", "exact_ms_per_query", "graph_ms_per_query"]]
        .rename(
            columns={
                "best_qps": "cfg0_qps",
                "recall_at_k": "cfg0_recall",
                "exact_ms_per_query": "cfg0_exact_ms",
                "graph_ms_per_query": "cfg0_graph_ms",
            }
        ),
        on="leaf_size",
    ).merge(
        cfg1[["leaf_size", "best_qps", "recall_at_k", "exact_ms_per_query", "graph_ms_per_query"]]
        .rename(
            columns={
                "best_qps": "cfg1_qps",
                "recall_at_k": "cfg1_recall",
                "exact_ms_per_query": "cfg1_exact_ms",
                "graph_ms_per_query": "cfg1_graph_ms",
            }
        ),
        on="leaf_size",
    )

    def pct_delta(new, old):
        return 100.0 * (float(new) - float(old)) / float(old)

    leaf_min = merged.iloc[0]
    leaf_1000 = merged.loc[merged["leaf_size"] == 1000].iloc[0]
    leaf_max = merged.iloc[-1]
    best_cfg0 = cfg0.loc[cfg0["best_qps"].idxmax()]
    best_cfg1 = cfg1.loc[cfg1["best_qps"].idxmax()]

    lines = [
        "# Leaf-size sweep: build/search trade-off",
        "",
        "Dataset/workload: `msong/pos_w50`; build policy: `adaptive_d32_i96_min8_i24`; queries: 2000.",
        "",
        "## Main Table",
        "",
        "| leaf | blocks | graphs | edge GiB | build s | peak build GiB | cfg0 QPS | cfg0 recall | cfg0 exact ms/q | cfg0 graph ms/q | cfg1 QPS | cfg1 recall | cfg1 exact ms/q | cfg1 graph ms/q |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {int(row.leaf_size)} | {int(row.leaf_blocks)} | {int(row.graph_count_est)} | "
            f"{row.edge_gib:.3f} | {row.build_seconds:.1f} | {row.build_peak_memory_gib:.2f} | "
            f"{row.cfg0_qps:.0f} | {row.cfg0_recall:.5f} | {row.cfg0_exact_ms:.3f} | {row.cfg0_graph_ms:.3f} | "
            f"{row.cfg1_qps:.0f} | {row.cfg1_recall:.5f} | {row.cfg1_exact_ms:.3f} | {row.cfg1_graph_ms:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            (
                f"- Compared with leaf=1000, leaf={int(leaf_max.leaf_size)} reduces build time by "
                f"{-pct_delta(leaf_max.build_seconds, leaf_1000.build_seconds):.1f}% "
                f"and peak build memory by {-pct_delta(leaf_max.build_peak_memory_gib, leaf_1000.build_peak_memory_gib):.1f}%, "
                f"but cfg1 QPS drops by {-pct_delta(leaf_max.cfg1_qps, leaf_1000.cfg1_qps):.1f}%."
            ),
            (
                f"- The best cfg0 QPS is leaf={int(best_cfg0.leaf_size)} at {best_cfg0.best_qps:.0f} QPS "
                f"with recall {best_cfg0.recall_at_k:.5f}."
            ),
            (
                f"- The best cfg1 QPS is leaf={int(best_cfg1.leaf_size)} at {best_cfg1.best_qps:.0f} QPS "
                f"with recall {best_cfg1.recall_at_k:.5f}."
            ),
            (
                f"- Exact scan time grows from {leaf_min.cfg1_exact_ms:.3f} ms/query at leaf={int(leaf_min.leaf_size)} "
                f"to {leaf_max.cfg1_exact_ms:.3f} ms/query at leaf={int(leaf_max.leaf_size)}, matching the leaf_size * dim cost model."
            ),
            "- Filter violations are zero for all rows.",
            "",
            "## Generated Files",
            "",
            "- `derived/leaf_size_build.csv`",
            "- `derived/leaf_size_search.csv`",
            "- `figures/fig_leaf_size_build_resource.{pdf,png}`",
            "- `figures/fig_leaf_size_search_quality.{pdf,png}`",
            "- `figures/fig_leaf_size_search_breakdown.{pdf,png}`",
        ]
    )
    (run_dir / "leaf_size_tradeoff.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    derived_dir = Path(args.derived_dir) if args.derived_dir else run_dir / "derived"

    summary = load_csv(run_dir / "aggregate_summary.csv")
    sweep = load_csv(run_dir / "aggregate_sweep.csv")
    phase = load_csv(run_dir / "aggregate_phase_gpu.csv")
    validate(summary, sweep, phase)
    build, search = prepare(summary, sweep, phase)

    derived_dir.mkdir(parents=True, exist_ok=True)
    build.to_csv(derived_dir / "leaf_size_build.csv", index=False)
    search.to_csv(derived_dir / "leaf_size_search.csv", index=False)
    write_report(run_dir, build, search)

    plot_build(build, output_dir, args.format)
    plot_search(search, output_dir, args.format)
    plot_breakdown(search, output_dir, args.format)

    print(f"Loaded build rows: {len(build)}")
    print(f"Loaded search rows: {len(search)}")
    print(f"Wrote {run_dir / 'leaf_size_tradeoff.md'}")
    print(f"Wrote {derived_dir / 'leaf_size_build.csv'}")
    print(f"Wrote {derived_dir / 'leaf_size_search.csv'}")


if __name__ == "__main__":
    main()
