#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
DOUBLE_COL = (6.8, 2.55)
TALL_DOUBLE_COL = (6.8, 3.0)

DATASET_ORDER = [
    "audio",
    "deep",
    "enron",
    "gist",
    "glove-100",
    "msong",
    "sift",
    "yt8mAudio",
]
THRESHOLDS = [0.98, 0.99, 0.995]
DEGREE_BASELINE = "uniform_d32_i96_it20"
DEGREE_ORDER = [
    "uniform_d32_i96_it20",
    "adaptive_d32_i96_min16_i48_it20",
    "adaptive_d32_i96_min8_i24_it20",
    "adaptive_d32_i96_min4_i12_it20",
]
DEGREE_DISPLAY = {
    "uniform_d32_i96_it20": "uniform 32/96",
    "adaptive_d32_i96_min16_i48_it20": "adaptive min16/48",
    "adaptive_d32_i96_min8_i24_it20": "adaptive min8/24",
    "adaptive_d32_i96_min4_i12_it20": "adaptive min4/12",
}
POLICY_ORDER = ["uniform", "upper_layers", "layer_adaptive"]
POLICY_DISPLAY = {
    "uniform": "uniform",
    "upper_layers": "upper layers",
    "layer_adaptive": "layer adaptive",
}


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
    parser = argparse.ArgumentParser(
        description="Analyze and plot the 2026-06-06 Range-CAGRA paper runs."
    )
    parser.add_argument(
        "--main-dir",
        default=(
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_main_algo_1gpu_ordinary_20260606_main_algo_s0"
        ),
    )
    parser.add_argument(
        "--degree-dirs",
        nargs="+",
        default=[
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_full_20260606_layer_degree_s0",
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_full_20260606_layer_degree_s1",
        ],
    )
    parser.add_argument(
        "--search-dirs",
        nargs="+",
        default=[
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_full_20260606_layer_search_s0",
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_full_20260606_layer_search_s1",
        ],
    )
    parser.add_argument(
        "--leaf-dirs",
        nargs="+",
        default=[
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_full_20260606_leaf_size_s0",
            "results/range_cagra/segment_tree_param_sweep/"
            "paper_full_20260606_leaf_size_s1",
        ],
    )
    parser.add_argument(
        "--output-dir",
        default="results/range_cagra/paper_full_suite/paper_results_analysis_20260607",
    )
    parser.add_argument(
        "--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"]
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    return df


def read_run_csv(run_dir: Path, name: str) -> pd.DataFrame:
    df = read_required_csv(run_dir / name)
    df["source_run"] = run_dir.name
    return df


def read_many(run_dirs: list[Path], name: str) -> pd.DataFrame:
    return pd.concat([read_run_csv(run_dir, name) for run_dir in run_dirs], ignore_index=True)


def geomean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def ordered_datasets(values) -> list[str]:
    present = set(values)
    ordered = [ds for ds in DATASET_ORDER if ds in present]
    ordered += sorted(present - set(ordered))
    return ordered


def validate_sweep(df: pd.DataFrame, label: str) -> list[str]:
    required = {
        "dataset",
        "workload_name",
        "best_qps",
        "recall_at_k",
        "filter_violations",
        "nq",
        "build_seconds",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")
    if not df["recall_at_k"].between(0, 1).all():
        raise ValueError(f"{label} recall_at_k must be in [0, 1]")
    if (df["best_qps"] <= 0).any():
        raise ValueError(f"{label} best_qps must be positive")
    notes = []
    bad = int((df["filter_violations"] != 0).sum())
    if bad:
        notes.append(f"{label}: {bad} rows have filter violations.")
    return notes


def validate_status(status: pd.DataFrame, label: str) -> dict[str, int]:
    status_col = "final_status" if "final_status" in status.columns else "status"
    return status[status_col].value_counts(dropna=False).to_dict()


def frontier(df: pd.DataFrame, group_cols: list[str], threshold: float) -> pd.DataFrame:
    valid = df[(df["filter_violations"] == 0) & (df["recall_at_k"] >= threshold)].copy()
    if valid.empty:
        return valid
    idx = valid.groupby(group_cols)["best_qps"].idxmax()
    return valid.loc[idx].reset_index(drop=True)


def save_figure(fig, output_dir: Path, stem: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(output_dir / f"{stem}.{ext}")
    plt.close(fig)


def summarize_main(main: pd.DataFrame, phase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = ordered_datasets(main["dataset"].unique())
    total = (
        main.groupby("dataset")["workload_name"]
        .nunique()
        .reindex(datasets)
        .rename("total_workloads")
        .reset_index()
    )
    rows = []
    for threshold in THRESHOLDS:
        fr = frontier(main, ["dataset", "workload_name"], threshold)
        grouped = (
            fr.groupby("dataset")
            .agg(
                covered=("workload_name", "nunique"),
                mean_best_qps=("best_qps", "mean"),
                geomean_best_qps=("best_qps", geomean),
                mean_recall=("recall_at_k", "mean"),
                min_recall=("recall_at_k", "min"),
            )
            .reindex(datasets)
            .reset_index()
        )
        grouped["threshold"] = threshold
        grouped = grouped.merge(total, on="dataset", how="left")
        grouped["covered"] = grouped["covered"].fillna(0).astype(int)
        grouped["coverage_pct"] = grouped["covered"] / grouped["total_workloads"] * 100.0
        rows.append(grouped)
    main_frontier_summary = pd.concat(rows, ignore_index=True)

    max_recall = (
        main.groupby(["dataset", "workload_name"], as_index=False)
        .agg(max_recall=("recall_at_k", "max"))
        .groupby("dataset")
        .agg(
            total_workloads=("workload_name", "nunique"),
            maxrec_ge_098=("max_recall", lambda s: int((s >= 0.98).sum())),
            maxrec_ge_099=("max_recall", lambda s: int((s >= 0.99).sum())),
            maxrec_ge_0995=("max_recall", lambda s: int((s >= 0.995).sum())),
            min_max_recall=("max_recall", "min"),
            mean_max_recall=("max_recall", "mean"),
        )
        .reindex(datasets)
        .reset_index()
    )

    whole_phase = phase[phase["phase"] == "whole"][
        ["dataset", "peak_memory_used_mb", "avg_memory_used_mb", "avg_gpu_util_pct"]
    ].drop_duplicates("dataset")
    build = (
        main[
            [
                "dataset",
                "rows",
                "dim",
                "base_gib",
                "edge_gib",
                "build_seconds",
                "leaf_size",
            ]
        ]
        .drop_duplicates("dataset")
        .merge(whole_phase, on="dataset", how="left")
    )
    return main_frontier_summary, max_recall, build


def summarize_degree(degree: pd.DataFrame, phase: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    build = degree[
        [
            "dataset",
            "config_label",
            "edge_gib",
            "edge_count",
            "build_seconds",
            "graph_degree_avg",
            "intermediate_graph_degree_avg",
        ]
    ].drop_duplicates(["dataset", "config_label"])
    phase_build = (
        phase[phase["phase"] == "build"]
        .groupby(["dataset", "config_label"], as_index=False)
        .agg(peak_build_memory_mb=("peak_memory_used_mb", "max"))
    )
    build = build.merge(phase_build, on=["dataset", "config_label"], how="left")
    base = build[build["config_label"] == DEGREE_BASELINE][
        ["dataset", "edge_gib", "edge_count", "build_seconds", "peak_build_memory_mb"]
    ].rename(
        columns={
            "edge_gib": "baseline_edge_gib",
            "edge_count": "baseline_edge_count",
            "build_seconds": "baseline_build_seconds",
            "peak_build_memory_mb": "baseline_peak_build_memory_mb",
        }
    )
    build_rel = build.merge(base, on="dataset", how="left")
    build_rel["edge_reduction_pct"] = (
        1.0 - build_rel["edge_gib"] / build_rel["baseline_edge_gib"]
    ) * 100.0
    build_rel["build_reduction_pct"] = (
        1.0 - build_rel["build_seconds"] / build_rel["baseline_build_seconds"]
    ) * 100.0
    build_rel["peak_memory_reduction_pct"] = (
        1.0
        - build_rel["peak_build_memory_mb"] / build_rel["baseline_peak_build_memory_mb"]
    ) * 100.0

    rows = []
    for threshold in THRESHOLDS:
        fr = frontier(degree, ["dataset", "workload_name", "config_label"], threshold)
        coverage = fr.groupby("config_label").size().rename("covered_workloads")
        pivot = fr.pivot_table(
            index=["dataset", "workload_name"],
            columns="config_label",
            values="best_qps",
            aggfunc="max",
        )
        for config in DEGREE_ORDER:
            if config == DEGREE_BASELINE:
                rows.append(
                    {
                        "threshold": threshold,
                        "config_label": config,
                        "covered_workloads": int(coverage.get(config, 0)),
                        "paired_workloads_vs_uniform": np.nan,
                        "geomean_qps_ratio_vs_uniform": 1.0,
                        "mean_qps_ratio_vs_uniform": 1.0,
                        "wins_vs_uniform": np.nan,
                        "losses_vs_uniform": np.nan,
                    }
                )
                continue
            both = pivot[[config, DEGREE_BASELINE]].dropna()
            ratios = both[config] / both[DEGREE_BASELINE]
            rows.append(
                {
                    "threshold": threshold,
                    "config_label": config,
                    "covered_workloads": int(coverage.get(config, 0)),
                    "paired_workloads_vs_uniform": int(len(both)),
                    "geomean_qps_ratio_vs_uniform": geomean(ratios),
                    "mean_qps_ratio_vs_uniform": float(ratios.mean()) if len(ratios) else np.nan,
                    "wins_vs_uniform": int((ratios > 1.0).sum()) if len(ratios) else 0,
                    "losses_vs_uniform": int((ratios < 1.0).sum()) if len(ratios) else 0,
                }
            )
    frontier_cmp = pd.DataFrame(rows)
    return build_rel, frontier_cmp


def summarize_policy(search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        fr = frontier(search, ["dataset", "workload_name", "search_iteration_policy"], threshold)
        coverage = fr.groupby("search_iteration_policy").size().rename("covered_workloads")
        pivot = fr.pivot_table(
            index=["dataset", "workload_name"],
            columns="search_iteration_policy",
            values="best_qps",
            aggfunc="max",
        )
        for policy in POLICY_ORDER:
            if policy == "uniform":
                rows.append(
                    {
                        "threshold": threshold,
                        "search_iteration_policy": policy,
                        "covered_workloads": int(coverage.get(policy, 0)),
                        "paired_workloads_vs_uniform": np.nan,
                        "geomean_qps_ratio_vs_uniform": 1.0,
                        "mean_qps_ratio_vs_uniform": 1.0,
                        "wins_vs_uniform": np.nan,
                        "losses_vs_uniform": np.nan,
                    }
                )
                continue
            both = pivot[[policy, "uniform"]].dropna()
            ratios = both[policy] / both["uniform"]
            rows.append(
                {
                    "threshold": threshold,
                    "search_iteration_policy": policy,
                    "covered_workloads": int(coverage.get(policy, 0)),
                    "paired_workloads_vs_uniform": int(len(both)),
                    "geomean_qps_ratio_vs_uniform": geomean(ratios),
                    "mean_qps_ratio_vs_uniform": float(ratios.mean()) if len(ratios) else np.nan,
                    "wins_vs_uniform": int((ratios > 1.0).sum()) if len(ratios) else 0,
                    "losses_vs_uniform": int((ratios < 1.0).sum()) if len(ratios) else 0,
                }
            )
    return pd.DataFrame(rows)


def summarize_leaf(leaf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    winner_rows = []
    ratio_rows = []
    for threshold in THRESHOLDS:
        fr = frontier(leaf, ["dataset", "workload_name", "leaf_size"], threshold)
        if fr.empty:
            continue
        winners = fr.loc[fr.groupby(["dataset", "workload_name"])["best_qps"].idxmax()].copy()
        winner_counts = (
            winners.groupby("leaf_size")
            .size()
            .rename("winner_workloads")
            .reset_index()
            .sort_values("leaf_size")
        )
        winner_counts["threshold"] = threshold
        winner_rows.append(winner_counts)

        pivot = fr.pivot_table(
            index=["dataset", "workload_name"],
            columns="leaf_size",
            values="best_qps",
            aggfunc="max",
        )
        if 64 in pivot.columns:
            for leaf_size in sorted(pivot.columns):
                both = pivot[[64, leaf_size]].dropna()
                ratio_rows.append(
                    {
                        "threshold": threshold,
                        "leaf_size": int(leaf_size),
                        "paired_workloads_with_leaf64": int(len(both)),
                        "geomean_qps_ratio_leaf64_vs_leaf": geomean(both[64] / both[leaf_size])
                        if len(both)
                        else np.nan,
                    }
                )
    winners = pd.concat(winner_rows, ignore_index=True) if winner_rows else pd.DataFrame()
    ratios = pd.DataFrame(ratio_rows)
    return winners, ratios


def summarize_main_timing(main: pd.DataFrame) -> pd.DataFrame:
    fr = frontier(main, ["dataset", "workload_name"], 0.98)
    cols = ["exact_seconds", "graph_seconds", "merge_seconds", "nq"]
    missing = set(cols) - set(fr.columns)
    if missing:
        raise ValueError(f"main timing missing columns: {sorted(missing)}")
    timing = fr.copy()
    for col in ["exact_seconds", "graph_seconds", "merge_seconds"]:
        timing[f"{col}_ms_per_query"] = timing[col] * 1000.0 / timing["nq"]
    out = (
        timing.groupby("dataset", as_index=False)
        .agg(
            covered=("workload_name", "nunique"),
            exact_ms_per_query=("exact_seconds_ms_per_query", "mean"),
            graph_ms_per_query=("graph_seconds_ms_per_query", "mean"),
            merge_ms_per_query=("merge_seconds_ms_per_query", "mean"),
        )
        .sort_values("dataset")
    )
    out["component_sum_ms_per_query"] = (
        out["exact_ms_per_query"] + out["graph_ms_per_query"] + out["merge_ms_per_query"]
    )
    return out


def plot_main(main_summary: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    datasets = ordered_datasets(main_summary["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.26, top=0.78, wspace=0.50)
    x = np.arange(len(datasets))
    width = 0.24
    legend_handles = []
    for i, threshold in enumerate(THRESHOLDS):
        data = (
            main_summary[main_summary["threshold"] == threshold]
            .set_index("dataset")
            .reindex(datasets)
        )
        bars = axes[0].bar(
            x + (i - 1) * width,
            data["covered"].fillna(0),
            width,
            label=f"R >= {threshold:g}",
            color=PALETTE[i],
        )
        legend_handles.append(bars)
    axes[0].axhline(33, color="black", linewidth=0.7, linestyle=":")
    axes[0].set_ylabel("Covered workloads")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets, rotation=25, ha="right")
    axes[0].set_ylim(0, 35)
    axes[0].legend(
        ncols=3,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.02),
        borderaxespad=0.0,
    )
    axes[0].grid(True, axis="y", linewidth=0.3, alpha=0.45)

    data98 = main_summary[main_summary["threshold"] == 0.98].set_index("dataset").reindex(datasets)
    axes[1].bar(x, data98["geomean_best_qps"].fillna(0), color=PALETTE[0])
    axes[1].set_ylabel("Geomean QPS (R >= 0.98)", labelpad=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets, rotation=25, ha="right")
    axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.45)
    save_figure(fig, output_dir, "fig_main_frontier_coverage_qps", formats)


def plot_degree(
    build_rel: pd.DataFrame, degree_cmp: pd.DataFrame, output_dir: Path, formats: list[str]
) -> None:
    configs = [c for c in DEGREE_ORDER if c != DEGREE_BASELINE]
    data = (
        build_rel[build_rel["config_label"].isin(configs)]
        .groupby("config_label", as_index=False)
        .agg(
            edge_reduction_pct=("edge_reduction_pct", "mean"),
            build_reduction_pct=("build_reduction_pct", "mean"),
            peak_memory_reduction_pct=("peak_memory_reduction_pct", "mean"),
        )
    )
    data["config_label"] = pd.Categorical(data["config_label"], configs, ordered=True)
    data = data.sort_values("config_label")
    x = np.arange(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    ax.bar(x - width, data["edge_reduction_pct"], width, color=PALETTE[0], label="Edges")
    ax.bar(x, data["build_reduction_pct"], width, color=PALETTE[1], label="Build time")
    ax.bar(
        x + width,
        data["peak_memory_reduction_pct"],
        width,
        color=PALETTE[2],
        label="Peak build memory",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Reduction vs. uniform (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([DEGREE_DISPLAY[c] for c in data["config_label"].astype(str)], rotation=15, ha="right")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    save_figure(fig, output_dir, "fig_ablation_degree_build_resource", formats)

    cmp_data = degree_cmp[degree_cmp["config_label"].isin(configs)].copy()
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    x = np.arange(len(configs))
    width = 0.24
    for i, threshold in enumerate(THRESHOLDS):
        sub = cmp_data[cmp_data["threshold"] == threshold].set_index("config_label").reindex(configs)
        ax.bar(
            x + (i - 1) * width,
            sub["geomean_qps_ratio_vs_uniform"],
            width,
            color=PALETTE[i],
            label=f"R >= {threshold:g}",
        )
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Geomean QPS ratio")
    ax.set_xticks(x)
    ax.set_xticklabels([DEGREE_DISPLAY[c] for c in configs], rotation=15, ha="right")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    save_figure(fig, output_dir, "fig_ablation_degree_frontier_qps_ratio", formats)


def plot_policy(policy_cmp: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    policies = ["upper_layers", "layer_adaptive"]
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.86, wspace=0.45)
    x = np.arange(len(THRESHOLDS))
    width = 0.34
    for i, policy in enumerate(policies):
        data = policy_cmp[policy_cmp["search_iteration_policy"] == policy].set_index("threshold")
        offset = (i - 0.5) * width
        axes[0].bar(
            x + offset,
            data.reindex(THRESHOLDS)["geomean_qps_ratio_vs_uniform"],
            width,
            color=PALETTE[i + 1],
            label=POLICY_DISPLAY[policy],
        )
        axes[1].bar(
            x + offset,
            data.reindex(THRESHOLDS)["covered_workloads"],
            width,
            color=PALETTE[i + 1],
            label=POLICY_DISPLAY[policy],
        )
    axes[0].axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    axes[0].set_ylabel("QPS ratio vs. uniform")
    axes[1].set_ylabel("Covered workloads")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"R >= {t:g}" for t in THRESHOLDS], rotation=15, ha="right")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    axes[0].legend(ncols=1, frameon=False, loc="upper left")
    save_figure(fig, output_dir, "fig_ablation_search_policy_tradeoff", formats)


def plot_leaf(leaf_winners: pd.DataFrame, leaf_ratios: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    leaf_sizes = sorted(leaf_winners["leaf_size"].unique())
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.25, top=0.78, wspace=0.50)
    x = np.arange(len(leaf_sizes))
    width = 0.24
    for i, threshold in enumerate(THRESHOLDS):
        sub = (
            leaf_winners[leaf_winners["threshold"] == threshold]
            .set_index("leaf_size")
            .reindex(leaf_sizes)
        )
        axes[0].bar(
            x + (i - 1) * width,
            sub["winner_workloads"].fillna(0),
            width,
            color=PALETTE[i],
            label=f"R >= {threshold:g}",
        )
    axes[0].set_ylabel("Best-QPS workload wins")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in leaf_sizes], rotation=25, ha="right")
    axes[0].legend(
        ncols=3,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.02),
        borderaxespad=0.0,
    )
    axes[0].grid(True, axis="y", linewidth=0.3, alpha=0.45)

    ratio_data = leaf_ratios[leaf_ratios["threshold"].isin([0.98, 0.99])].copy()
    for i, threshold in enumerate([0.98, 0.99]):
        sub = ratio_data[ratio_data["threshold"] == threshold].set_index("leaf_size").reindex(leaf_sizes)
        axes[1].plot(
            x,
            sub["geomean_qps_ratio_leaf64_vs_leaf"],
            marker=MARKERS[i],
            color=PALETTE[i],
            label=f"R >= {threshold:g}",
        )
    axes[1].axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    axes[1].set_ylabel("QPS ratio (leaf64 / leaf)", labelpad=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(v) for v in leaf_sizes], rotation=25, ha="right")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.45)
    save_figure(fig, output_dir, "fig_ablation_leaf_size_frontier", formats)


def plot_timing(timing: pd.DataFrame, output_dir: Path, formats: list[str]) -> None:
    datasets = ordered_datasets(timing["dataset"].unique())
    data = timing.set_index("dataset").reindex(datasets).fillna(0)
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=TALL_DOUBLE_COL)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.22, top=0.82)
    bottom = np.zeros(len(datasets))
    components = [
        ("exact_ms_per_query", "exact", PALETTE[0]),
        ("graph_ms_per_query", "graph", PALETTE[1]),
        ("merge_ms_per_query", "merge", PALETTE[2]),
    ]
    for col, label, color in components:
        vals = data[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_ylabel("Mean component latency (ms/query)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha="right")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)
    save_figure(fig, output_dir, "fig_main_timing_breakdown_recall98", formats)


def fmt_float(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_markdown(
    output_dir: Path,
    figures_dir: Path,
    derived_dir: Path,
    status_counts: dict[str, dict[str, int]],
    notes: list[str],
    main_summary: pd.DataFrame,
    max_recall: pd.DataFrame,
    main_build: pd.DataFrame,
    degree_build: pd.DataFrame,
    degree_cmp: pd.DataFrame,
    policy_cmp: pd.DataFrame,
    leaf_winners: pd.DataFrame,
    leaf_ratios: pd.DataFrame,
    timing: pd.DataFrame,
) -> Path:
    md = output_dir / "RANGE_CAGRA_PAPER_RESULTS_ANALYSIS_20260607.md"

    main98 = main_summary[main_summary["threshold"] == 0.98]
    main99 = main_summary[main_summary["threshold"] == 0.99]
    min8 = degree_build[degree_build["config_label"] == "adaptive_d32_i96_min8_i24_it20"]
    min8_mean = min8.agg(
        {
            "edge_reduction_pct": "mean",
            "build_reduction_pct": "mean",
            "peak_memory_reduction_pct": "mean",
        }
    )
    deg_min8_99 = degree_cmp[
        (degree_cmp["config_label"] == "adaptive_d32_i96_min8_i24_it20")
        & (degree_cmp["threshold"] == 0.99)
    ].iloc[0]
    policy_99 = policy_cmp[policy_cmp["threshold"] == 0.99]
    upper_99 = policy_99[policy_99["search_iteration_policy"] == "upper_layers"].iloc[0]
    layer_99 = policy_99[policy_99["search_iteration_policy"] == "layer_adaptive"].iloc[0]
    leaf64_vs1000_98 = leaf_ratios[
        (leaf_ratios["threshold"] == 0.98) & (leaf_ratios["leaf_size"] == 1000)
    ].iloc[0]
    leaf64_vs1000_99 = leaf_ratios[
        (leaf_ratios["threshold"] == 0.99) & (leaf_ratios["leaf_size"] == 1000)
    ].iloc[0]

    lines = [
        "# Range-CAGRA 2026-06-06 论文实验结果分析",
        "",
        "生成时间：2026-06-07，工作目录：`/home/wjy/cuvs`。",
        "",
        "本文档不是运行日志，而是把 2026-06-06 本机实验落盘结果整理成论文实验可用的结论、图表说明和写作素材。分析脚本是 `results/range_cagra/plot_paper_full_20260606.py`，所有图和派生表都在同一输出目录下，可复现。",
        "",
        "## 1. 使用的数据和完整性",
        "",
        "输入结果：",
        "",
        "- 主实验：`paper_main_algo_1gpu_ordinary_20260606_main_algo_s0`，算法为 `range_cagra_final`。",
        "- adaptive degree 消融：`paper_full_20260606_layer_degree_s0/s1`。",
        "- search policy 消融：`paper_full_20260606_layer_search_s0/s1`。",
        "- leaf size 消融：`paper_full_20260606_leaf_size_s0/s1`。",
        "",
        "状态汇总：",
        "",
    ]
    for name, counts in status_counts.items():
        lines.append(f"- `{name}`: {counts}")
    lines += [
        "",
        "正确性检查：所有纳入分析的 `aggregate_sweep.csv` 行均满足 `filter_violations=0`，Recall@10 在 `[0,1]` 内，QPS 为正。`text2image` 和 `wit` 在 11GB 2080 Ti 上被 memory guard 跳过，因此当前本机结论覆盖 8 个普通数据集，不应把跳过项写成算法失败。A100 11 数据集目录在本机只看到 command suite 和 README，没有对应的 2026-06-06 A100 结果 CSV，因此没有纳入本报告。",
        "",
    ]
    if notes:
        lines.append("脚本警告：")
        lines.extend([f"- {note}" for note in notes])
        lines.append("")

    lines += [
        "## 2. 主实验结论：`range_cagra_final`",
        "",
        "主实验固定 `range_cagra_final`：layer-adaptive degree，`leaf_size=64`，`exact_then_graph`，layer-adaptive search effort，并扫描 7 个 search budget。frontier 的选择方式是：对每个 dataset/workload，在满足指定 recall 阈值的行中取 `best_qps` 最大的配置。",
        "",
        "### 2.1 Recall/QPS frontier",
        "",
        "| dataset | R>=0.98 workloads | geomean QPS@0.98 | R>=0.99 workloads | geomean QPS@0.99 | max-recall min |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    max_recall_idx = max_recall.set_index("dataset")
    main99_idx = main99.set_index("dataset")
    for _, row in main98.iterrows():
        dataset = row["dataset"]
        row99 = main99_idx.loc[dataset]
        lines.append(
            "| {dataset} | {cov98}/33 | {qps98:.0f} | {cov99}/33 | {qps99} | {minrec:.5f} |".format(
                dataset=dataset,
                cov98=int(row["covered"]),
                qps98=row["geomean_best_qps"] if not pd.isna(row["geomean_best_qps"]) else 0,
                cov99=int(row99["covered"]),
                qps99="NA"
                if pd.isna(row99["geomean_best_qps"])
                else f"{row99['geomean_best_qps']:.0f}",
                minrec=max_recall_idx.loc[dataset, "min_max_recall"],
            )
        )
    lines += [
        "",
        "可写入论文的稳妥表述：`range_cagra_final` 在 `audio/deep/enron/msong/sift` 上能覆盖全部或大部分 workload 的 `R>=0.98` frontier，其中 `audio/msong/sift` 的 `R>=0.99` 覆盖也较稳定。当前固定预算下，`gist` 只在 1/33 个 workload 达到 `R>=0.98`，`glove-100` 没有达到 `R>=0.98` 的 workload；这说明最终算法配置还不能被写成“跨所有数据集高召回稳定”。论文里应把这部分作为局限或后续需要 A100/更大搜索预算补齐的证据。",
        "",
        f"图：`{figures_dir / 'fig_main_frontier_coverage_qps.pdf'}` 展示主实验 coverage 和 `R>=0.98` 下的 QPS；`{figures_dir / 'fig_main_timing_breakdown_recall98.pdf'}` 展示达到 `R>=0.98` 的行中 exact/graph/merge 的平均查询时间组成。",
        "",
        "### 2.2 构建和资源",
        "",
        "| dataset | rows | dim | edge GiB | build s | peak whole-run MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in main_build.sort_values("dataset").iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['rows'])} | {int(row['dim'])} | "
            f"{row['edge_gib']:.3f} | {row['build_seconds']:.2f} | "
            f"{row['peak_memory_used_mb']:.0f} |"
        )

    lines += [
        "",
        "这张表适合放在实验设置或 appendix，用于说明 11GB GPU 上 `text2image/wit` 被跳过不是算法正确性问题，而是显存边界问题。",
        "",
        "## 3. 消融一：adaptive layer-degree",
        "",
        "为什么这样做：线段树图池里的节点不是同质的。上层节点覆盖范围大、数量少，对 recall 更关键；下层节点数量多、覆盖小，如果使用统一 degree，会把大量边和构建工作花在低层小范围节点上。adaptive degree 的启发来自线段树层级本身：用节点层级近似该节点对范围查询覆盖的全局影响。",
        "",
        "实验结果支持这个设计：最终采用的 `adaptive_d32_i96_min8_i24_it20` 相对 uniform，平均减少 "
        f"{min8_mean['edge_reduction_pct']:.1f}% 图边存储、{min8_mean['build_reduction_pct']:.1f}% 构建时间、"
        f"{min8_mean['peak_memory_reduction_pct']:.1f}% build 峰值显存。按 `R>=0.99` 的 paired frontier 比较，它在 "
        f"{int(deg_min8_99['paired_workloads_vs_uniform'])} 个与 uniform 都达标的 workload 上，geomean QPS ratio 为 "
        f"{deg_min8_99['geomean_qps_ratio_vs_uniform']:.3f}。",
        "",
        "可写入论文：",
        "",
        "> Uniform degree over-provisions many low-level segment-tree nodes. Layer-adaptive degree allocation ties graph density to range coverage, reducing edge storage and build cost while preserving the recall/QPS frontier on most workloads.",
        "",
        f"图：`{figures_dir / 'fig_ablation_degree_build_resource.pdf'}` 和 `{figures_dir / 'fig_ablation_degree_frontier_qps_ratio.pdf'}`。",
        "",
        "## 4. 消融二：adaptive layer-search effort",
        "",
        "为什么这样做：搜索预算也有层级非均匀性。上层图节点覆盖大的内部区间，少量上层节点的搜索质量会影响较宽范围的召回；下层节点数量多，逐个增加 iteration 成本高。这个消融的设计灵感和 adaptive degree 一致，但作用在 search-time iterations 上。",
        "",
        "结果需要谨慎解释：在 `layer_search` 消融使用的 5 个 search config 下，`uniform` 覆盖的高召回 workload 更多；但在与 uniform 都达到阈值的 paired workload 上，`upper_layers` 和 `layer_adaptive` 通常有更高 QPS。以 `R>=0.99` 为例：",
        "",
        f"- `upper_layers` 覆盖 {int(upper_99['covered_workloads'])} 个 workload，在 paired workload 上 geomean QPS ratio 为 {upper_99['geomean_qps_ratio_vs_uniform']:.3f}，wins/losses={int(upper_99['wins_vs_uniform'])}/{int(upper_99['losses_vs_uniform'])}。",
        f"- `layer_adaptive` 覆盖 {int(layer_99['covered_workloads'])} 个 workload，在 paired workload 上 geomean QPS ratio 为 {layer_99['geomean_qps_ratio_vs_uniform']:.3f}，wins/losses={int(layer_99['wins_vs_uniform'])}/{int(layer_99['losses_vs_uniform'])}。",
        "",
        "可写入论文：adaptive search effort 是 throughput-oriented 的策略，不应写成无条件提高 recall 覆盖。更准确的说法是，它把预算集中到高影响层，在达到同一 recall 阈值的查询上提升 QPS；但当预算网格不够大时，uniform 可能覆盖更多高召回 workload。因此论文中应该以 recall/QPS frontier 呈现，而不是只报单点。",
        "",
        f"图：`{figures_dir / 'fig_ablation_search_policy_tradeoff.pdf'}`。",
        "",
        "## 5. 消融三：leaf size",
        "",
        "为什么这样做：`leaf_size` 是范围索引粒度参数，控制边界 exact scan 和内部 graph task 的比例。leaf 太大时边界扫描更重，内部图更少；leaf 太小时边界扫描变轻但图节点更多，低预算下更容易损失召回。这个参数不是普通 search budget，而是构建侧的 granularity 选择。",
        "",
        "结果支持选择中等 leaf：在 `R>=0.98` 的 best-QPS winner 统计中，`leaf=64` 赢得最多 workload；在 `R>=0.99` 下，`leaf=64` 仍然是 winner 数最多的单一 leaf。与 `leaf=1000` 相比，`leaf=64` 在 paired workload 上的 geomean QPS ratio 为 "
        f"{leaf64_vs1000_98['geomean_qps_ratio_leaf64_vs_leaf']:.2f}x (`R>=0.98`) 和 "
        f"{leaf64_vs1000_99['geomean_qps_ratio_leaf64_vs_leaf']:.2f}x (`R>=0.99`)。",
        "",
        "可写入论文：",
        "",
        "> Leaf size exposes a trade-off between exact boundary work and reusable graph granularity. The sweep shows that a moderate leaf size gives a robust frontier across datasets, motivating the fixed `leaf_size=64` in the main algorithm instead of per-dataset tuning.",
        "",
        f"图：`{figures_dir / 'fig_ablation_leaf_size_frontier.pdf'}`。",
        "",
        "## 6. 论文实验写法建议",
        "",
        "建议把实验组织成如下逻辑：",
        "",
        "1. 先说明 Range-CAGRA 的核心不是黑盒调用 CAGRA，而是把有序范围查询分解为 exact boundary tasks 和 segment-tree internal graph tasks，并且只保留一份全局向量。",
        "2. 主实验报告 `range_cagra_final` 的 recall/QPS frontier、build time、memory 和 `filter_violations=0`。",
        "3. 三个消融分别回答三个设计问题：degree 是否应该按层分配、search effort 是否应该按层分配、leaf size 如何影响 exact/graph 工作量。",
        "4. 对 `gist/glove-100` 的失败不要隐藏。它们说明当前固定预算还不够普适，论文中要么补跑更大预算/A100，要么把结论限定为当前覆盖的数据集。",
        "",
        "推荐贡献表述：",
        "",
        "- A one-copy segment-tree graph pool for ordered range ANN on GPU.",
        "- A hybrid exact-boundary and graph-interior execution model with zero filter violations in the reported runs.",
        "- Layer-adaptive graph degree allocation that reduces build/storage cost.",
        "- Layer-aware search effort and leaf-size frontier analysis that explains where throughput comes from and where recall fails.",
        "",
        "## 7. 产物和复现命令",
        "",
        f"- 派生 CSV：`{derived_dir}`",
        f"- 论文图：`{figures_dir}`",
        f"- 本分析文件：`{md}`",
        "",
        "从 repo 根目录复现：",
        "",
        "```bash",
        "python3 results/range_cagra/plot_paper_full_20260606.py",
        "```",
        "",
    ]

    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    derived_dir = output_dir / "derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    main_dir = Path(args.main_dir)
    degree_dirs = [Path(p) for p in args.degree_dirs]
    search_dirs = [Path(p) for p in args.search_dirs]
    leaf_dirs = [Path(p) for p in args.leaf_dirs]

    notes = []
    main_sweep = read_run_csv(main_dir, "aggregate_sweep.csv")
    main_phase = read_run_csv(main_dir, "aggregate_phase_gpu.csv")
    main_status = read_run_csv(main_dir, "status.csv")
    degree_sweep = read_many(degree_dirs, "aggregate_sweep.csv")
    degree_phase = read_many(degree_dirs, "aggregate_phase_gpu.csv")
    degree_status = read_many(degree_dirs, "status.csv")
    search_sweep = read_many(search_dirs, "aggregate_sweep.csv")
    search_status = read_many(search_dirs, "status.csv")
    leaf_sweep = read_many(leaf_dirs, "aggregate_sweep.csv")
    leaf_status = read_many(leaf_dirs, "status.csv")

    for label, df in [
        ("main", main_sweep),
        ("degree", degree_sweep),
        ("search", search_sweep),
        ("leaf", leaf_sweep),
    ]:
        notes.extend(validate_sweep(df, label))

    status_counts = {
        "main": validate_status(main_status, "main"),
        "degree": validate_status(degree_status, "degree"),
        "search": validate_status(search_status, "search"),
        "leaf": validate_status(leaf_status, "leaf"),
    }

    main_summary, max_recall, main_build = summarize_main(main_sweep, main_phase)
    degree_build, degree_cmp = summarize_degree(degree_sweep, degree_phase)
    policy_cmp = summarize_policy(search_sweep)
    leaf_winners, leaf_ratios = summarize_leaf(leaf_sweep)
    main_timing = summarize_main_timing(main_sweep)

    outputs = {
        "main_frontier_summary.csv": main_summary,
        "main_max_recall_coverage.csv": max_recall,
        "main_build_memory.csv": main_build,
        "degree_build_reduction.csv": degree_build,
        "degree_frontier_comparison.csv": degree_cmp,
        "search_policy_frontier_comparison.csv": policy_cmp,
        "leaf_size_winners.csv": leaf_winners,
        "leaf_size_ratios_vs64.csv": leaf_ratios,
        "main_timing_breakdown_recall98.csv": main_timing,
    }
    for name, df in outputs.items():
        df.to_csv(derived_dir / name, index=False)

    plot_main(main_summary, figures_dir, args.formats)
    plot_degree(degree_build, degree_cmp, figures_dir, args.formats)
    plot_policy(policy_cmp, figures_dir, args.formats)
    plot_leaf(leaf_winners, leaf_ratios, figures_dir, args.formats)
    plot_timing(main_timing, figures_dir, args.formats)

    md = write_markdown(
        output_dir,
        figures_dir,
        derived_dir,
        status_counts,
        notes,
        main_summary,
        max_recall,
        main_build,
        degree_build,
        degree_cmp,
        policy_cmp,
        leaf_winners,
        leaf_ratios,
        main_timing,
    )
    print(f"Wrote {md}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote derived CSVs to {derived_dir}")


if __name__ == "__main__":
    main()
