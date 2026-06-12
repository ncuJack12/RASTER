#!/usr/bin/env python3
"""Analyze the 2026-06-07 Range-CAGRA throughput-at-recall90 suite."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


RECALL_TARGET = 0.90


RUNS = {
    "throughput90_full_20260607_leaf_fast_s0_no_leaf8": ("leaf", "s0"),
    "throughput90_full_20260607_leaf_fast_s1": ("leaf", "s1"),
    "throughput90_full_20260607_degree_fast_s0": ("degree", "s0"),
    "throughput90_full_20260607_degree_fast_s1": ("degree", "s1"),
    "throughput90_full_20260607_policy_fast_s0": ("policy", "s0"),
    "throughput90_full_20260607_policy_fast_s1": ("policy", "s1"),
    "throughput90_full_20260607_schedule_fast_s0": ("schedule_internal", "s0"),
    "throughput90_full_20260607_schedule_fast_s1": ("schedule_internal", "s1"),
}


NUMERIC_COLUMNS = [
    "leaf_size",
    "topk",
    "rows",
    "dim",
    "target_width",
    "leaf_blocks",
    "graph_count_est",
    "edge_gib_est",
    "exact_vectors_est",
    "exact_dim_work_est",
    "leaf_dim_work",
    "nq",
    "edge_count",
    "base_gib",
    "edge_gib",
    "build_seconds",
    "search_config_id",
    "ef",
    "graph_iterations",
    "entry_count",
    "search_repeats",
    "search_iteration_base_graph_iterations",
    "search_iteration_min_graph_iterations",
    "search_iteration_max_graph_iterations",
    "search_iteration_avg_graph_iterations",
    "search_iteration_max_graph_layer",
    "search_iteration_override_graph_count",
    "graph_degree_min",
    "graph_degree_max",
    "graph_degree_avg",
    "intermediate_graph_degree_min",
    "intermediate_graph_degree_max",
    "intermediate_graph_degree_avg",
    "best_search_seconds",
    "avg_search_seconds",
    "best_qps",
    "avg_qps",
    "recall_at_k",
    "filter_violations",
    "exact_seconds",
    "graph_seconds",
    "merge_seconds",
    "exact_vectors_scanned",
    "graph_node_tasks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/range_cagra/segment_tree_param_sweep"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/range_cagra/paper_full_suite/throughput90_full_20260607_analysis"),
    )
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_sweep_rows(result_root: Path) -> pd.DataFrame:
    frames = []
    for run_id, (phase, shard) in RUNS.items():
        path = result_root / run_id / "aggregate_sweep.csv"
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        df["run_id"] = run_id
        df["phase"] = phase
        df["shard"] = shard
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return coerce_numeric(pd.concat(frames, ignore_index=True))


def load_summary_rows(result_root: Path) -> pd.DataFrame:
    frames = []
    for run_id, (phase, shard) in RUNS.items():
        path = result_root / run_id / "aggregate_summary.csv"
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        df["run_id"] = run_id
        df["phase"] = phase
        df["shard"] = shard
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return coerce_numeric(pd.concat(frames, ignore_index=True))


def load_status_rows(result_root: Path) -> pd.DataFrame:
    frames = []
    for run_id, (phase, shard) in RUNS.items():
        path = result_root / run_id / "status.csv"
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        df["run_id"] = run_id
        df["phase"] = phase
        df["shard"] = shard
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return coerce_numeric(pd.concat(frames, ignore_index=True))


def pick_best(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(["best_qps", "recall_at_k"], ascending=[False, False])
        .drop_duplicates(keys, keep="first")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def pick_max_recall(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(["recall_at_k", "best_qps"], ascending=[False, False])
        .drop_duplicates(keys, keep="first")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def mode_text(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    counts = values.value_counts()
    return str(counts.index[0])


def winner_counts(best: pd.DataFrame, col: str, out_path: Path, keys: list[str] | None = None) -> pd.DataFrame:
    if best.empty or col not in best.columns:
        return pd.DataFrame()
    group_keys = keys or [col]
    out = (
        best.groupby(group_keys, dropna=False)
        .agg(
            wins=("workload_name", "count"),
            median_qps=("best_qps", "median"),
            mean_qps=("best_qps", "mean"),
            median_recall=("recall_at_k", "median"),
            min_recall=("recall_at_k", "min"),
        )
        .reset_index()
        .sort_values(["wins", "median_qps"], ascending=[False, False])
    )
    out.to_csv(out_path, index=False)
    return out


def quantile(p: float):
    def inner(s: pd.Series) -> float:
        return float(s.quantile(p))

    inner.__name__ = f"p{int(p * 100)}"
    return inner


def add_time_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    denom = df[["exact_seconds", "graph_seconds", "merge_seconds"]].sum(axis=1, min_count=1)
    for col in ["exact_seconds", "graph_seconds", "merge_seconds"]:
        share_col = col.replace("_seconds", "_share")
        df[share_col] = np.where(denom > 0, df[col] / denom, np.nan)
    return df


def format_int(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{int(round(float(x))):,}"


def format_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def format_qps(x: float) -> str:
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.1f}k"
    return f"{x:.0f}"


def md_table(df: pd.DataFrame, columns: list[tuple[str, str]], max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["No rows."]
    rows = df.head(max_rows)
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in rows.iterrows():
        cells = []
        for key, _ in columns:
            val = row.get(key, "")
            if isinstance(val, float):
                cells.append(format_float(val, 3))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = load_sweep_rows(args.result_root)
    summary = load_summary_rows(args.result_root)
    status = load_status_rows(args.result_root)

    if sweep.empty:
        raise SystemExit("no aggregate_sweep rows found")

    sweep = add_time_shares(sweep)
    valid = sweep[
        (sweep["recall_at_k"] >= RECALL_TARGET)
        & (sweep["filter_violations"].fillna(0) == 0)
        & (sweep["best_qps"] > 0)
    ].copy()

    keys = ["dataset", "workload_name"]
    comparable = sweep[(sweep["filter_violations"].fillna(0) == 0) & (sweep["best_qps"] > 0)].copy()
    best_attempt_by_recall = pick_max_recall(comparable, keys)
    best_all = pick_best(valid, keys)
    best_no_schedule = pick_best(valid[valid["phase"] != "schedule_internal"], keys)

    # Per-phase best tables.
    leaf = valid[valid["phase"] == "leaf"].copy()
    degree = valid[valid["phase"] == "degree"].copy()
    policy = valid[valid["phase"] == "policy"].copy()
    schedule = valid[valid["phase"] == "schedule_internal"].copy()

    leaf_by_size = pick_best(leaf, keys + ["leaf_size"])
    leaf_winners = pick_best(leaf_by_size, keys)

    degree_by_config = pick_best(degree, keys + ["config_label"])
    degree_winners = pick_best(degree_by_config, keys)

    policy_by_policy = pick_best(policy, keys + ["search_iteration_policy"])
    policy_winners = pick_best(policy_by_policy, keys)

    schedule_by_schedule = pick_best(schedule, keys + ["search_schedule"])
    schedule_winners = pick_best(schedule_by_schedule, keys)

    dataset_summary = (
        best_all.groupby("dataset")
        .agg(
            workloads=("workload_name", "nunique"),
            median_qps=("best_qps", "median"),
            mean_qps=("best_qps", "mean"),
            p10_qps=("best_qps", quantile(0.10)),
            max_qps=("best_qps", "max"),
            median_recall=("recall_at_k", "median"),
            min_recall=("recall_at_k", "min"),
            median_leaf=("leaf_size", "median"),
            mode_leaf=("leaf_size", mode_text),
            mode_schedule=("search_schedule", mode_text),
            mode_policy=("search_iteration_policy", mode_text),
            median_exact_share=("exact_share", "median"),
            median_graph_share=("graph_share", "median"),
            median_merge_share=("merge_share", "median"),
        )
        .reset_index()
        .sort_values("median_qps", ascending=False)
    )
    dataset_summary_csv = dataset_summary.copy()
    dataset_summary_csv.to_csv(args.output_dir / "dataset_summary_recall90.csv", index=False)

    best_all.to_csv(args.output_dir / "best_recall90_per_workload.csv", index=False)
    best_no_schedule.to_csv(args.output_dir / "best_recall90_per_workload_no_schedule.csv", index=False)
    valid.to_csv(args.output_dir / "valid_configs_recall90.csv", index=False)
    sweep.to_csv(args.output_dir / "merged_all_sweep.csv", index=False)

    valid_key_index = pd.MultiIndex.from_frame(best_all[keys]) if not best_all.empty else pd.MultiIndex.from_arrays([[], []])
    attempted_key_index = pd.MultiIndex.from_frame(best_attempt_by_recall[keys])
    below_target = best_attempt_by_recall[~attempted_key_index.isin(valid_key_index)].copy()
    below_target.to_csv(args.output_dir / "workloads_below_recall90_best_attempt.csv", index=False)

    attempted_counts = best_attempt_by_recall.groupby("dataset").agg(attempted_workloads=("workload_name", "nunique"))
    valid_counts = best_all.groupby("dataset").agg(valid_workloads=("workload_name", "nunique"))
    coverage_summary = attempted_counts.join(valid_counts, how="left").fillna(0).reset_index()
    coverage_summary["valid_workloads"] = coverage_summary["valid_workloads"].astype(int)
    coverage_summary["below_recall90_workloads"] = (
        coverage_summary["attempted_workloads"] - coverage_summary["valid_workloads"]
    ).astype(int)
    coverage_summary["coverage_rate"] = coverage_summary["valid_workloads"] / coverage_summary["attempted_workloads"]
    coverage_summary = coverage_summary.sort_values(["coverage_rate", "dataset"], ascending=[False, True])
    coverage_summary.to_csv(args.output_dir / "coverage_summary_recall90.csv", index=False)

    if not summary.empty:
        summary.to_csv(args.output_dir / "merged_all_summary.csv", index=False)
    if not status.empty:
        status.to_csv(args.output_dir / "merged_status.csv", index=False)
        (
            status.groupby(["phase", "dataset", "final_status", "final_reason"], dropna=False)
            .size()
            .reset_index(name="tasks")
            .sort_values(["phase", "dataset", "final_status", "tasks"], ascending=[True, True, True, False])
            .to_csv(args.output_dir / "status_summary.csv", index=False)
        )

    leaf_by_size_summary = (
        leaf_by_size.groupby(["dataset", "leaf_size"])
        .agg(
            valid_workloads=("workload_name", "count"),
            median_qps=("best_qps", "median"),
            mean_qps=("best_qps", "mean"),
            median_recall=("recall_at_k", "median"),
            min_recall=("recall_at_k", "min"),
            median_exact_vectors=("exact_vectors_scanned", "median"),
            median_graph_tasks=("graph_node_tasks", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "leaf_size"])
    )
    leaf_by_size_summary.to_csv(args.output_dir / "leaf_ablation_summary.csv", index=False)
    leaf_winners.to_csv(args.output_dir / "leaf_winners_per_workload.csv", index=False)
    leaf_winner_counts = winner_counts(leaf_winners, "leaf_size", args.output_dir / "leaf_winner_counts.csv")

    degree_by_config_summary = (
        degree_by_config.groupby(["dataset", "config_label"])
        .agg(
            valid_workloads=("workload_name", "count"),
            median_qps=("best_qps", "median"),
            mean_qps=("best_qps", "mean"),
            median_recall=("recall_at_k", "median"),
            min_recall=("recall_at_k", "min"),
            median_edge_gib=("edge_gib", "median"),
            median_graph_degree=("graph_degree_avg", "median"),
            median_intermediate_degree=("intermediate_graph_degree_avg", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "median_qps"], ascending=[True, False])
    )
    degree_by_config_summary.to_csv(args.output_dir / "degree_ablation_summary.csv", index=False)
    degree_winners.to_csv(args.output_dir / "degree_winners_per_workload.csv", index=False)
    degree_winner_counts = winner_counts(degree_winners, "config_label", args.output_dir / "degree_winner_counts.csv")

    policy_by_policy_summary = (
        policy_by_policy.groupby(["dataset", "search_iteration_policy"])
        .agg(
            valid_workloads=("workload_name", "count"),
            median_qps=("best_qps", "median"),
            mean_qps=("best_qps", "mean"),
            median_recall=("recall_at_k", "median"),
            min_recall=("recall_at_k", "min"),
            median_graph_iterations=("search_iteration_avg_graph_iterations", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "median_qps"], ascending=[True, False])
    )
    policy_by_policy_summary.to_csv(args.output_dir / "policy_ablation_summary.csv", index=False)
    policy_winners.to_csv(args.output_dir / "policy_winners_per_workload.csv", index=False)
    policy_winner_counts = winner_counts(
        policy_winners,
        "search_iteration_policy",
        args.output_dir / "policy_winner_counts.csv",
    )

    schedule_by_schedule_summary = (
        schedule_by_schedule.groupby(["dataset", "search_schedule"])
        .agg(
            valid_workloads=("workload_name", "count"),
            median_qps=("best_qps", "median"),
            mean_qps=("best_qps", "mean"),
            median_recall=("recall_at_k", "median"),
            min_recall=("recall_at_k", "min"),
        )
        .reset_index()
        .sort_values(["dataset", "median_qps"], ascending=[True, False])
    )
    schedule_by_schedule_summary.to_csv(args.output_dir / "schedule_internal_summary.csv", index=False)
    schedule_winners.to_csv(args.output_dir / "schedule_winners_per_workload.csv", index=False)
    schedule_winner_counts = winner_counts(
        schedule_winners,
        "search_schedule",
        args.output_dir / "schedule_winner_counts.csv",
    )

    # Pairwise comparisons.
    degree_pivot = degree_by_config.pivot_table(
        index=keys,
        columns="config_label",
        values="best_qps",
        aggfunc="max",
    )
    if "uniform_d32_i96_it20" in degree_pivot.columns:
        adaptive_cols = [c for c in degree_pivot.columns if str(c).startswith("adaptive")]
        degree_pivot["best_adaptive_qps"] = degree_pivot[adaptive_cols].max(axis=1)
        degree_pivot["adaptive_vs_uniform_speedup"] = (
            degree_pivot["best_adaptive_qps"] / degree_pivot["uniform_d32_i96_it20"]
        )
        degree_pivot.reset_index().to_csv(args.output_dir / "degree_adaptive_vs_uniform.csv", index=False)

    policy_pivot = policy_by_policy.pivot_table(
        index=keys,
        columns="search_iteration_policy",
        values="best_qps",
        aggfunc="max",
    )
    if {"layer_adaptive", "uniform"}.issubset(set(policy_pivot.columns)):
        policy_pivot["layer_adaptive_vs_uniform_speedup"] = (
            policy_pivot["layer_adaptive"] / policy_pivot["uniform"]
        )
        policy_pivot.reset_index().to_csv(args.output_dir / "policy_layer_adaptive_vs_uniform.csv", index=False)

    schedule_pivot = schedule_by_schedule.pivot_table(
        index=keys,
        columns="search_schedule",
        values="best_qps",
        aggfunc="max",
    )
    if "exact_then_graph" in schedule_pivot.columns:
        schedule_cols = [c for c in schedule_pivot.columns if c in {"exact_then_graph", "graph_then_exact", "overlap"}]
        schedule_pivot["best_schedule_qps"] = schedule_pivot[schedule_cols].max(axis=1)
        schedule_pivot["best_vs_exact_then_graph_speedup"] = (
            schedule_pivot["best_schedule_qps"] / schedule_pivot["exact_then_graph"]
        )
        schedule_pivot.reset_index().to_csv(args.output_dir / "schedule_internal_vs_exact_then_graph.csv", index=False)

    skipped_datasets = []
    if not status.empty:
        by_dataset = status.groupby("dataset").agg(
            tasks=("dataset", "count"),
            done=("final_status", lambda s: int((s == "done").sum())),
            skip=("final_status", lambda s: int((s == "skip").sum())),
            failed=("final_status", lambda s: int((s == "failed").sum())),
        )
        skipped_datasets = [
            f"{idx} ({row['skip']}/{row['tasks']} tasks skipped)"
            for idx, row in by_dataset.iterrows()
            if row["done"] == 0 and row["skip"] > 0
        ]

    below_target_counts = (
        below_target.groupby("dataset").agg(
            below_recall90_workloads=("workload_name", "nunique"),
            max_recall_seen=("recall_at_k", "max"),
            median_best_attempt_recall=("recall_at_k", "median"),
        )
        .reset_index()
        .sort_values(["below_recall90_workloads", "dataset"], ascending=[False, True])
        if not below_target.empty
        else pd.DataFrame()
    )

    total_workloads = int(best_all[keys].drop_duplicates().shape[0])
    total_datasets = int(best_all["dataset"].nunique())
    max_row = best_all.sort_values("best_qps", ascending=False).head(1)
    max_text = ""
    if not max_row.empty:
        r = max_row.iloc[0]
        max_text = (
            f"{r['dataset']}/{r['workload_name']} reached {format_qps(r['best_qps'])} QPS "
            f"at recall {format_float(r['recall_at_k'], 4)}"
        )

    phase_counts = best_all["phase"].value_counts().rename_axis("phase").reset_index(name="wins")
    phase_counts.to_csv(args.output_dir / "best_phase_counts.csv", index=False)

    report_lines: list[str] = []
    report_lines.append("# Throughput90 Full Suite Analysis")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(
        f"- The completed suite produced {len(sweep):,} search-config rows; "
        f"{len(valid):,} rows satisfy Recall@10 >= {RECALL_TARGET:.2f}, zero filter violations, and positive QPS."
    )
    report_lines.append(
        f"- The best-valid selection covers {total_workloads} workloads across {total_datasets} evaluated datasets. "
        f"Datasets fully skipped by the memory guard: {', '.join(skipped_datasets) if skipped_datasets else 'none'}."
    )
    if not below_target_counts.empty:
        parts = [
            f"{row['dataset']} ({int(row['below_recall90_workloads'])} workloads below target, max recall {format_float(row['max_recall_seen'], 3)})"
            for _, row in below_target_counts.iterrows()
        ]
        report_lines.append(
            "- Workloads that ran but did not reach Recall@10 >= 0.90 are concentrated in "
            + "; ".join(parts)
            + "."
        )
    if max_text:
        report_lines.append(f"- Highest observed selected throughput: {max_text}.")
    report_lines.append(
        "- Schedule results are treated as internal algorithm selection, not a paper ablation; "
        "the paper-facing ablations should focus on leaf size, layer-adaptive degree, and layer-adaptive search budget."
    )
    report_lines.append(
        "- The old `throughput90_full_20260607_leaf_fast_s0` run is intentionally excluded because `audio/leaf=8` hung; "
        "GPU0 leaf analysis uses `throughput90_full_20260607_leaf_fast_s0_no_leaf8`."
    )

    report_lines.append("")
    report_lines.append("## Dataset-Level Best Valid Results")
    report_lines.append("")
    display_dataset = dataset_summary.copy()
    for col in ["median_qps", "mean_qps", "p10_qps", "max_qps"]:
        display_dataset[col] = display_dataset[col].map(format_qps)
    for col in ["median_recall", "min_recall", "median_exact_share", "median_graph_share", "median_merge_share"]:
        display_dataset[col] = display_dataset[col].map(lambda x: format_float(x, 3))
    display_dataset["median_leaf"] = display_dataset["median_leaf"].map(lambda x: format_float(x, 0))
    report_lines.extend(
        md_table(
            display_dataset,
            [
                ("dataset", "dataset"),
                ("workloads", "workloads"),
                ("median_qps", "median QPS"),
                ("p10_qps", "p10 QPS"),
                ("max_qps", "max QPS"),
                ("median_recall", "median recall"),
                ("min_recall", "min recall"),
                ("mode_leaf", "mode leaf"),
                ("mode_schedule", "mode schedule"),
                ("mode_policy", "mode policy"),
            ],
            max_rows=20,
        )
    )

    report_lines.append("")
    report_lines.append("## Recall-Coverage Diagnostics")
    report_lines.append("")
    coverage_display = coverage_summary.copy()
    coverage_display["coverage_rate"] = coverage_display["coverage_rate"].map(lambda x: format_float(x, 3))
    report_lines.extend(
        md_table(
            coverage_display,
            [
                ("dataset", "dataset"),
                ("attempted_workloads", "attempted"),
                ("valid_workloads", "valid >=0.90"),
                ("below_recall90_workloads", "below target"),
                ("coverage_rate", "coverage"),
            ],
            max_rows=20,
        )
    )
    report_lines.append(
        "Interpretation: `glove-100` is a quality failure under this fast-search grid, not a memory skip. "
        "`gist` only meets the target on the narrowest workloads. These datasets need either stronger search parameters, "
        "a separate high-quality configuration, or exclusion from the throughput90 claim."
    )

    report_lines.append("")
    report_lines.append("## Leaf-Size Sensitivity")
    report_lines.append("")
    if not leaf_winner_counts.empty:
        leaf_display = leaf_winner_counts.copy()
        leaf_display["median_qps"] = leaf_display["median_qps"].map(format_qps)
        leaf_display["mean_qps"] = leaf_display["mean_qps"].map(format_qps)
        leaf_display["median_recall"] = leaf_display["median_recall"].map(lambda x: format_float(x, 3))
        leaf_display["min_recall"] = leaf_display["min_recall"].map(lambda x: format_float(x, 3))
        report_lines.append(
            "Winner counts below select the fastest valid leaf size per dataset/workload from the leaf sweep."
        )
        report_lines.extend(
            md_table(
                leaf_display,
                [
                    ("leaf_size", "leaf"),
                    ("wins", "wins"),
                    ("median_qps", "median QPS"),
                    ("median_recall", "median recall"),
                    ("min_recall", "min recall"),
                ],
                max_rows=20,
            )
        )
    report_lines.append(
        "Interpretation: the useful region is small-to-moderate leaves, but `leaf=8` is not stable enough for the paper result. "
        "The selected leaf should be justified as the point that maximizes QPS under Recall@10 >= 0.90 while avoiding pathological tiny-leaf behavior."
    )

    report_lines.append("")
    report_lines.append("## Layer-Adaptive Degree Ablation")
    report_lines.append("")
    if not degree_winner_counts.empty:
        degree_display = degree_winner_counts.copy()
        degree_display["median_qps"] = degree_display["median_qps"].map(format_qps)
        degree_display["mean_qps"] = degree_display["mean_qps"].map(format_qps)
        degree_display["median_recall"] = degree_display["median_recall"].map(lambda x: format_float(x, 3))
        degree_display["min_recall"] = degree_display["min_recall"].map(lambda x: format_float(x, 3))
        report_lines.append("Winner counts select the fastest valid degree configuration per dataset/workload.")
        report_lines.extend(
            md_table(
                degree_display,
                [
                    ("config_label", "degree config"),
                    ("wins", "wins"),
                    ("median_qps", "median QPS"),
                    ("median_recall", "median recall"),
                    ("min_recall", "min recall"),
                ],
                max_rows=12,
            )
        )
    if (args.output_dir / "degree_adaptive_vs_uniform.csv").exists():
        cmp_df = pd.read_csv(args.output_dir / "degree_adaptive_vs_uniform.csv")
        speed = cmp_df["adaptive_vs_uniform_speedup"].replace([np.inf, -np.inf], np.nan).dropna()
        if not speed.empty:
            report_lines.append(
                f"Across common workload cells, the best adaptive-degree option has median speedup "
                f"{format_float(speed.median(), 2)}x over the uniform d32/i96 configuration."
            )

    report_lines.append("")
    report_lines.append("## Search-Budget Policy Ablation")
    report_lines.append("")
    if not policy_winner_counts.empty:
        policy_display = policy_winner_counts.copy()
        policy_display["median_qps"] = policy_display["median_qps"].map(format_qps)
        policy_display["mean_qps"] = policy_display["mean_qps"].map(format_qps)
        policy_display["median_recall"] = policy_display["median_recall"].map(lambda x: format_float(x, 3))
        policy_display["min_recall"] = policy_display["min_recall"].map(lambda x: format_float(x, 3))
        report_lines.append("Winner counts select the fastest valid search-budget policy per dataset/workload.")
        report_lines.extend(
            md_table(
                policy_display,
                [
                    ("search_iteration_policy", "policy"),
                    ("wins", "wins"),
                    ("median_qps", "median QPS"),
                    ("median_recall", "median recall"),
                    ("min_recall", "min recall"),
                ],
                max_rows=12,
            )
        )
    if (args.output_dir / "policy_layer_adaptive_vs_uniform.csv").exists():
        cmp_df = pd.read_csv(args.output_dir / "policy_layer_adaptive_vs_uniform.csv")
        speed = cmp_df["layer_adaptive_vs_uniform_speedup"].replace([np.inf, -np.inf], np.nan).dropna()
        if not speed.empty:
            report_lines.append(
                f"Layer-adaptive iteration policy has median speedup {format_float(speed.median(), 2)}x "
                "over uniform iteration on common valid workload cells. In this run it is not a throughput win; "
                "uniform/upper-layer policies should be considered stronger paper choices unless a later targeted run reverses this."
            )

    report_lines.append("")
    report_lines.append("## Internal Schedule Selection")
    report_lines.append("")
    if not schedule_winner_counts.empty:
        schedule_display = schedule_winner_counts.copy()
        schedule_display["median_qps"] = schedule_display["median_qps"].map(format_qps)
        schedule_display["mean_qps"] = schedule_display["mean_qps"].map(format_qps)
        schedule_display["median_recall"] = schedule_display["median_recall"].map(lambda x: format_float(x, 3))
        schedule_display["min_recall"] = schedule_display["min_recall"].map(lambda x: format_float(x, 3))
        report_lines.append(
            "This table is for internal implementation selection only; do not present it as a paper ablation."
        )
        report_lines.extend(
            md_table(
                schedule_display,
                [
                    ("search_schedule", "schedule"),
                    ("wins", "wins"),
                    ("median_qps", "median QPS"),
                    ("median_recall", "median recall"),
                    ("min_recall", "min recall"),
                ],
                max_rows=12,
            )
        )
    if (args.output_dir / "schedule_internal_vs_exact_then_graph.csv").exists():
        cmp_df = pd.read_csv(args.output_dir / "schedule_internal_vs_exact_then_graph.csv")
        speed = cmp_df["best_vs_exact_then_graph_speedup"].replace([np.inf, -np.inf], np.nan).dropna()
        if not speed.empty:
            report_lines.append(
                f"Best schedule vs exact_then_graph median speedup: {format_float(speed.median(), 2)}x."
            )

    report_lines.append("")
    report_lines.append("## Caveats And Next Analysis")
    report_lines.append("")
    report_lines.append(
        "- `wit` and `text2image` are mostly or fully excluded by the 9.90 GiB planning guard on this 2080 Ti setup; "
        "do not claim full-dataset coverage for those unless rerun on larger memory or lower-memory configs."
    )
    report_lines.append(
        "- `glove-100` produced results but no tested fast configuration reached Recall@10 >= 0.90; "
        "`gist` reached the target only on 4/33 workloads."
    )
    report_lines.append(
        "- `leaf=8` produced a live-lock/pathological kernel case on audio `pos_w15`; keep it as an engineering finding, not as a paper result."
    )
    report_lines.append(
        "- The suite does not exhaustively cross product best leaf, degree, policy, and schedule; the reported best is the best among tested configurations. "
        "A final camera-ready run can freeze the selected settings and rerun only those configurations."
    )
    report_lines.append(
        "- The paper should frame the objective as maximizing throughput under Recall@10 >= 0.90, not as high-recall frontier optimization."
    )

    report_lines.append("")
    report_lines.append("## Generated Files")
    report_lines.append("")
    for path in sorted(args.output_dir.glob("*.csv")):
        report_lines.append(f"- `{path.name}`")

    report_path = args.output_dir / "paper_experiment_analysis.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
