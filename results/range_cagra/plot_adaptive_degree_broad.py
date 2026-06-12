#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#9C755F"]
CONFIG_ORDER = [
    "uniform_d32_i96",
    "adaptive_d32_i96_min16_i48",
    "adaptive_d32_i96_min8_i24",
    "adaptive_d32_i96_min4_i12",
]
DISPLAY = {
    "uniform_d32_i96": "uniform 32/96",
    "adaptive_d32_i96_min16_i48": "adaptive min16/48",
    "adaptive_d32_i96_min8_i24": "adaptive min8/24",
    "adaptive_d32_i96_min4_i12": "adaptive min4/12",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze adaptive range-CAGRA degree sweeps.")
    parser.add_argument("--run-roots", nargs="+", required=True, help="Sweep run directories.")
    parser.add_argument(
        "--output-dir",
        default="results/range_cagra/segment_tree_param_sweep/adaptive_degree_rfann_broad_combined_20260606",
    )
    parser.add_argument("--high-search-config-id", type=int, default=1)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def read_required_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    df["source_run"] = root.name
    return df


def load_inputs(roots):
    frames = {}
    for name in ["aggregate_summary.csv", "aggregate_sweep.csv", "aggregate_phase_gpu.csv", "status.csv"]:
        frames[name] = pd.concat([read_required_csv(root, name) for root in roots], ignore_index=True)
    return frames


def validate(summary: pd.DataFrame, sweep: pd.DataFrame, status: pd.DataFrame) -> list[str]:
    notes = []
    required_summary = {"dataset", "config_label", "edge_gib", "build_seconds"}
    required_sweep = {
        "dataset",
        "workload_name",
        "config_label",
        "search_config_id",
        "recall_at_k",
        "best_qps",
        "filter_violations",
    }
    missing_summary = required_summary - set(summary.columns)
    missing_sweep = required_sweep - set(sweep.columns)
    if missing_summary:
        raise ValueError(f"summary missing columns: {sorted(missing_summary)}")
    if missing_sweep:
        raise ValueError(f"sweep missing columns: {sorted(missing_sweep)}")
    if not sweep["recall_at_k"].between(0, 1).all():
        raise ValueError("recall_at_k must be in [0, 1]")
    if (sweep["best_qps"] <= 0).any():
        raise ValueError("best_qps must be positive")
    violation_count = int(sweep["filter_violations"].sum())
    if violation_count:
        notes.append(f"WARNING: filter_violations sum is {violation_count}.")
    failed = status[status["final_status"] != "done"] if "final_status" in status else pd.DataFrame()
    if not failed.empty:
        notes.append(f"WARNING: {len(failed)} task(s) did not finish with final_status=done.")
    dup = sweep.duplicated(["dataset", "workload_name", "config_label", "search_config_id"]).sum()
    if dup:
        notes.append(f"WARNING: {dup} duplicate sweep key rows.")
    return notes


def build_relative(summary: pd.DataFrame, phase: pd.DataFrame) -> pd.DataFrame:
    build_peak = (
        phase[phase["phase"] == "build"]
        .groupby(["dataset", "config_label"], as_index=False)
        .agg(peak_build_memory_mb=("peak_memory_used_mb", "max"))
    )
    cols = ["dataset", "config_label", "edge_gib", "build_seconds"]
    out = summary[cols].drop_duplicates(["dataset", "config_label"]).merge(
        build_peak, on=["dataset", "config_label"], how="left"
    )
    base = out[out["config_label"] == "uniform_d32_i96"][
        ["dataset", "edge_gib", "build_seconds", "peak_build_memory_mb"]
    ].rename(
        columns={
            "edge_gib": "uniform_edge_gib",
            "build_seconds": "uniform_build_seconds",
            "peak_build_memory_mb": "uniform_peak_build_memory_mb",
        }
    )
    out = out.merge(base, on="dataset", how="left")
    out["edge_reduction_pct"] = (1.0 - out["edge_gib"] / out["uniform_edge_gib"]) * 100.0
    out["build_reduction_pct"] = (
        1.0 - out["build_seconds"] / out["uniform_build_seconds"]
    ) * 100.0
    out["peak_memory_reduction_pct"] = (
        1.0 - out["peak_build_memory_mb"] / out["uniform_peak_build_memory_mb"]
    ) * 100.0
    return out


def search_relative(sweep: pd.DataFrame, search_config_id: int) -> pd.DataFrame:
    current = sweep[sweep["search_config_id"] == search_config_id].copy()
    keys = ["dataset", "workload_name", "search_config_id"]
    base = current[current["config_label"] == "uniform_d32_i96"][
        keys + ["recall_at_k", "best_qps"]
    ].rename(columns={"recall_at_k": "uniform_recall", "best_qps": "uniform_qps"})
    out = current.merge(base, on=keys, how="left")
    out["recall_delta"] = out["recall_at_k"] - out["uniform_recall"]
    out["qps_ratio"] = out["best_qps"] / out["uniform_qps"]
    return out


def save_figure(fig, output_dir: Path, stem: str, formats):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = fig_dir / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def plot_build_reduction(build_rel: pd.DataFrame, output_dir: Path, formats):
    data = (
        build_rel[build_rel["config_label"] != "uniform_d32_i96"]
        .groupby("config_label", as_index=False)
        .agg(
            edge_reduction_pct=("edge_reduction_pct", "mean"),
            build_reduction_pct=("build_reduction_pct", "mean"),
            peak_memory_reduction_pct=("peak_memory_reduction_pct", "mean"),
        )
    )
    data["config_label"] = pd.Categorical(data["config_label"], CONFIG_ORDER, ordered=True)
    data = data.sort_values("config_label")
    labels = [DISPLAY.get(v, v) for v in data["config_label"].astype(str)]
    x = np.arange(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    ax.bar(x - width, data["edge_reduction_pct"], width, label="Edge storage", color=PALETTE[0])
    ax.bar(x, data["build_reduction_pct"], width, label="Build time", color=PALETTE[1])
    ax.bar(x + width, data["peak_memory_reduction_pct"], width, label="Peak build memory", color=PALETTE[2])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Reduction vs. uniform (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    ax.set_title("Adaptive degree reduces build cost across fitted datasets")
    save_figure(fig, output_dir, "fig_adaptive_degree_build_reduction", formats)


def plot_recall_delta(search_rel: pd.DataFrame, output_dir: Path, formats):
    data = (
        search_rel[search_rel["config_label"] != "uniform_d32_i96"]
        .groupby(["dataset", "config_label"], as_index=False)
        .agg(mean_recall_delta=("recall_delta", "mean"), min_recall_delta=("recall_delta", "min"))
    )
    datasets = sorted(data["dataset"].unique())
    configs = [c for c in CONFIG_ORDER if c != "uniform_d32_i96"]
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    for i, cfg in enumerate(configs):
        vals = [
            data[(data["dataset"] == ds) & (data["config_label"] == cfg)]["mean_recall_delta"].iloc[0]
            for ds in datasets
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=DISPLAY[cfg], color=PALETTE[i])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Mean Recall@10 delta")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    ax.set_title("Recall sensitivity varies strongly by dataset")
    save_figure(fig, output_dir, "fig_adaptive_degree_recall_delta_by_dataset", formats)


def plot_qps_ratio(search_rel: pd.DataFrame, output_dir: Path, formats):
    data = (
        search_rel[search_rel["config_label"] != "uniform_d32_i96"]
        .groupby(["dataset", "config_label"], as_index=False)
        .agg(mean_qps_ratio=("qps_ratio", "mean"))
    )
    datasets = sorted(data["dataset"].unique())
    configs = [c for c in CONFIG_ORDER if c != "uniform_d32_i96"]
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    for i, cfg in enumerate(configs):
        vals = [
            data[(data["dataset"] == ds) & (data["config_label"] == cfg)]["mean_qps_ratio"].iloc[0]
            for ds in datasets
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=DISPLAY[cfg], color=PALETTE[i])
    ax.axhline(1.0, color="black", linewidth=0.7)
    ax.set_ylabel("QPS ratio vs. uniform")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    ax.set_title("Adaptive degree usually improves search throughput")
    save_figure(fig, output_dir, "fig_adaptive_degree_qps_ratio_by_dataset", formats)


def write_analysis(
    output_dir: Path,
    summary: pd.DataFrame,
    sweep: pd.DataFrame,
    status: pd.DataFrame,
    build_rel: pd.DataFrame,
    search_rel: pd.DataFrame,
    notes: list[str],
    search_config_id: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Adaptive Layer-Degree Broad Sweep",
        "",
        "## Data Quality",
        "",
        f"- Completed task rows: {(status['final_status'] == 'done').sum()} / {len(status)}",
        f"- Build summary rows: {len(summary)}",
        f"- Search rows: {len(sweep)}",
        f"- Datasets: {', '.join(sorted(sweep['dataset'].unique()))}",
        f"- Configs: {', '.join(CONFIG_ORDER)}",
        f"- Filter violations: {int(sweep['filter_violations'].sum())}",
    ]
    for note in notes:
        lines.append(f"- {note}")

    build_avg = (
        build_rel.groupby("config_label", as_index=False)
        .agg(
            edge_reduction_pct=("edge_reduction_pct", "mean"),
            build_reduction_pct=("build_reduction_pct", "mean"),
            peak_memory_reduction_pct=("peak_memory_reduction_pct", "mean"),
        )
        .sort_values("config_label")
    )
    lines.extend(["", "## Build-Cost Average vs Uniform", ""])
    lines.append("| config | edge reduction | build reduction | peak memory reduction |")
    lines.append("|---|---:|---:|---:|")
    for row in build_avg.itertuples(index=False):
        lines.append(
            f"| {DISPLAY.get(row.config_label, row.config_label)} | "
            f"{row.edge_reduction_pct:.2f}% | {row.build_reduction_pct:.2f}% | "
            f"{row.peak_memory_reduction_pct:.2f}% |"
        )

    rel_non_uniform = search_rel[search_rel["config_label"] != "uniform_d32_i96"].copy()
    search_avg = (
        rel_non_uniform.groupby("config_label", as_index=False)
        .agg(
            mean_recall_delta=("recall_delta", "mean"),
            min_recall_delta=("recall_delta", "min"),
            mean_qps_ratio=("qps_ratio", "mean"),
            min_qps_ratio=("qps_ratio", "min"),
        )
        .sort_values("config_label")
    )
    lines.extend(["", f"## Search Impact at search_config_id={search_config_id}", ""])
    lines.append("| config | mean recall delta | worst recall delta | mean QPS ratio | worst QPS ratio |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in search_avg.itertuples(index=False):
        lines.append(
            f"| {DISPLAY.get(row.config_label, row.config_label)} | "
            f"{row.mean_recall_delta:.5f} | {row.min_recall_delta:.5f} | "
            f"{row.mean_qps_ratio:.3f}x | {row.min_qps_ratio:.3f}x |"
        )

    per_dataset = (
        rel_non_uniform.groupby(["dataset", "config_label"], as_index=False)
        .agg(
            mean_recall=("recall_at_k", "mean"),
            min_recall=("recall_at_k", "min"),
            mean_recall_delta=("recall_delta", "mean"),
            min_recall_delta=("recall_delta", "min"),
            mean_qps_ratio=("qps_ratio", "mean"),
        )
        .sort_values(["dataset", "config_label"])
    )
    lines.extend(["", "## Per-Dataset Search Sensitivity", ""])
    lines.append(
        "| dataset | config | mean recall | min recall | mean recall delta | worst recall delta | mean QPS ratio |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in per_dataset.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {DISPLAY.get(row.config_label, row.config_label)} | "
            f"{row.mean_recall:.5f} | {row.min_recall:.5f} | "
            f"{row.mean_recall_delta:.5f} | {row.min_recall_delta:.5f} | "
            f"{row.mean_qps_ratio:.3f}x |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `adaptive min8/24` is the best current broad default candidate: it cuts edge storage by about 30% and build time by about 36% on average.",
            "- The same policy is low-risk on `audio`, `deep`, `msong`, `sift`, and `yt8mAudio`, but it is visibly recall-sensitive on `gist` and especially `glove-100`.",
            "- `adaptive min4/12` gives more QPS/build savings but is too aggressive for a paper default without per-dataset tuning.",
            "- `adaptive min16/48` is conservative; it preserves recall better but saves about half as much edge storage as min8/24.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    roots = [Path(p) for p in args.run_roots]
    output_dir = Path(args.output_dir)
    frames = load_inputs(roots)
    summary = frames["aggregate_summary.csv"]
    sweep = frames["aggregate_sweep.csv"]
    phase = frames["aggregate_phase_gpu.csv"]
    status = frames["status.csv"]
    notes = validate(summary, sweep, status)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "combined_summary.csv", index=False)
    sweep.to_csv(output_dir / "combined_sweep.csv", index=False)
    phase.to_csv(output_dir / "combined_phase_gpu.csv", index=False)
    status.to_csv(output_dir / "combined_status.csv", index=False)

    build_rel = build_relative(summary, phase)
    search_rel = search_relative(sweep, args.high_search_config_id)
    build_rel.to_csv(output_dir / "build_relative_to_uniform.csv", index=False)
    search_rel.to_csv(output_dir / "search_relative_to_uniform.csv", index=False)

    plot_build_reduction(build_rel, output_dir, args.formats)
    plot_recall_delta(search_rel, output_dir, args.formats)
    plot_qps_ratio(search_rel, output_dir, args.formats)
    write_analysis(
        output_dir,
        summary,
        sweep,
        status,
        build_rel,
        search_rel,
        notes,
        args.high_search_config_id,
    )
    print(f"Wrote {output_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()
