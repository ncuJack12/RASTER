#!/usr/bin/env python3
"""Create the Fast90 paper-readiness report for Range-CAGRA.

The report is deliberately file-backed: it reuses completed Range-CAGRA CSVs
and the CPU baseline CSVs under /home/wjy/rfann-range-gpu/results, while
avoiding the 18GB UNIFY all_grouped_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
from typing import Iterable

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[2]
RFANN_RESULTS = pathlib.Path("/home/wjy/rfann-range-gpu/results")
SWEEP_ROOT = ROOT / "results" / "range_cagra" / "segment_tree_param_sweep"
TARGET_RECALL = 0.90

A100_BASE_RUN = "a100_full_paper_11ds_reuse_20260606_221122_base_final_search_s0"
LOCAL_ANALYSIS = ROOT / "results/range_cagra/paper_full_suite/throughput90_full_20260607_analysis"
RATIO_ANALYSIS = (
    ROOT
    / "results/range_cagra/paper_full_suite/search_adaptive_ratio_sweep_20260607/analysis_search_ratio"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "results/range_cagra/paper_full_suite/fast90_paper_readiness_20260607",
    )
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    return parser.parse_args()


def f(value, default=float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def geomean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if f(v) > 0 and math.isfinite(f(v))]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def median(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(f(v))]
    if not vals:
        return float("nan")
    return statistics.median(vals)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(f(value)):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_qps(value: float) -> str:
    if value is None or not math.isfinite(f(value)):
        return ""
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def clean_workload(row: dict[str, str]) -> str:
    return row.get("workload_name") or row.get("workload") or row.get("range_mode") or ""


def best_frontier(
    rows: list[dict[str, object]], target_recall: float, qps_key: str = "qps"
) -> list[dict[str, object]]:
    best: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        if f(row.get("recall_at_k")) < target_recall:
            continue
        qps = f(row.get(qps_key))
        if not qps or qps <= 0:
            continue
        key = (str(row.get("algorithm", "")), str(row.get("dataset", "")), str(row.get("workload_name", "")))
        old = best.get(key)
        if old is None or qps > f(old.get(qps_key)):
            best[key] = row
    return sorted(best.values(), key=lambda r: (str(r.get("algorithm")), str(r.get("dataset")), str(r.get("workload_name"))))


def load_a100_uniform_frontier(target_recall: float) -> list[dict[str, object]]:
    path = SWEEP_ROOT / A100_BASE_RUN / "aggregate_sweep.csv"
    rows = []
    for row in read_csv(path):
        if row.get("search_iteration_policy") != "uniform":
            continue
        if row.get("search_schedule") != "exact_then_graph":
            continue
        if int(f(row.get("filter_violations"), 1)) != 0:
            continue
        rows.append(
            {
                "algorithm": "Range-CAGRA-A100-uniform-old",
                "dataset": row.get("dataset", ""),
                "workload_name": row.get("workload_name", ""),
                "recall_at_k": f(row.get("recall_at_k")),
                "qps": f(row.get("best_qps")),
                "ef": row.get("ef", ""),
                "graph_iterations": row.get("graph_iterations", ""),
                "entry_count": row.get("entry_count", ""),
                "search_policy": row.get("search_iteration_policy", ""),
                "source": str(path.relative_to(ROOT)),
            }
        )
    return best_frontier(rows, target_recall)


def add_grouped_baseline(
    rows: list[dict[str, object]],
    path: pathlib.Path,
    algorithm: str,
    workload_col: str = "range_mode",
    qps_col: str = "qps",
    group_col: str | None = "group",
    group_value: str | None = "overall",
) -> None:
    for row in read_csv(path):
        if group_col and group_value and row.get(group_col) != group_value:
            continue
        if row.get("status", "passed") not in ("", "passed", "ok", "done"):
            continue
        rows.append(
            {
                "algorithm": algorithm,
                "dataset": row.get("dataset", ""),
                "workload_name": row.get(workload_col, ""),
                "recall_at_k": f(row.get("recall_at_k", row.get("recall", ""))),
                "qps": f(row.get(qps_col)),
                "source": str(path),
                "qps_semantics": qps_col,
            }
        )


def add_unify_first_row_baseline(rows: list[dict[str, object]], per_workload_dir: pathlib.Path) -> None:
    for path in sorted(per_workload_dir.glob("*.csv")):
        with path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            try:
                row = next(reader)
            except StopIteration:
                continue
        if row.get("status", "passed") not in ("", "passed", "ok", "done"):
            continue
        rows.append(
            {
                "algorithm": "UNIFY",
                "dataset": row.get("dataset", ""),
                "workload_name": row.get("range_mode", ""),
                "recall_at_k": f(row.get("recall_at_k")),
                "qps": f(row.get("wall_qps", row.get("qps", ""))),
                "source": str(path),
                "qps_semantics": "first-row wall_qps from per_workload CSV",
            }
        )


def load_cpu_baseline_frontier(target_recall: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    add_grouped_baseline(
        rows,
        RFANN_RESULTS / "serf_order_range_threads40_single_param/all_grouped_results.csv",
        "SeRF",
        qps_col="wall_qps",
    )
    add_grouped_baseline(
        rows,
        RFANN_RESULTS / "serf_order_range_threads40_single_param_arxiv_fix_20260607_181755/all_grouped_results.csv",
        "SeRF",
        qps_col="wall_qps",
    )
    add_grouped_baseline(
        rows,
        RFANN_RESULTS
        / "wow_order_range_thread_sweep_compact/M32_efc128_bt16_threads_2_4_8_16_32_40_60_80/all_grouped_results.csv",
        "WoW",
        workload_col="workload",
        group_col=None,
        group_value=None,
    )
    add_grouped_baseline(
        rows,
        RFANN_RESULTS / "irangegraph_order_range_threads40_single_param/all_grouped_results.csv",
        "iRangeGraph",
    )
    add_unify_first_row_baseline(
        rows, RFANN_RESULTS / "unify_order_range_threads40_single_param/per_workload"
    )
    return best_frontier(rows, target_recall)


def summarize_frontier(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("algorithm", "")), []).append(row)
    out = []
    for algorithm, group in sorted(grouped.items()):
        out.append(
            {
                "algorithm": algorithm,
                "covered_workloads": len(group),
                "covered_datasets": len({row.get("dataset") for row in group}),
                "geomean_qps": geomean(row.get("qps", 0) for row in group),
                "median_qps": median(row.get("qps", 0) for row in group),
                "max_qps": max([f(row.get("qps")) for row in group] or [float("nan")]),
                "min_recall": min([f(row.get("recall_at_k")) for row in group] or [float("nan")]),
            }
        )
    return out


def compare_to_baselines(
    range_rows: list[dict[str, object]], baseline_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    range_by = {(row["dataset"], row["workload_name"]): row for row in range_rows}
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in baseline_rows:
        grouped.setdefault(str(row.get("algorithm", "")), []).append(row)
    out = []
    for algorithm, rows in sorted(grouped.items()):
        ratios = []
        base_qps = []
        range_qps = []
        wins = 0
        losses = 0
        for row in rows:
            key = (row.get("dataset"), row.get("workload_name"))
            ours = range_by.get(key)
            if not ours:
                continue
            rq = f(ours.get("qps"))
            bq = f(row.get("qps"))
            if rq <= 0 or bq <= 0:
                continue
            ratios.append(rq / bq)
            range_qps.append(rq)
            base_qps.append(bq)
            if rq > bq:
                wins += 1
            else:
                losses += 1
        out.append(
            {
                "baseline": algorithm,
                "paired_workloads": len(ratios),
                "range_cagra_geomean_qps": geomean(range_qps),
                "baseline_geomean_qps": geomean(base_qps),
                "geomean_speedup": geomean(ratios),
                "median_speedup": median(ratios),
                "wins": wins,
                "losses": losses,
            }
        )
    return out


def load_ratio_summary() -> list[dict[str, object]]:
    path = RATIO_ANALYSIS / "ratio_summary.csv"
    rows = []
    for row in read_csv(path):
        rows.append({key: f(value) if key not in ("search_iteration_policy_label",) else value for key, value in row.items()})
    return rows


def md_table(rows: list[dict[str, object]], columns: list[tuple[str, str]], max_rows: int = 20) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows[:max_rows]:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if "qps" in key:
                    cells.append(fmt_qps(value))
                elif "speedup" in key:
                    cells.append(fmt(value, 3) + "x")
                else:
                    cells.append(fmt(value, 3))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    out_path: pathlib.Path,
    target_recall: float,
    a100_summary: list[dict[str, object]],
    cpu_summary: list[dict[str, object]],
    comparisons: list[dict[str, object]],
    ratio_summary: list[dict[str, object]],
) -> None:
    ratio_best = sorted(
        ratio_summary,
        key=lambda r: f(r.get("frontier_geomean_speedup_vs_uniform")),
        reverse=True,
    )
    lines: list[str] = [
        "# Range-CAGRA Fast90 Paper Readiness",
        "",
        "Date: 2026-06-07.",
        "",
        "This file is separate from the previous high-recall analysis. The new paper target is to maximize QPS subject to `Recall@10 >= 0.90` and `filter_violations = 0`.",
        "",
        "## Conclusions",
        "",
        "- Existing results already support a useful paper direction: Range-CAGRA should be framed as a high-throughput range-correct ANN retrieval stage, not as a high-recall-only system.",
        "- The stable Fast90 default should be `leaf_size=64`, layer-adaptive degree `d24/i64 -> min4/i12`, fixed `exact_then_graph` schedule, and low-budget search sweep.",
        "- After the search-adaptive fix, the best tested stacked search-effort ratio on the adaptive-degree local sweep is `layer_adaptive:8:32:4`. Treat this as final-combination evidence, not as the primary search-effort ablation.",
        "- The primary search-effort ablation should be rerun on fixed uniform degree `uniform_d32_i96_it20`, matching the way the degree-adaptive ablation is compared against a fixed-degree baseline.",
        "- Yesterday's A100 run is useful as performance evidence for uniform-search rows, but the search-adaptive rows must be rerun because they were produced before today's semantic fix.",
        "- The A100 evidence below used the older `d32/i96 -> min8/i24` degree setting. The proposed Fast90 `d24/i64 -> min4/i12` default comes from today's local throughput experiment and must be confirmed by the A100 `layer_degree` ablation.",
        "",
        "## Existing A100 Evidence",
        "",
        "The table below uses only `uniform` search rows from the stopped A100 run, so it avoids the fixed search-adaptive bug. It is not the final paper result, but it shows the expected throughput scale.",
        "",
        "Important: this old A100 table is not the final Fast90 configuration; it is a conservative reference for scale and CPU-baseline comparison until the fresh A100 run finishes.",
        "",
        *md_table(
            a100_summary,
            [
                ("algorithm", "source"),
                ("covered_workloads", "workloads"),
                ("covered_datasets", "datasets"),
                ("geomean_qps", "geomean QPS"),
                ("median_qps", "median QPS"),
                ("max_qps", "max QPS"),
                ("min_recall", "min recall"),
            ],
        ),
        "",
        "## CPU Baseline Comparison",
        "",
        "Comparison is paired by identical dataset/workload and uses each method's fastest row satisfying Recall@10 >= 0.90. SeRF and UNIFY use `wall_qps`, matching the existing baseline plotting scripts. UNIFY is read from per-workload CSV first-row `wall_qps` because its `all_grouped_results.csv` is an 18GB row-level file.",
        "",
        *md_table(
            comparisons,
            [
                ("baseline", "baseline"),
                ("paired_workloads", "paired"),
                ("range_cagra_geomean_qps", "Range-CAGRA gQPS"),
                ("baseline_geomean_qps", "baseline gQPS"),
                ("geomean_speedup", "gmean speedup"),
                ("median_speedup", "median speedup"),
                ("wins", "wins"),
                ("losses", "losses"),
            ],
        ),
        "",
        "Baseline coverage at the same target:",
        "",
        *md_table(
            cpu_summary,
            [
                ("algorithm", "baseline"),
                ("covered_workloads", "workloads"),
                ("covered_datasets", "datasets"),
                ("geomean_qps", "geomean QPS"),
                ("median_qps", "median QPS"),
                ("min_recall", "min recall"),
            ],
        ),
        "",
        "## Search-Effort Result After Fix",
        "",
        "The fixed implementation now treats layer-adaptive search as a reduction relative to the active search budget. No row in the ratio sweep has `max_iters > base_iters`.",
        "",
        *md_table(
            ratio_best,
            [
                ("search_iteration_policy_label", "policy"),
                ("frontier_coverage", "coverage"),
                ("frontier_geomean_speedup_vs_uniform", "frontier gmean speedup"),
                ("frontier_median_speedup_vs_uniform", "frontier median speedup"),
                ("frontier_wins", "wins"),
                ("frontier_losses", "losses"),
                ("same_config_valid_median_speedup", "same-config valid median"),
                ("same_config_median_recall_delta", "median recall delta"),
                ("max_iters_gt_base_rows", "bad rows"),
            ],
        ),
        "",
        "Interpretation: for the final main experiment, use `layer_adaptive:8:32:4` and sweep search budgets. For the paper search-effort ablation, run the policy sweep on fixed uniform degree `uniform_d32_i96_it20`, including `uniform`, `layer_adaptive:12:32:4`, `10:32:2`, `8:32:4`, and `6:32:2` so the paper can show the throughput/recall trade-off rather than only one chosen point.",
        "",
        "## What Can Be Written In The Paper Now",
        "",
        "1. Ordered range ANN can be decomposed into exact boundary work and internal segment-tree graph search while keeping returned IDs range-correct (`filter_violations=0`).",
        "2. Because the paper target is Recall@10 >= 0.90, the winning search budgets are low: reducing graph search effort is the main source of extreme QPS.",
        "3. Layer-adaptive degree and layer-adaptive search follow the same principle: upper segment-tree nodes cover larger ranges and deserve more graph/search budget; lower nodes are numerous and can be cheaper.",
        "4. Leaf size controls the exact-boundary versus graph-interior trade-off. Existing sweeps support `leaf_size=64` as the stable fixed default.",
        "",
        "## A100 Rerun Plan",
        "",
        "Rerun after syncing today's code to A100 and rebuilding `NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST`.",
        "",
        "```bash",
        "python3 results/range_cagra/run_paper_full_experiments.py \\",
        "  --run-id a100_fast90_$(date +%Y%m%d_%H%M%S) \\",
        "  --phases smoke main_algo layer_degree layer_search leaf_size \\",
        "  --gpu-ids 0 \\",
        "  --dataset-shards 1 \\",
        "  --build-dir cpp/build-a100-sm80-cuda129 \\",
        "  --max-est-peak-gib 72 \\",
        "  --gpu-fraction 0.95 \\",
        "  --scratch-guard-gib 8",
        "```",
        "",
        "Then inspect the generated `suite_plan.csv` and run `commands.sh` or `commands_2gpu.sh` depending on the actual A100 GPU count.",
        "",
        "Required phases:",
        "",
        "- `main_algo`: final Fast90 config, selected adaptive degree/leaf/schedule, search-budget sweep, default search ratio `8/32/4`.",
        "- `layer_degree`: degree ablation with search/leaf fixed; include `d24/i64 min4/i12` and uniform `d32/i96`.",
        "- `layer_search`: search-effort ablation on fixed uniform degree; do not present schedule as a paper ablation.",
        "- `leaf_size`: leaf-size ablation with degree/search fixed.",
        "",
        "Things not to miss:",
        "",
        "- Re-run A100 after the code fix; old search-adaptive A100 rows are not paper-valid.",
        "- Analyze by per-workload frontier at Recall@10 >= 0.90, not dataset averages alone.",
        "- Keep CPU baseline comparison paired by common dataset/workload and state the hardware caveat.",
        "- If A100 `d24/i64 min4/i12` loses Recall@10 >= 0.90 on weak datasets, keep it as the throughput variant and use `d32/i96 min8/i24` as the robust fallback.",
        "- Treat `filter_violations != 0` as invalid.",
        "- Keep `gist/glove-100/text2image/wit` caveats explicit if any remain below recall target or are skipped by memory guard.",
        "",
        "## Generated Files",
        "",
        f"- `{out_path.relative_to(ROOT)}`",
        f"- `{(out_path.parent / 'range_cagra_a100_uniform_recall90_frontier.csv').relative_to(ROOT)}`",
        f"- `{(out_path.parent / 'cpu_baseline_frontier_recall90.csv').relative_to(ROOT)}`",
        f"- `{(out_path.parent / 'a100_vs_cpu_baseline_recall90.csv').relative_to(ROOT)}`",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    a100_frontier = load_a100_uniform_frontier(args.target_recall)
    cpu_frontier = load_cpu_baseline_frontier(args.target_recall)
    a100_summary = summarize_frontier(a100_frontier)
    cpu_summary = summarize_frontier(cpu_frontier)
    comparisons = compare_to_baselines(a100_frontier, cpu_frontier)
    ratio_summary = load_ratio_summary()

    write_csv(out_dir / "range_cagra_a100_uniform_recall90_frontier.csv", a100_frontier)
    write_csv(out_dir / "cpu_baseline_frontier_recall90.csv", cpu_frontier)
    write_csv(out_dir / "range_cagra_a100_uniform_recall90_summary.csv", a100_summary)
    write_csv(out_dir / "cpu_baseline_recall90_summary.csv", cpu_summary)
    write_csv(out_dir / "a100_vs_cpu_baseline_recall90.csv", comparisons)
    write_csv(out_dir / "search_adaptive_ratio_summary.csv", ratio_summary)
    write_report(
        out_dir / "RANGE_CAGRA_FAST90_PAPER_READINESS_20260607.md",
        args.target_recall,
        a100_summary,
        cpu_summary,
        comparisons,
        ratio_summary,
    )

    print(out_dir.relative_to(ROOT))
    print((out_dir / "RANGE_CAGRA_FAST90_PAPER_READINESS_20260607.md").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
