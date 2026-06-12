#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib
import statistics
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def median(values):
    values = [v for v in values if math.isfinite(v)]
    return statistics.median(values) if values else float("nan")


def geomean(values):
    values = [v for v in values if v > 0 and math.isfinite(v)]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fmt(value, digits=3):
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def policy_label(row):
    label = row.get("search_iteration_policy_label", "")
    if label:
        return label
    policy = row.get("search_iteration_policy", "")
    if policy == "layer_adaptive":
        mn = row.get("search_iteration_min_graph_iterations", "")
        mx = row.get("search_iteration_max_graph_iterations", "")
        return f"layer_adaptive_min{mn}_max{mx}"
    return policy


def workload_key(row):
    return (row.get("dataset", ""), row.get("workload_name", ""))


def collect_rows(suite_dir):
    plan = read_csv(suite_dir / "suite_plan.csv")
    rows = []
    status_rows = []
    for plan_row in plan:
        child_root = ROOT / plan_row["expected_run_root"]
        child_status = read_csv(child_root / "status.csv")
        for status in child_status:
            status_rows.append({"child_run_id": plan_row["child_run_id"], **status})
        for row in read_csv(child_root / "aggregate_sweep.csv"):
            rows.append(
                {
                    "child_run_id": plan_row["child_run_id"],
                    "suite_phase": plan_row.get("phase", ""),
                    **row,
                    "search_iteration_policy_label": policy_label(row),
                }
            )
    return plan, status_rows, rows


def best_frontier(rows, target_recall):
    valid = [
        row
        for row in rows
        if as_float(row, "recall_at_k") >= target_recall
        and as_int(row, "filter_violations") == 0
        and as_float(row, "best_qps") > 0
    ]
    best = {}
    for row in valid:
        key = (*workload_key(row), row["search_iteration_policy_label"])
        if key not in best or as_float(row, "best_qps") > as_float(best[key], "best_qps"):
            best[key] = row
    return valid, best


def build_frontier_rows(best):
    out = []
    for (dataset, workload, label), row in sorted(best.items()):
        out.append(
            {
                "dataset": dataset,
                "workload_name": workload,
                "search_iteration_policy_label": label,
                "search_iteration_policy": row.get("search_iteration_policy", ""),
                "search_config_id": row.get("search_config_id", ""),
                "ef": row.get("ef", ""),
                "graph_iterations": row.get("graph_iterations", ""),
                "best_qps": row.get("best_qps", ""),
                "recall_at_k": row.get("recall_at_k", ""),
                "search_iteration_min_graph_iterations": row.get(
                    "search_iteration_min_graph_iterations", ""
                ),
                "search_iteration_max_graph_iterations": row.get(
                    "search_iteration_max_graph_iterations", ""
                ),
                "search_iteration_avg_graph_iterations": row.get(
                    "search_iteration_avg_graph_iterations", ""
                ),
                "child_run_id": row.get("child_run_id", ""),
            }
        )
    return out


def build_same_config_pairs(rows, target_recall):
    by = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("workload_name", ""),
            row.get("search_schedule", ""),
            row.get("search_config_id", ""),
            row["search_iteration_policy_label"],
        )
        by[key] = row

    pairs = []
    labels = sorted({row["search_iteration_policy_label"] for row in rows if row["search_iteration_policy_label"] != "uniform"})
    base_keys = {
        (row.get("dataset", ""), row.get("workload_name", ""), row.get("search_schedule", ""), row.get("search_config_id", ""))
        for row in rows
    }
    for dataset, workload, schedule, cfg in sorted(base_keys):
        uniform = by.get((dataset, workload, schedule, cfg, "uniform"))
        if not uniform:
            continue
        uniform_qps = as_float(uniform, "best_qps")
        uniform_recall = as_float(uniform, "recall_at_k")
        if uniform_qps <= 0:
            continue
        for label in labels:
            row = by.get((dataset, workload, schedule, cfg, label))
            if not row:
                continue
            adaptive_qps = as_float(row, "best_qps")
            adaptive_recall = as_float(row, "recall_at_k")
            pairs.append(
                {
                    "dataset": dataset,
                    "workload_name": workload,
                    "search_schedule": schedule,
                    "search_config_id": cfg,
                    "search_iteration_policy_label": label,
                    "uniform_qps": f"{uniform_qps:.3f}",
                    "adaptive_qps": f"{adaptive_qps:.3f}",
                    "qps_speedup_vs_uniform": f"{adaptive_qps / uniform_qps:.6f}",
                    "uniform_recall": f"{uniform_recall:.8f}",
                    "adaptive_recall": f"{adaptive_recall:.8f}",
                    "recall_delta": f"{adaptive_recall - uniform_recall:.8f}",
                    "adaptive_recall_ge_target": int(adaptive_recall >= target_recall),
                    "search_iteration_base_graph_iterations": row.get(
                        "search_iteration_base_graph_iterations", ""
                    ),
                    "search_iteration_min_graph_iterations": row.get(
                        "search_iteration_min_graph_iterations", ""
                    ),
                    "search_iteration_max_graph_iterations": row.get(
                        "search_iteration_max_graph_iterations", ""
                    ),
                    "search_iteration_avg_graph_iterations": row.get(
                        "search_iteration_avg_graph_iterations", ""
                    ),
                }
            )
    return pairs


def summarize(best, same_config_pairs, all_rows, target_recall):
    labels = sorted({label for (_, _, label) in best if label != "uniform"})
    baseline = {
        (dataset, workload): row
        for (dataset, workload, label), row in best.items()
        if label == "uniform"
    }
    summary = []
    for label in labels:
        frontier_rows = {
            (dataset, workload): row
            for (dataset, workload, row_label), row in best.items()
            if row_label == label
        }
        paired = []
        for key, row in frontier_rows.items():
            base = baseline.get(key)
            if not base:
                continue
            base_qps = as_float(base, "best_qps")
            qps = as_float(row, "best_qps")
            if base_qps <= 0 or qps <= 0:
                continue
            paired.append((row, base, qps / base_qps, as_float(row, "recall_at_k") - as_float(base, "recall_at_k")))

        same_pairs = [p for p in same_config_pairs if p["search_iteration_policy_label"] == label]
        same_valid = [p for p in same_pairs if int(p["adaptive_recall_ge_target"])]
        speedups = [item[2] for item in paired]
        recall_deltas = [item[3] for item in paired]
        same_speedups = [as_float(p, "qps_speedup_vs_uniform") for p in same_pairs]
        same_valid_speedups = [as_float(p, "qps_speedup_vs_uniform") for p in same_valid]
        same_recall_deltas = [as_float(p, "recall_delta") for p in same_pairs]
        boost_violations = [
            row
            for row in all_rows
            if row["search_iteration_policy_label"] == label
            and as_float(row, "search_iteration_max_graph_iterations")
            > as_float(row, "search_iteration_base_graph_iterations")
        ]
        summary.append(
            {
                "search_iteration_policy_label": label,
                "frontier_coverage": len(frontier_rows),
                "uniform_frontier_coverage": len(baseline),
                "paired_frontier_workloads": len(paired),
                "frontier_wins": sum(1 for _, _, speedup, _ in paired if speedup > 1.000001),
                "frontier_losses": sum(1 for _, _, speedup, _ in paired if speedup < 0.999999),
                "frontier_median_speedup_vs_uniform": fmt(median(speedups), 6),
                "frontier_geomean_speedup_vs_uniform": fmt(geomean(speedups), 6),
                "frontier_median_recall_delta_vs_uniform": fmt(median(recall_deltas), 8),
                "same_config_pairs": len(same_pairs),
                "same_config_adaptive_recall_ge_target": len(same_valid),
                "same_config_median_speedup": fmt(median(same_speedups), 6),
                "same_config_valid_median_speedup": fmt(median(same_valid_speedups), 6),
                "same_config_median_recall_delta": fmt(median(same_recall_deltas), 8),
                "min_frontier_recall": fmt(
                    min((as_float(row, "recall_at_k") for row in frontier_rows.values()), default=float("nan")),
                    8,
                ),
                "max_iters_gt_base_rows": len(boost_violations),
            }
        )
    summary.sort(
        key=lambda row: (
            -int(row["paired_frontier_workloads"]),
            -float(row["frontier_geomean_speedup_vs_uniform"] or "0"),
            row["search_iteration_policy_label"],
        )
    )
    return summary


def write_analysis(path, suite_dir, args, rows, valid, frontier_rows, summary):
    lines = [
        "# Search-Adaptive Ratio Sweep Analysis",
        "",
        f"suite_dir: `{suite_dir.relative_to(ROOT)}`",
        f"target_recall: `{args.target_recall}`",
        "",
        "## Data",
        "",
        f"- merged rows: {len(rows)}",
        f"- valid rows: {len(valid)}",
        f"- frontier rows: {len(frontier_rows)}",
        "",
        "## Ratio Summary",
        "",
        "| label | coverage | paired | wins/losses | frontier geomean speedup | frontier median speedup | same-config median speedup | same-config recall delta | max-iters>base rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {coverage} | {paired} | {wins}/{losses} | {geo} | {med} | {same} | {delta} | {boost} |".format(
                label=row["search_iteration_policy_label"],
                coverage=row["frontier_coverage"],
                paired=row["paired_frontier_workloads"],
                wins=row["frontier_wins"],
                losses=row["frontier_losses"],
                geo=row["frontier_geomean_speedup_vs_uniform"],
                med=row["frontier_median_speedup_vs_uniform"],
                same=row["same_config_median_speedup"],
                delta=row["same_config_median_recall_delta"],
                boost=row["max_iters_gt_base_rows"],
            )
        )
    lines.extend(
        [
            "",
            "## Selection Rule",
            "",
            "Prefer the label with full or near-full paired frontier coverage and the highest geomean QPS speedup. If two labels are close, choose the one with smaller recall loss and zero `max_iters>base` rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Range-CAGRA search-adaptive ratio sweeps.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = pathlib.Path(args.suite_dir)
    if not suite_dir.is_absolute():
        suite_dir = ROOT / suite_dir
    if not suite_dir.exists():
        raise SystemExit(f"missing suite dir: {suite_dir}")
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else suite_dir / "analysis_search_ratio"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plan, status_rows, rows = collect_rows(suite_dir)
    if not plan:
        raise SystemExit(f"missing or empty suite_plan.csv under {suite_dir}")
    if not rows:
        write_csv(out_dir / "merged_status.csv", status_rows)
        raise SystemExit("no aggregate_sweep.csv rows found; run the suite first")

    valid, best = best_frontier(rows, args.target_recall)
    frontier_rows = build_frontier_rows(best)
    same_pairs = build_same_config_pairs(rows, args.target_recall)
    summary = summarize(best, same_pairs, rows, args.target_recall)

    write_csv(out_dir / "merged_status.csv", status_rows)
    write_csv(out_dir / "merged_sweep.csv", rows)
    write_csv(out_dir / "valid_rows.csv", valid)
    write_csv(out_dir / "frontier_by_workload.csv", frontier_rows)
    write_csv(out_dir / "same_config_pairs.csv", same_pairs)
    write_csv(out_dir / "ratio_summary.csv", summary)
    write_analysis(out_dir / "analysis.md", suite_dir, args, rows, valid, frontier_rows, summary)

    print(f"output_dir={out_dir.relative_to(ROOT)}")
    print((out_dir / "analysis.md").relative_to(ROOT))
    print((out_dir / "ratio_summary.csv").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
