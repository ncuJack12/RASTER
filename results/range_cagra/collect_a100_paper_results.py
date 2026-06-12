#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(row, key, default=math.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def enrich(prefix, row):
    out = dict(prefix)
    out.update(row)
    return out


def collect(suite_dir):
    plan = read_csv(suite_dir / "suite_plan.csv")
    summary_rows = []
    sweep_rows = []
    phase_rows = []
    missing = []
    for item in plan:
        run_root = ROOT / item["expected_run_root"]
        prefix = {
            "suite_run_id": suite_dir.name,
            "suite_phase": item["phase"],
            "suite_workload": item["workload"],
            "suite_primary_workload": item.get("primary_workload", ""),
            "suite_workload_sweep": item.get("workload_sweep", ""),
            "suite_assigned_gpu": item.get("assigned_gpu", ""),
            "suite_child_run_id": item["run_id"],
            "suite_expected_run_root": item["expected_run_root"],
        }
        summary = read_csv(run_root / "aggregate_summary.csv")
        sweep = read_csv(run_root / "aggregate_sweep.csv")
        phase = read_csv(run_root / "aggregate_phase_gpu.csv")
        if not summary and not sweep:
            missing.append(item)
        summary_rows.extend(enrich(prefix, row) for row in summary)
        sweep_rows.extend(enrich(prefix, row) for row in sweep)
        phase_rows.extend(enrich(prefix, row) for row in phase)
    return plan, summary_rows, sweep_rows, phase_rows, missing


def best_by_threshold(sweep_rows, thresholds):
    out = []
    for threshold in thresholds:
        grouped = {}
        for row in sweep_rows:
            if int(float(row.get("filter_violations", 1) or 1)) != 0:
                continue
            if as_float(row, "recall_at_k", -1.0) < threshold:
                continue
            key = (
                row.get("suite_phase", ""),
                row.get("dataset", ""),
                row.get("workload_name", row.get("suite_workload", "")),
            )
            current = grouped.get(key)
            if current is None or as_float(row, "best_qps", -1.0) > as_float(current, "best_qps", -1.0):
                grouped[key] = row
        for key, row in sorted(grouped.items()):
            out.append(
                {
                    "recall_threshold": threshold,
                    "suite_phase": key[0],
                    "dataset": key[1],
                    "workload_name": key[2],
                    "best_qps": row.get("best_qps", ""),
                    "recall_at_k": row.get("recall_at_k", ""),
                    "build_seconds": row.get("build_seconds", ""),
                    "leaf_size": row.get("leaf_size", ""),
                    "config_label": row.get("config_label", ""),
                    "build_algo": row.get("build_algo", ""),
                    "graph_degree": row.get("graph_degree", ""),
                    "intermediate_graph_degree": row.get("intermediate_graph_degree", ""),
                    "nn_descent_iters": row.get("nn_descent_iters", ""),
                    "search_config_id": row.get("search_config_id", ""),
                    "search_iteration_policy": row.get("search_iteration_policy", ""),
                    "ef": row.get("ef", ""),
                    "graph_iterations": row.get("graph_iterations", ""),
                    "search_width": row.get("search_width", row.get("graph_search_concurrency", "")),
                    "entry_count": row.get("entry_count", ""),
                    "exact_seconds": row.get("exact_seconds", ""),
                    "graph_seconds": row.get("graph_seconds", ""),
                    "merge_seconds": row.get("merge_seconds", ""),
                    "suite_expected_run_root": row.get("suite_expected_run_root", ""),
                }
            )
    return out


def write_analysis(path, plan, summary_rows, sweep_rows, phase_rows, best_rows, missing):
    lines = [
        "# A100 Range-CAGRA Paper Suite Analysis",
        "",
        f"- planned child runs: {len(plan)}",
        f"- completed summary rows: {len(summary_rows)}",
        f"- completed search rows: {len(sweep_rows)}",
        f"- completed GPU phase rows: {len(phase_rows)}",
        f"- missing child runs: {len(missing)}",
        "",
    ]
    if missing:
        lines.append("## Missing")
        lines.append("")
        for row in missing[:50]:
            lines.append(f"- `{row['run_id']}` expected at `{row['expected_run_root']}`")
        if len(missing) > 50:
            lines.append(f"- ... {len(missing) - 50} more")
        lines.append("")

    lines.append("## Best QPS By Recall Threshold")
    lines.append("")
    if best_rows:
        lines.append("| threshold | phase | dataset | workload | qps | recall | config | search |")
        lines.append("|---:|---|---|---|---:|---:|---|---|")
        for row in best_rows[:80]:
            config = f"{row.get('config_label','')} leaf={row.get('leaf_size','')}"
            search = (
                f"ef={row.get('ef','')} it={row.get('graph_iterations','')} "
                f"w={row.get('search_width','')} policy={row.get('search_iteration_policy','')}"
            )
            lines.append(
                f"| {row['recall_threshold']} | {row['suite_phase']} | {row['dataset']} | "
                f"{row['workload_name']} | {row['best_qps']} | {row['recall_at_k']} | "
                f"{config} | {search} |"
            )
    else:
        lines.append("No valid frontier rows yet.")
    path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect A100 paper suite child outputs.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--thresholds", default="0.95 0.98 0.99 0.995")
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = pathlib.Path(args.suite_dir)
    if not suite_dir.is_absolute():
        suite_dir = ROOT / suite_dir
    if not (suite_dir / "suite_plan.csv").exists():
        raise SystemExit(f"missing suite_plan.csv under {suite_dir}")
    thresholds = [float(item) for item in args.thresholds.replace(",", " ").split()]
    plan, summary_rows, sweep_rows, phase_rows, missing = collect(suite_dir)
    best_rows = best_by_threshold(sweep_rows, thresholds)
    out_dir = suite_dir / "combined"
    write_csv(out_dir / "combined_summary.csv", summary_rows)
    write_csv(out_dir / "combined_sweep.csv", sweep_rows)
    write_csv(out_dir / "combined_phase_gpu.csv", phase_rows)
    write_csv(out_dir / "best_by_recall_threshold.csv", best_rows)
    write_analysis(out_dir / "analysis.md", plan, summary_rows, sweep_rows, phase_rows, best_rows, missing)
    print((out_dir / "combined_summary.csv").relative_to(ROOT))
    print((out_dir / "combined_sweep.csv").relative_to(ROOT))
    print((out_dir / "best_by_recall_threshold.csv").relative_to(ROOT))
    print((out_dir / "analysis.md").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
