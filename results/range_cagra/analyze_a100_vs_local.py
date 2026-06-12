#!/usr/bin/env python3
import csv
import datetime as dt
import math
import pathlib
from collections import defaultdict


ROOT = pathlib.Path(__file__).resolve().parents[2]
SWEEP_ROOT = ROOT / "results" / "range_cagra" / "segment_tree_param_sweep"

A100_BASE = "a100_full_paper_11ds_reuse_20260606_221122_base_final_search_s0"
A100_DEGREE = "a100_full_paper_11ds_reuse_20260606_221122_layer_degree_other_s0"
LOCAL_MAIN = "paper_main_algo_1gpu_ordinary_20260606_main_algo_s0"
LOCAL_POLICY = [
    "paper_full_20260606_layer_search_s0",
    "paper_full_20260606_layer_search_s1",
]
LOCAL_DEGREE = [
    "paper_full_20260606_layer_degree_s0",
    "paper_full_20260606_layer_degree_s1",
]

THRESHOLDS = [0.95, 0.99, 0.995, 0.999]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def i(row, key, default=0):
    try:
        value = row.get(key, "")
        if value == "" or value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def gmean(values):
    values = [v for v in values if v and v > 0 and not math.isnan(v)]
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def avg(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def search_key(row):
    return (
        row.get("ef", ""),
        row.get("graph_iterations", ""),
        row.get("search_width", ""),
        row.get("entry_count", ""),
    )


def pair_key(row):
    workload = row.get("workload_name") or pathlib.Path(row.get("workload", "")).name
    return (row.get("dataset", ""), workload) + search_key(row)


def workload_key(row):
    workload = row.get("workload_name") or pathlib.Path(row.get("workload", "")).name
    return (row.get("dataset", ""), workload)


def cfg_label(row):
    return row.get("task_label") or row.get("config_label") or ""


def load_sweep(run_name):
    rows = read_csv(SWEEP_ROOT / run_name / "aggregate_sweep.csv")
    for row in rows:
        row["_run"] = run_name
    return rows


def load_phase(run_name):
    rows = read_csv(SWEEP_ROOT / run_name / "aggregate_phase_gpu.csv")
    for row in rows:
        row["_run"] = run_name
    return rows


def rows_for_policy(rows, policy):
    return [r for r in rows if r.get("search_iteration_policy") == policy]


def restrict_search_configs(rows, configs):
    return [r for r in rows if search_key(r) in configs]


def frontier(rows, threshold, group_func=workload_key):
    best = {}
    for row in rows:
        recall = f(row, "recall_at_k", float("nan"))
        if math.isnan(recall) or recall < threshold:
            continue
        qps = f(row, "avg_qps", f(row, "best_qps", float("nan")))
        if math.isnan(qps):
            continue
        key = group_func(row)
        old = best.get(key)
        if old is None or qps > f(old, "avg_qps", f(old, "best_qps", 0.0)):
            best[key] = row
    return best


def summarize_frontier(left_rows, right_rows, left_name, right_name, mode):
    out = []
    for th in THRESHOLDS:
        left = frontier(left_rows, th)
        right = frontier(right_rows, th)
        common = sorted(set(left) & set(right))
        ratios = [
            f(right[k], "avg_qps", f(right[k], "best_qps"))
            / f(left[k], "avg_qps", f(left[k], "best_qps"))
            for k in common
            if f(left[k], "avg_qps", f(left[k], "best_qps")) > 0
        ]
        wins = sum(
            1
            for k in common
            if f(right[k], "avg_qps", f(right[k], "best_qps"))
            > f(left[k], "avg_qps", f(left[k], "best_qps")) * 1.001
        )
        losses = sum(
            1
            for k in common
            if f(right[k], "avg_qps", f(right[k], "best_qps"))
            < f(left[k], "avg_qps", f(left[k], "best_qps")) / 1.001
        )
        out.append(
            {
                "mode": mode,
                "threshold": th,
                "left": left_name,
                "right": right_name,
                "left_coverage": len(left),
                "right_coverage": len(right),
                "paired_workloads": len(common),
                "right_over_left_gmean_qps": fmt(gmean(ratios), 4),
                "right_wins": wins,
                "right_losses": losses,
                "ties": len(common) - wins - losses,
            }
        )
    return out


def unique_builds(rows, policy=None):
    out = {}
    for row in rows:
        if policy and row.get("search_iteration_policy") != policy:
            continue
        key = (row.get("dataset", ""), cfg_label(row))
        if key not in out:
            out[key] = row
    return out


def phase_max_by_dataset(rows):
    out = {}
    for row in rows:
        ds = row.get("dataset", "")
        phase = row.get("phase", "")
        out.setdefault(ds, {})
        out[ds][f"{phase}_peak_memory_mb"] = max(
            out[ds].get(f"{phase}_peak_memory_mb", 0.0), f(row, "peak_memory_used_mb")
        )
        out[ds][f"{phase}_avg_gpu_util_pct"] = max(
            out[ds].get(f"{phase}_avg_gpu_util_pct", 0.0), f(row, "avg_gpu_util_pct")
        )
    return out


def matched_final_rows(local_main, a100_base):
    local = rows_for_policy(local_main, "layer_adaptive")
    a100 = rows_for_policy(a100_base, "layer_adaptive")
    local_configs = {search_key(r) for r in local}
    a100_common = restrict_search_configs(a100, local_configs)
    local_idx = {pair_key(r): r for r in local}
    a100_idx = {pair_key(r): r for r in a100_common}
    rows = []
    for key in sorted(set(local_idx) & set(a100_idx)):
        lrow, arow = local_idx[key], a100_idx[key]
        lqps = f(lrow, "avg_qps", f(lrow, "best_qps"))
        aqps = f(arow, "avg_qps", f(arow, "best_qps"))
        lrec = f(lrow, "recall_at_k")
        arec = f(arow, "recall_at_k")
        rows.append(
            {
                "dataset": key[0],
                "workload_name": key[1],
                "ef": key[2],
                "graph_iterations": key[3],
                "search_width": key[4],
                "entry_count": key[5],
                "local_avg_qps": fmt(lqps, 3),
                "a100_avg_qps": fmt(aqps, 3),
                "a100_over_local_qps": fmt(aqps / lqps if lqps > 0 else float("nan"), 4),
                "local_recall_at_k": fmt(lrec, 8),
                "a100_recall_at_k": fmt(arec, 8),
                "recall_delta_a100_minus_local": fmt(arec - lrec, 8),
                "local_exact_seconds": fmt(f(lrow, "exact_seconds"), 6),
                "a100_exact_seconds": fmt(f(arow, "exact_seconds"), 6),
                "local_graph_seconds": fmt(f(lrow, "graph_seconds"), 6),
                "a100_graph_seconds": fmt(f(arow, "graph_seconds"), 6),
                "local_merge_seconds": fmt(f(lrow, "merge_seconds"), 6),
                "a100_merge_seconds": fmt(f(arow, "merge_seconds"), 6),
            }
        )
    return rows


def summarize_matched_by_dataset(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    out = []
    for ds, rs in sorted(groups.items()):
        ratios = [float(r["a100_over_local_qps"]) for r in rs if r["a100_over_local_qps"]]
        recall_d = [float(r["recall_delta_a100_minus_local"]) for r in rs]
        out.append(
            {
                "dataset": ds,
                "matched_rows": len(rs),
                "matched_workloads": len({r["workload_name"] for r in rs}),
                "matched_search_configs": len({(r["ef"], r["graph_iterations"], r["search_width"], r["entry_count"]) for r in rs}),
                "a100_over_local_qps_gmean": fmt(gmean(ratios), 4),
                "a100_over_local_qps_avg": fmt(avg(ratios), 4),
                "recall_delta_avg": fmt(avg(recall_d), 8),
                "recall_abs_delta_avg": fmt(avg([abs(x) for x in recall_d]), 8),
                "recall_abs_delta_max": fmt(max(abs(x) for x in recall_d), 8),
                "a100_wins_rows": sum(1 for x in ratios if x > 1.001),
                "a100_losses_rows": sum(1 for x in ratios if x < 1 / 1.001),
            }
        )
    all_ratios = [float(r["a100_over_local_qps"]) for r in rows if r["a100_over_local_qps"]]
    all_d = [float(r["recall_delta_a100_minus_local"]) for r in rows]
    out.append(
        {
            "dataset": "ALL_COMMON",
            "matched_rows": len(rows),
            "matched_workloads": len({(r["dataset"], r["workload_name"]) for r in rows}),
            "matched_search_configs": len({(r["ef"], r["graph_iterations"], r["search_width"], r["entry_count"]) for r in rows}),
            "a100_over_local_qps_gmean": fmt(gmean(all_ratios), 4),
            "a100_over_local_qps_avg": fmt(avg(all_ratios), 4),
            "recall_delta_avg": fmt(avg(all_d), 8),
            "recall_abs_delta_avg": fmt(avg([abs(x) for x in all_d]), 8),
            "recall_abs_delta_max": fmt(max(abs(x) for x in all_d), 8),
            "a100_wins_rows": sum(1 for x in all_ratios if x > 1.001),
            "a100_losses_rows": sum(1 for x in all_ratios if x < 1 / 1.001),
        }
    )
    return out


def build_memory_summary(local_main, a100_base, local_phase, a100_phase):
    local_build = unique_builds(rows_for_policy(local_main, "layer_adaptive"))
    a100_build = unique_builds(rows_for_policy(a100_base, "layer_adaptive"))
    local_mem = phase_max_by_dataset(local_phase)
    a100_mem = phase_max_by_dataset(a100_phase)
    out = []
    for (ds, _cfg), lrow in sorted(local_build.items()):
        arow = a100_build.get((ds, "adaptive_d32_i96_min8_i24_it20"))
        if not arow:
            continue
        lb, ab = f(lrow, "build_seconds"), f(arow, "build_seconds")
        out.append(
            {
                "dataset": ds,
                "local_build_seconds": fmt(lb, 3),
                "a100_build_seconds": fmt(ab, 3),
                "build_speedup_a100_vs_local": fmt(lb / ab if ab > 0 else float("nan"), 4),
                "local_edge_gib": fmt(f(lrow, "edge_gib"), 4),
                "a100_edge_gib": fmt(f(arow, "edge_gib"), 4),
                "local_whole_peak_memory_mb": fmt(local_mem.get(ds, {}).get("whole_peak_memory_mb", float("nan")), 1),
                "a100_whole_peak_memory_mb": fmt(a100_mem.get(ds, {}).get("whole_peak_memory_mb", float("nan")), 1),
                "local_search_peak_memory_mb": fmt(local_mem.get(ds, {}).get("search_peak_memory_mb", float("nan")), 1),
                "a100_search_peak_memory_mb": fmt(a100_mem.get(ds, {}).get("search_peak_memory_mb", float("nan")), 1),
            }
        )
    return out


def search_policy_effect(local_policy, a100_base):
    local_configs = {search_key(r) for r in local_policy}
    datasets = {r.get("dataset") for r in local_policy}
    a100_common = [
        r for r in a100_base if r.get("dataset") in datasets and search_key(r) in local_configs
    ]
    out = []
    for hardware, rows in [("local_2080ti", local_policy), ("a100", a100_common)]:
        for cmp_policy in ["uniform", "upper_layers"]:
            out += summarize_frontier(
                rows_for_policy(rows, cmp_policy),
                rows_for_policy(rows, "layer_adaptive"),
                cmp_policy,
                "layer_adaptive",
                f"{hardware}_common_5configs_{cmp_policy}_to_layer_adaptive",
            )
    return out


def degree_summary(local_degree, a100_degree, a100_base):
    local_configs = {cfg_label(r) for r in local_degree}
    local_search = {search_key(r) for r in local_degree}
    local_datasets = {r.get("dataset") for r in local_degree}

    # Use A100 base as the final min8 build/search evidence, because the A100 degree
    # run intentionally excludes that already-completed base config.
    a100_degree_aug = list(a100_degree)
    for row in a100_base:
        if row.get("dataset") in local_datasets and row.get("search_iteration_policy") == "uniform":
            copied = dict(row)
            copied["task_label"] = "adaptive_d32_i96_min8_i24_it20"
            copied["config_label"] = "adaptive_d32_i96_min8_i24_it20"
            a100_degree_aug.append(copied)

    a100_configs = {cfg_label(r) for r in a100_degree_aug}
    common_configs = sorted(local_configs & a100_configs)

    out = []
    for cfg in common_configs:
        local_cfg_all = [
            r
            for r in local_degree
            if cfg_label(r) == cfg and r.get("search_iteration_policy") == "uniform"
        ]
        a100_cfg_all = [
            r
            for r in a100_degree_aug
            if r.get("dataset") in local_datasets
            and cfg_label(r) == cfg
            and search_key(r) in local_search
            and r.get("search_iteration_policy") == "uniform"
        ]
        common_ds = sorted({r.get("dataset") for r in local_cfg_all} & {r.get("dataset") for r in a100_cfg_all})
        local_cfg = [r for r in local_cfg_all if r.get("dataset") in common_ds]
        a100_cfg = [r for r in a100_cfg_all if r.get("dataset") in common_ds]
        denominator = len(common_ds) * 33
        for hardware, cfg_rows in [("local_2080ti", local_cfg), ("a100", a100_cfg)]:
            builds = unique_builds(cfg_rows)
            build_sum = sum(f(r, "build_seconds") for r in builds.values())
            edge_sum = sum(f(r, "edge_gib", f(r, "edge_gib_est")) for r in builds.values())
            for th in THRESHOLDS:
                best = frontier(cfg_rows, th)
                out.append(
                    {
                        "hardware": hardware,
                        "config": cfg,
                        "compared_datasets": ";".join(common_ds),
                        "threshold": th,
                        "coverage_workloads": len(best),
                        "coverage_denominator": denominator,
                        "frontier_gmean_qps": fmt(gmean([f(r, "avg_qps") for r in best.values()]), 3),
                        "unique_builds": len(builds),
                        "build_seconds_sum": fmt(build_sum, 3),
                        "build_seconds_avg": fmt(build_sum / len(builds) if builds else float("nan"), 3),
                        "edge_gib_sum": fmt(edge_sum, 4),
                    }
                )
    return out


def a100_extra_dataset_summary(a100_base, a100_phase, local_main):
    local_datasets = {r.get("dataset") for r in local_main}
    extras = sorted({r.get("dataset") for r in a100_base} - local_datasets)
    mem = phase_max_by_dataset(a100_phase)
    out = []
    for ds in extras:
        rows = [r for r in a100_base if r.get("dataset") == ds and r.get("search_iteration_policy") == "layer_adaptive"]
        builds = unique_builds(rows)
        build = next(iter(builds.values()), {})
        rec_min = min([f(r, "recall_at_k") for r in rows], default=float("nan"))
        rec_max = max([f(r, "recall_at_k") for r in rows], default=float("nan"))
        item = {
            "dataset": ds,
            "rows": len(rows),
            "workloads": len({workload_key(r) for r in rows}),
            "search_configs": len({search_key(r) for r in rows}),
            "build_seconds": fmt(f(build, "build_seconds"), 3),
            "edge_gib": fmt(f(build, "edge_gib"), 4),
            "a100_whole_peak_memory_mb": fmt(mem.get(ds, {}).get("whole_peak_memory_mb", float("nan")), 1),
            "recall_min": fmt(rec_min, 6),
            "recall_max": fmt(rec_max, 6),
        }
        for th in THRESHOLDS:
            item[f"coverage_ge_{th}"] = len(frontier(rows, th))
        out.append(item)
    return out


def markdown_table(rows, columns, limit=None):
    if limit:
        rows = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def write_report(out_dir, tables):
    dataset_rows = tables["final_dataset_summary"]
    frontier_rows = tables["final_frontier_summary"]
    build_rows = tables["final_build_memory"]
    degree_rows = tables["degree_config_summary"]
    policy_rows = tables["search_policy_effect"]
    extra_rows = tables["a100_extra_dataset_summary"]

    overall = [r for r in dataset_rows if r["dataset"] == "ALL_COMMON"][0]
    fair_frontier = [r for r in frontier_rows if r["mode"] == "common_7configs_local_vs_a100"]
    degree_995 = [r for r in degree_rows if str(r["threshold"]) == "0.995"]
    degree_995 = sorted(degree_995, key=lambda r: (r["hardware"], r["config"]))
    policy_995 = [r for r in policy_rows if str(r["threshold"]) == "0.995"]

    text = f"""# A100 vs Local Range-CAGRA Comparison

Generated: {dt.datetime.now().isoformat(timespec='seconds')}

## Scope

- A100 source: `{A100_BASE}` and completed rows from `{A100_DEGREE}` copied from `root@10.0.0.9:/wjy/cuvs`.
- Local sources: `{LOCAL_MAIN}`, `{', '.join(LOCAL_POLICY)}`, and `{', '.join(LOCAL_DEGREE)}`.
- Hardware comparison is restricted to common datasets and common search configs. A100-only datasets are reported separately.
- For frontier rows, the rule is: for each dataset/workload and recall threshold, choose the row with highest `avg_qps` among rows with `recall_at_k >= threshold`.

## Main Result

On the common final-algorithm slice, A100 is consistently faster but not perfectly recall-identical:

- Matched final rows: `{overall['matched_rows']}` rows across `{overall['matched_workloads']}` dataset/workloads and `{overall['matched_search_configs']}` shared search configs.
- Matched-row geometric mean QPS ratio, A100/local: `{overall['a100_over_local_qps_gmean']}x`.
- Average recall delta, A100 - local: `{overall['recall_delta_avg']}`; average absolute delta `{overall['recall_abs_delta_avg']}`.
- Maximum absolute recall delta on a matched row: `{overall['recall_abs_delta_max']}`.

### Final Algorithm By Dataset

{markdown_table(dataset_rows, ['dataset','matched_rows','a100_over_local_qps_gmean','recall_delta_avg','recall_abs_delta_avg','recall_abs_delta_max','a100_wins_rows','a100_losses_rows'])}

### Recall/QPS Frontier

Fair comparison uses the 7 search configs that exist on both machines. The expanded comparison lets A100 use its two extra high-budget configs and should be read as A100-capability evidence, not a hardware-only comparison.

{markdown_table(frontier_rows, ['mode','threshold','left_coverage','right_coverage','paired_workloads','right_over_left_gmean_qps','right_wins','right_losses','ties'])}

## Build And Memory

`build_seconds` comes from `aggregate_sweep.csv`; peak memory comes from `aggregate_phase_gpu.csv` samples. Memory is not normalized by card capacity.

{markdown_table(build_rows, ['dataset','local_build_seconds','a100_build_seconds','build_speedup_a100_vs_local','local_whole_peak_memory_mb','a100_whole_peak_memory_mb','local_search_peak_memory_mb','a100_search_peak_memory_mb'])}

## Search Policy Effect

This table compares `layer_adaptive` against `uniform` and `upper_layers` on the common 5-config local policy sweep. Ratios are `layer_adaptive / comparator`.

{markdown_table(policy_995, ['mode','threshold','left_coverage','right_coverage','paired_workloads','right_over_left_gmean_qps','right_wins','right_losses'])}

## Degree Ablation

For degree-ablation comparison, both machines are restricted to common datasets available for each config, common degree configs, common search configs, and `uniform` search policy. The denominator can differ by config because the A100 degree run was still partial when these artifacts were copied; use the `coverage_denominator` and `unique_builds` columns as the exact scope.

{markdown_table(degree_995, ['hardware','config','threshold','coverage_workloads','coverage_denominator','frontier_gmean_qps','unique_builds','build_seconds_sum','edge_gib_sum'])}

## A100-Only Completed Main Results

These datasets are not part of the local final-run comparison because the local run did not complete them under the same final-algorithm setup.

{markdown_table(extra_rows, ['dataset','rows','workloads','search_configs','build_seconds','edge_gib','a100_whole_peak_memory_mb','recall_min','recall_max','coverage_ge_0.99','coverage_ge_0.995','coverage_ge_0.999'])}

## Interpretation

1. A100 is useful for final paper evidence because it finishes the full 11-dataset main run and allows the high-budget 192/256 search configs. Those configs materially improve high-recall coverage on hard datasets, but they should not be mixed into a pure hardware-speed comparison.
2. The local 2080 Ti final run is still valuable as a portability check: the common 8-dataset slice shows the same qualitative behavior, but the recall deltas mean row-by-row comparisons should use frontier thresholds rather than assuming bit-identical graph/search outcomes.
3. The A100 degree data currently supports the same direction as local: smaller/adaptive degree saves build time and graph storage; larger degree improves high-recall coverage but costs build time and edge memory.
4. A100-only `arxiv-for-fanns-large`, `text2image`, and `wit` should be treated as scalability evidence, not as local-vs-A100 paired evidence.

## Output Files

- `final_matched_rows.csv`
- `final_dataset_summary.csv`
- `final_frontier_summary.csv`
- `final_build_memory.csv`
- `search_policy_effect.csv`
- `degree_config_summary.csv`
- `a100_extra_dataset_summary.csv`
"""
    (out_dir / "analysis.md").write_text(text)


def main():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "range_cagra" / "a100_vs_local_comparison" / f"a100_vs_local_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    a100_base = load_sweep(A100_BASE)
    a100_degree = load_sweep(A100_DEGREE)
    local_main = load_sweep(LOCAL_MAIN)
    local_policy = []
    for name in LOCAL_POLICY:
        local_policy.extend(load_sweep(name))
    local_degree = []
    for name in LOCAL_DEGREE:
        local_degree.extend(load_sweep(name))

    a100_phase = load_phase(A100_BASE)
    local_phase = load_phase(LOCAL_MAIN)

    matched = matched_final_rows(local_main, a100_base)
    final_dataset_summary = summarize_matched_by_dataset(matched)

    local_main_la = rows_for_policy(local_main, "layer_adaptive")
    local_configs = {search_key(r) for r in local_main_la}
    local_datasets = {r.get("dataset") for r in local_main_la}
    a100_la = rows_for_policy(a100_base, "layer_adaptive")
    a100_la_common_ds = [r for r in a100_la if r.get("dataset") in local_datasets]
    a100_la_common = restrict_search_configs(a100_la_common_ds, local_configs)
    final_frontier = []
    final_frontier += summarize_frontier(
        local_main_la,
        a100_la_common,
        "local_2080ti_final_7configs",
        "a100_final_7configs",
        "common_8datasets_7configs_local_vs_a100",
    )
    final_frontier += summarize_frontier(
        local_main_la,
        a100_la_common_ds,
        "local_2080ti_final_7configs",
        "a100_final_9configs",
        "common_8datasets_a100_9configs_vs_local_7configs",
    )

    final_build_memory = build_memory_summary(local_main, a100_base, local_phase, a100_phase)
    policy_effect = search_policy_effect(local_policy, a100_base)
    degree = degree_summary(local_degree, a100_degree, a100_base)
    extra = a100_extra_dataset_summary(a100_base, a100_phase, local_main)

    write_csv(out_dir / "final_matched_rows.csv", matched)
    write_csv(out_dir / "final_dataset_summary.csv", final_dataset_summary)
    write_csv(out_dir / "final_frontier_summary.csv", final_frontier)
    write_csv(out_dir / "final_build_memory.csv", final_build_memory)
    write_csv(out_dir / "search_policy_effect.csv", policy_effect)
    write_csv(out_dir / "degree_config_summary.csv", degree)
    write_csv(out_dir / "a100_extra_dataset_summary.csv", extra)
    write_report(
        out_dir,
        {
            "final_dataset_summary": final_dataset_summary,
            "final_frontier_summary": final_frontier,
            "final_build_memory": final_build_memory,
            "search_policy_effect": policy_effect,
            "degree_config_summary": degree,
            "a100_extra_dataset_summary": extra,
        },
    )
    print(out_dir)


if __name__ == "__main__":
    main()
