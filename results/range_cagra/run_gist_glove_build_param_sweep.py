#!/usr/bin/env python3
import argparse
import csv
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time

from run_all_dataset_benchmark import (
    DATASETS,
    ROOT,
    RUNNER,
    edge_degree_sum,
    fvecs_shape,
    gpu_memory_gib,
    read_csv,
    write_csv,
)


DEFAULT_CONFIGS = ",".join(
    [
        "d32_i128_it20:32:128:20",
        "d32_i128_it30:32:128:30",
        "d48_i128_it20:48:128:20",
        "d64_i128_it20:64:128:20",
    ]
)


def parse_configs(configs_text):
    configs = []
    for item in configs_text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise SystemExit(f"bad config '{item}', expected label:graph_degree:intermediate_degree:iters")
        label, graph_degree, intermediate_degree, iters = parts
        configs.append(
            {
                "config_label": label,
                "graph_degree": int(graph_degree),
                "intermediate_graph_degree": int(intermediate_degree),
                "nn_descent_iters": int(iters),
            }
        )
    if not configs:
        raise SystemExit("empty build config list")
    return configs


def estimate_peak_gib(args, rows, dim, leaf_size, graph_degree, scratch_guard_gib):
    leaf_blocks = max(1, math.ceil(rows / leaf_size))
    levels = max(1, math.ceil(math.log2(leaf_blocks)))
    base_gib = rows * dim * 4 / 1024**3
    edge_args = argparse.Namespace(
        layer_adaptive_degree=args.layer_adaptive_degree,
        graph_degree=graph_degree,
        min_graph_degree=args.min_graph_degree,
        degree_granularity=args.degree_granularity,
    )
    edge_gib = rows * edge_degree_sum(edge_args, leaf_blocks) * 4 / 1024**3
    return levels, base_gib, edge_gib, base_gib + edge_gib + scratch_guard_gib


def build_plan(args, configs):
    total_gib, free_gib = gpu_memory_gib(args.gpu_id)
    safe_limit = min(args.max_est_peak_gib, total_gib * args.gpu_fraction)
    rows = []
    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"unknown dataset: {dataset}")
        base = ROOT / DATASETS[dataset]
        workload = ROOT / "generated_queries" / "order_range_raw_attr" / dataset / args.workload
        if not base.exists():
            n_rows = dim = levels = 0
            base_gib = edge_gib = est_peak_gib = 0.0
            base_rel = DATASETS[dataset]
            workload_rel = str(workload.relative_to(ROOT))
            missing = "missing_base"
        else:
            n_rows, dim = fvecs_shape(base)
            base_rel = str(base.relative_to(ROOT))
            workload_rel = str(workload.relative_to(ROOT))
            missing = "" if workload.exists() else "missing_workload"
        for cfg in configs:
            if missing:
                levels = 0
                base_gib = edge_gib = est_peak_gib = 0.0
                status, reason = "skip", missing
            else:
                levels, base_gib, edge_gib, est_peak_gib = estimate_peak_gib(
                    args, n_rows, dim, args.leaf_size, cfg["graph_degree"], args.scratch_guard_gib
                )
                if args.include_risky or est_peak_gib <= safe_limit:
                    status, reason = "pending", ""
                else:
                    status, reason = "skip", f"est_peak_gib>{safe_limit:.2f}"
            rows.append(
                {
                    "dataset": dataset,
                    **cfg,
                    "status": status,
                    "reason": reason,
                    "base": base_rel,
                    "workload": workload_rel,
                    "rows": n_rows,
                    "dim": dim,
                    "levels_est": levels,
                    "base_gib": f"{base_gib:.6f}",
                    "edge_gib_est": f"{edge_gib:.6f}",
                    "est_peak_gib": f"{est_peak_gib:.6f}",
                    "layer_adaptive_degree": int(args.layer_adaptive_degree),
                    "min_graph_degree": args.min_graph_degree,
                    "min_intermediate_graph_degree": args.min_intermediate_graph_degree,
                    "degree_granularity": args.degree_granularity,
                    "gpu_total_gib": f"{total_gib:.3f}",
                    "gpu_free_gib_at_plan": f"{free_gib:.3f}",
                }
            )
    return rows


def run_one(args, run_root, row):
    label = row["config_label"]
    dataset = row["dataset"]
    out_dir = run_root / label / f"{dataset}_{args.workload}"
    if out_dir.exists() and not args.force:
        return "skip_existing", "out_dir_exists", out_dir

    env = os.environ.copy()
    env.update(
        {
            "GPU_ID": args.gpu_id,
            "SAMPLE_INTERVAL": str(args.sample_interval),
            "OUT_DIR": str(out_dir),
            "RUN_ID": f"{label}_{dataset}_{args.workload}",
            "RANGE_CAGRA_SEGMENT_BASE": row["base"],
            "RANGE_CAGRA_SEGMENT_WORKLOAD": row["workload"],
            "RANGE_CAGRA_SEGMENT_MAX_QUERIES": str(args.max_queries),
            "RANGE_CAGRA_SEGMENT_LEAF_SIZE": str(args.leaf_size),
            "RANGE_CAGRA_SEGMENT_GRAPH_DEGREE": str(row["graph_degree"]),
            "RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE": str(row["intermediate_graph_degree"]),
            "RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS": str(row["nn_descent_iters"]),
            "RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE": str(int(args.layer_adaptive_degree)),
            "RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE": str(args.min_graph_degree),
            "RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE": str(args.min_intermediate_graph_degree),
            "RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY": str(args.degree_granularity),
            "RANGE_CAGRA_SEGMENT_SEARCH_REPEATS": str(args.search_repeats),
            "RANGE_CAGRA_SEGMENT_SEARCH_SWEEP": args.search_sweep,
            "RANGE_CAGRA_SEGMENT_GRAPH_THREADS": str(args.graph_threads),
        }
    )
    started = time.time()
    proc = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True)
    elapsed = time.time() - started
    if proc.returncode == 0:
        return "done", f"elapsed_seconds={elapsed:.3f}", out_dir
    return "failed", f"exit_code={proc.returncode};elapsed_seconds={elapsed:.3f}", out_dir


def aggregate(run_root, status_rows):
    summary_rows = []
    sweep_rows = []
    phase_rows = []
    for row in status_rows:
        out_dir_text = row.get("out_dir", "")
        if not out_dir_text:
            continue
        out_dir = ROOT / out_dir_text
        if not out_dir.exists():
            continue
        prefix = {
            "config_label": row["config_label"],
            "dataset": row["dataset"],
            "graph_degree": row["graph_degree"],
            "intermediate_graph_degree": row["intermediate_graph_degree"],
            "nn_descent_iters": row["nn_descent_iters"],
            "layer_adaptive_degree": row.get("layer_adaptive_degree", ""),
            "min_graph_degree": row.get("min_graph_degree", ""),
            "min_intermediate_graph_degree": row.get("min_intermediate_graph_degree", ""),
            "degree_granularity": row.get("degree_granularity", ""),
        }
        for item in read_csv(out_dir / "summary.csv"):
            summary_rows.append({**prefix, **item})
        for item in read_csv(out_dir / "sweep_summary.csv"):
            sweep_rows.append({**prefix, **item})
        for item in read_csv(out_dir / "phase_gpu_summary.csv"):
            phase_rows.append({**prefix, **item})

    if summary_rows:
        keys = list(summary_rows[0])
        write_csv(run_root / "aggregate_summary.csv", summary_rows, keys)
    if sweep_rows:
        keys = list(sweep_rows[0])
        write_csv(run_root / "aggregate_sweep.csv", sweep_rows, keys)
    if phase_rows:
        keys = list(phase_rows[0])
        write_csv(run_root / "aggregate_phase_gpu.csv", phase_rows, keys)


def main():
    parser = argparse.ArgumentParser(description="Sweep build parameters on GIST and GloVe range-CAGRA.")
    parser.add_argument("--run-id", default=time.strftime("gist_glove_build_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--datasets", nargs="*", default=["gist", "glove-100"])
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--workload", default="pos_w50")
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--leaf-size", type=int, default=1000)
    parser.add_argument("--layer-adaptive-degree", action="store_true")
    parser.add_argument("--min-graph-degree", type=int, default=0)
    parser.add_argument("--min-intermediate-graph-degree", type=int, default=0)
    parser.add_argument("--degree-granularity", type=int, default=8)
    parser.add_argument("--search-repeats", type=int, default=1)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument(
        "--search-sweep",
        default="64:48:1:32;128:128:2:64;256:256:4:64",
        help="semicolon separated ef:iterations:concurrency:entry_count configs",
    )
    parser.add_argument("--scratch-guard-gib", type=float, default=3.5)
    parser.add_argument("--max-est-peak-gib", type=float, default=10.0)
    parser.add_argument("--gpu-fraction", type=float, default=0.9)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not RUNNER.exists():
        raise SystemExit(f"missing runner: {RUNNER}")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found")

    configs = parse_configs(args.configs)
    run_root = ROOT / "results" / "range_cagra" / "build_param_sweep" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    plan_rows = build_plan(args, configs)
    plan_keys = [
        "dataset",
        "config_label",
        "graph_degree",
        "intermediate_graph_degree",
        "nn_descent_iters",
        "status",
        "reason",
        "base",
        "workload",
        "rows",
        "dim",
        "levels_est",
        "base_gib",
        "edge_gib_est",
        "est_peak_gib",
        "layer_adaptive_degree",
        "min_graph_degree",
        "min_intermediate_graph_degree",
        "degree_granularity",
        "gpu_total_gib",
        "gpu_free_gib_at_plan",
    ]
    write_csv(run_root / "plan.csv", plan_rows, plan_keys)
    print(f"run_root={run_root.relative_to(ROOT)}")
    print((run_root / "plan.csv").relative_to(ROOT))

    status_rows = []
    for row in plan_rows:
        status_row = dict(row)
        if row["status"] != "pending":
            status_row["final_status"] = row["status"]
            status_row["final_reason"] = row["reason"]
            status_row["out_dir"] = ""
            print(f"skip {row['config_label']} {row['dataset']}: {row['reason']}", flush=True)
        elif args.dry_run:
            status_row["final_status"] = "dry_run"
            status_row["final_reason"] = ""
            status_row["out_dir"] = ""
            print(f"would_run {row['config_label']} {row['dataset']}", flush=True)
        else:
            print(
                f"run {row['config_label']} {row['dataset']}: rows={row['rows']} dim={row['dim']} "
                f"est_peak_gib={row['est_peak_gib']}",
                flush=True,
            )
            final_status, final_reason, out_dir = run_one(args, run_root, row)
            status_row["final_status"] = final_status
            status_row["final_reason"] = final_reason
            status_row["out_dir"] = str(out_dir.relative_to(ROOT))
            print(f"{final_status} {row['config_label']} {row['dataset']}: {final_reason}", flush=True)
        status_rows.append(status_row)
        write_csv(
            run_root / "status.csv",
            status_rows,
            plan_keys + ["final_status", "final_reason", "out_dir"],
        )
        aggregate(run_root, status_rows)

    aggregate(run_root, status_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
