#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "results" / "range_cagra" / "run_msong_gpu_benchmark.sh"

DATASETS = {
    "audio": "data/audio/audio_base.fvecs",
    "arxiv-for-fanns-large": "data/arxiv-for-fanns-large/arxiv-for-fanns-large_base.fvecs",
    "deep": "data/deep/deep-base-1M.fvecs",
    "enron": "data/enron/enron_base.fvecs",
    "gist": "data/gist/gist_base.fvecs",
    "glove-100": "data/glove-100/glove-100_base.fvecs",
    "msong": "data/msong/msong_base.fvecs",
    "sift": "data/sift/sift_base.fvecs",
    "text2image": "data/text2image/text2image_base.fvecs",
    "wit": "data/wit/wiki_image-base-1M.fvecs",
    "yt8mAudio": "data/yt8mAudio/yt8m_audio-base-1M.fvecs",
}

DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_QUERY_ROOT = ROOT / "generated_queries" / "order_range_raw_attr"

DEFAULT_DEGREE_CONFIGS = ";".join(
    [
        "uniform_d32_i96:0:32:96:0:0:8",
        "adaptive_d32_i96_min16_i48:1:32:96:16:48:8",
        "adaptive_d32_i96_min8_i24:1:32:96:8:24:8",
        "adaptive_d32_i96_min4_i12:1:32:96:4:12:8",
        "adaptive_d24_i72_min8_i24:1:24:72:8:24:8",
        "adaptive_d16_i48_min4_i16:1:16:48:4:16:8",
    ]
)


def fvecs_shape(path: pathlib.Path):
    with path.open("rb") as f:
        dim = struct.unpack("<i", f.read(4))[0]
    rows = path.stat().st_size // ((dim + 1) * 4)
    return rows, dim


def read_csv(path: pathlib.Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataset_base_path(args, dataset: str):
    path = pathlib.Path(DATASETS[dataset])
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "data":
        path = pathlib.Path(*parts[1:])
    return pathlib.Path(args.data_root) / path


def workload_path(args, dataset: str, workload: str):
    return pathlib.Path(args.query_root) / dataset / workload


def display_path(path: pathlib.Path):
    return str(path)


def metadata(args, dataset: str, workload: str):
    paths = [
        workload_path(args, dataset, workload) / "metadata.json",
        pathlib.Path(args.query_root) / dataset / "metadata.json",
    ]
    for path in paths:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}
    return {}


def next_power_of_two(value: int):
    out = 1
    while out < value:
        out <<= 1
    return out


def round_up_multiple(value: int, multiple: int):
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def default_min_degree(max_degree: int):
    return max(1, max_degree // 4)


def layer_degree(max_degree: int, min_degree: int, layer: int, max_layer: int, granularity: int):
    min_degree = min(min_degree, max_degree)
    if max_layer <= 0:
        return max_degree
    if layer == 0:
        return min_degree
    if layer == max_layer:
        return max_degree
    degree = min_degree + ((max_degree - min_degree) * layer + max_layer // 2) // max_layer
    return max(min_degree, min(max_degree, round_up_multiple(degree, granularity)))


def degree_sum_for_config(config, leaf_blocks: int):
    if leaf_blocks <= 1:
        return 0
    if not config["layer_adaptive_degree"]:
        return config["graph_degree"] * max(1, math.ceil(math.log2(leaf_blocks)))
    leaf_base = next_power_of_two(leaf_blocks)
    max_layer = max(0, math.ceil(math.log2(leaf_base)) - 1)
    min_degree = config["min_graph_degree"] or default_min_degree(config["graph_degree"])
    return sum(
        layer_degree(config["graph_degree"], min_degree, layer, max_layer, config["degree_granularity"])
        for layer in range(max_layer + 1)
    )


def gpu_memory_gib(gpu_id: str):
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu_id,
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    total_mb, free_mb = [float(x.strip()) for x in out.split(",")]
    return total_mb / 1024.0, free_mb / 1024.0


def parse_degree_configs(text: str):
    configs = []
    for item in text.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (7, 8):
            raise SystemExit(
                f"bad degree config '{item}', expected "
                "label:adaptive:graph_degree:intermediate_degree:min_graph:min_intermediate:"
                "granularity[:nn_descent_iters]"
            )
        label, adaptive, graph, intermediate, min_graph, min_intermediate, granularity = parts[:7]
        nn_iters = int(parts[7]) if len(parts) == 8 else 0
        configs.append(
            {
                "config_label": label,
                "layer_adaptive_degree": int(adaptive) != 0,
                "graph_degree": int(graph),
                "intermediate_graph_degree": int(intermediate),
                "min_graph_degree": int(min_graph),
                "min_intermediate_graph_degree": int(min_intermediate),
                "degree_granularity": int(granularity),
                "nn_descent_iters": nn_iters,
            }
        )
    if not configs:
        raise SystemExit("empty degree config list")
    return configs


def parse_int_list(text: str):
    values = []
    for item in text.replace(",", " ").split():
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise SystemExit("empty integer list")
    return values


def parse_workload_list(text: str):
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def base_config(args):
    return {
        "config_label": args.config_label,
        "layer_adaptive_degree": args.layer_adaptive_degree,
        "graph_degree": args.graph_degree,
        "intermediate_graph_degree": args.intermediate_graph_degree,
        "min_graph_degree": args.min_graph_degree,
        "min_intermediate_graph_degree": args.min_intermediate_graph_degree,
        "degree_granularity": args.degree_granularity,
        "nn_descent_iters": args.nn_descent_iters,
    }


def estimate_task(args, dataset: str, workload: str, leaf_size: int, config, scratch_guard_gib: float):
    base = dataset_base_path(args, dataset)
    workload_dir = workload_path(args, dataset, workload)
    if not base.exists():
        return {
            "status": "skip",
            "reason": "missing_base",
            "base": display_path(base),
            "workload": display_path(workload_dir),
            "rows": 0,
            "dim": 0,
        }
    rows, dim = fvecs_shape(base)
    if not workload_dir.exists():
        return {
            "status": "skip",
            "reason": "missing_workload",
            "base": display_path(base),
            "workload": display_path(workload_dir),
            "rows": rows,
            "dim": dim,
        }

    meta = metadata(args, dataset, workload)
    target_width = int(float(meta.get("target_width", 0) or 0))
    leaf_blocks = max(1, math.ceil(rows / leaf_size))
    leaf_base = next_power_of_two(leaf_blocks)
    graph_count_est = max(0, leaf_blocks - 1)
    tree_height = max(0, math.ceil(math.log2(leaf_base)))
    base_gib = rows * dim * 4 / 1024**3
    edge_gib = rows * degree_sum_for_config(config, leaf_blocks) * 4 / 1024**3
    exact_vectors_est = min(target_width, 2 * leaf_size) if target_width > 0 else 2 * leaf_size
    exact_dim_work_est = exact_vectors_est * dim
    est_peak_gib = base_gib + edge_gib + scratch_guard_gib
    return {
        "status": "pending",
        "reason": "",
        "base": display_path(base),
        "workload": display_path(workload_dir),
        "rows": rows,
        "dim": dim,
        "target_width": target_width,
        "leaf_blocks": leaf_blocks,
        "leaf_base": leaf_base,
        "tree_height": tree_height,
        "graph_count_est": graph_count_est,
        "base_gib": f"{base_gib:.6f}",
        "edge_gib_est": f"{edge_gib:.6f}",
        "est_peak_gib": f"{est_peak_gib:.6f}",
        "exact_vectors_est": exact_vectors_est,
        "exact_dim_work_est": exact_dim_work_est,
        "leaf_dim_work": leaf_size * dim,
    }


def build_tasks(args):
    datasets = args.datasets
    if args.sweep == "degree":
        degree_configs = parse_degree_configs(args.degree_configs)
        leaf_sizes = [args.leaf_size]
    elif args.sweep == "leaf":
        degree_configs = [base_config(args)]
        leaf_sizes = parse_int_list(args.leaf_sizes)
    else:
        degree_configs = parse_degree_configs(args.degree_configs)
        leaf_sizes = parse_int_list(args.leaf_sizes)

    total_gib, free_gib = gpu_memory_gib(args.gpu_id)
    safe_limit = min(args.max_est_peak_gib, total_gib * args.gpu_fraction)
    tasks = []
    for dataset in datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"unknown dataset: {dataset}")
        for config in degree_configs:
            for leaf_size in leaf_sizes:
                est = estimate_task(args, dataset, args.workload, leaf_size, config, args.scratch_guard_gib)
                if est["status"] == "pending" and not args.include_risky:
                    if float(est["est_peak_gib"]) > safe_limit:
                        est["status"] = "skip"
                        est["reason"] = f"est_peak_gib>{safe_limit:.2f}"
                label = config["config_label"]
                if args.sweep == "leaf":
                    label = f"{label}_leaf{leaf_size}"
                task = {
                    "sweep": args.sweep,
                    "dataset": dataset,
                    "workload_name": args.workload,
                    "leaf_size": leaf_size,
                    "topk": args.topk,
                    "build_algo": args.build_algo,
                    "exact_threads": args.exact_threads,
                    **config,
                    **est,
                    "gpu_total_gib": f"{total_gib:.3f}",
                    "gpu_free_gib_at_plan": f"{free_gib:.3f}",
                    "task_label": label,
                }
                if not task.get("nn_descent_iters"):
                    task["nn_descent_iters"] = args.nn_descent_iters
                tasks.append(task)
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]
    return tasks


PLAN_KEYS = [
    "sweep",
    "dataset",
    "workload_name",
    "workload_sweep",
    "task_label",
    "config_label",
    "leaf_size",
    "topk",
    "build_algo",
    "layer_adaptive_degree",
    "graph_degree",
    "intermediate_graph_degree",
    "min_graph_degree",
    "min_intermediate_graph_degree",
    "degree_granularity",
    "nn_descent_iters",
    "exact_threads",
    "status",
    "reason",
    "base",
    "workload",
    "rows",
    "dim",
    "target_width",
    "leaf_blocks",
    "leaf_base",
    "tree_height",
    "graph_count_est",
    "base_gib",
    "edge_gib_est",
    "est_peak_gib",
    "exact_vectors_est",
    "exact_dim_work_est",
    "leaf_dim_work",
    "gpu_total_gib",
    "gpu_free_gib_at_plan",
]


def task_out_dir(run_root: pathlib.Path, task):
    parts = [task["sweep"], task["dataset"], task["task_label"]]
    if task["sweep"] != "leaf":
        parts.append(f"leaf{task['leaf_size']}")
    return run_root / "_".join(str(p) for p in parts)


def has_data_row(path: pathlib.Path):
    if not path.exists():
        return False
    with path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return next(reader, None) is not None


def completed_out_dir(out_dir: pathlib.Path):
    required = [
        "summary.csv",
        "sweep_summary.csv",
        "phase_gpu_summary.csv",
        "result_lines.csv",
    ]
    return all(has_data_row(out_dir / name) for name in required)


def run_task(args, run_root: pathlib.Path, task):
    out_dir = task_out_dir(run_root, task)
    if out_dir.exists() and not args.force:
        if completed_out_dir(out_dir):
            return "skip_existing", "completed_out_dir_exists", out_dir
        print(f"rerun_incomplete {task['task_label']} {task['dataset']}: out_dir_exists", flush=True)

    env = os.environ.copy()
    env.update(
        {
            "GPU_ID": args.gpu_id,
            "SAMPLE_INTERVAL": str(args.sample_interval),
            "OUT_DIR": str(out_dir),
            "RUN_ID": out_dir.name,
            "RANGE_CAGRA_BUILD_DIR": args.build_dir,
            "RANGE_CAGRA_SKIP_BUILD": str(int(args.skip_build)),
            "RANGE_CAGRA_SEGMENT_BASE": task["base"],
            "RANGE_CAGRA_SEGMENT_WORKLOAD": task["workload"],
            "RANGE_CAGRA_SEGMENT_WORKLOAD_SWEEP": task.get("workload_sweep", ""),
            "RANGE_CAGRA_SEGMENT_MAX_QUERIES": str(args.max_queries),
            "RANGE_CAGRA_SEGMENT_TOPK": str(args.topk),
            "RANGE_CAGRA_SEGMENT_LEAF_SIZE": str(task["leaf_size"]),
            "RANGE_CAGRA_SEGMENT_BUILD_ALGO": args.build_algo,
            "RANGE_CAGRA_SEGMENT_GRAPH_DEGREE": str(task["graph_degree"]),
            "RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE": str(task["intermediate_graph_degree"]),
            "RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS": str(task["nn_descent_iters"]),
            "RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE": str(int(task["layer_adaptive_degree"])),
            "RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE": str(task["min_graph_degree"]),
            "RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE": str(
                task["min_intermediate_graph_degree"]
            ),
            "RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY": str(task["degree_granularity"]),
            "RANGE_CAGRA_SEGMENT_SEARCH_REPEATS": str(args.search_repeats),
            "RANGE_CAGRA_SEGMENT_SEARCH_SWEEP": args.search_sweep,
            "RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE_SWEEP": args.search_schedule_sweep,
            "RANGE_CAGRA_SEGMENT_EXACT_THREADS": str(args.exact_threads),
            "RANGE_CAGRA_SEGMENT_GRAPH_THREADS": str(args.graph_threads),
            "RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY": args.search_iteration_policy,
            "RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY_SWEEP": args.search_iteration_policy_sweep,
            "RANGE_CAGRA_SEGMENT_LOW_LAYER_SEARCH_LAYERS": str(args.low_layer_search_layers),
            "RANGE_CAGRA_SEGMENT_LOW_LAYER_GRAPH_ITERATIONS": str(
                args.low_layer_graph_iterations
            ),
            "RANGE_CAGRA_SEGMENT_UPPER_LAYER_SEARCH_LAYERS": str(args.upper_layer_search_layers),
            "RANGE_CAGRA_SEGMENT_UPPER_LAYER_GRAPH_ITERATIONS": str(
                args.upper_layer_graph_iterations
            ),
            "RANGE_CAGRA_SEGMENT_ADAPTIVE_MIN_GRAPH_ITERATIONS": str(
                args.adaptive_min_graph_iterations
            ),
            "RANGE_CAGRA_SEGMENT_ADAPTIVE_MAX_GRAPH_ITERATIONS": str(
                args.adaptive_max_graph_iterations
            ),
            "RANGE_CAGRA_SEGMENT_ADAPTIVE_ITERATION_GRANULARITY": str(
                args.adaptive_iteration_granularity
            ),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out_dir / "runner_stdout.log").open("w") as log:
        proc = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - started
    if proc.returncode == 0:
        return "done", f"elapsed_seconds={elapsed:.3f}", out_dir
    return "failed", f"exit_code={proc.returncode};elapsed_seconds={elapsed:.3f}", out_dir


def aggregate(run_root: pathlib.Path, status_rows):
    summary_rows = []
    sweep_rows = []
    phase_rows = []
    for status in status_rows:
        out_dir_text = status.get("out_dir", "")
        if not out_dir_text:
            continue
        out_dir = ROOT / out_dir_text
        if not out_dir.exists():
            continue
        prefix = {
            key: status.get(key, "")
            for key in [
                "sweep",
                "dataset",
                "task_label",
                "config_label",
                "leaf_size",
                "topk",
                "build_algo",
                "layer_adaptive_degree",
                "graph_degree",
                "intermediate_graph_degree",
                "min_graph_degree",
                "min_intermediate_graph_degree",
                "degree_granularity",
                "nn_descent_iters",
                "exact_threads",
                "rows",
                "dim",
                "target_width",
                "leaf_blocks",
                "graph_count_est",
                "edge_gib_est",
                "exact_vectors_est",
                "exact_dim_work_est",
                "leaf_dim_work",
                "out_dir",
            ]
        }
        for item in read_csv(out_dir / "summary.csv"):
            summary_rows.append({**prefix, **item})
        for item in read_csv(out_dir / "sweep_summary.csv"):
            sweep_rows.append({**prefix, **item})
        for item in read_csv(out_dir / "phase_gpu_summary.csv"):
            phase_rows.append({**prefix, **item})

    if summary_rows:
        write_csv(run_root / "aggregate_summary.csv", summary_rows, list(summary_rows[0]))
    if sweep_rows:
        write_csv(run_root / "aggregate_sweep.csv", sweep_rows, list(sweep_rows[0]))
    if phase_rows:
        write_csv(run_root / "aggregate_phase_gpu.csv", phase_rows, list(phase_rows[0]))
    write_analysis(run_root, summary_rows, sweep_rows)


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def write_analysis(run_root: pathlib.Path, summary_rows, sweep_rows):
    lines = ["# Range-CAGRA segment-tree parameter sweep", ""]
    if not summary_rows:
        lines.append("No completed summary rows yet.")
        (run_root / "analysis.md").write_text("\n".join(lines))
        return

    lines.append("## Build summary")
    lines.append("")
    lines.append(
        "| sweep | dataset | task | leaf | graph_avg | inter_avg | edge_gib | build_s | peak_mem_mb | recall | qps |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    by_task = {}
    for row in summary_rows:
        by_task[(row["dataset"], row["task_label"])] = row
    for row in sorted(
        summary_rows,
        key=lambda r: (r.get("dataset", ""), as_float(r, "build_seconds"), r.get("task_label", "")),
    ):
        phase_peak = ""
        for phase in read_csv(ROOT / row.get("out_dir", "") / "phase_gpu_summary.csv"):
            if phase.get("phase") == "build":
                phase_peak = phase.get("peak_memory_used_mb", "")
                break
        lines.append(
            "| {sweep} | {dataset} | {task} | {leaf} | {gavg} | {iavg} | {edge} | {build} | {mem} | {recall} | {qps} |".format(
                sweep=row.get("sweep", ""),
                dataset=row.get("dataset", ""),
                task=row.get("task_label", ""),
                leaf=row.get("leaf_size", ""),
                gavg=row.get("graph_degree_avg", ""),
                iavg=row.get("intermediate_graph_degree_avg", ""),
                edge=row.get("edge_gib", row.get("edge_gib_est", "")),
                build=row.get("build_seconds", ""),
                mem=phase_peak,
                recall=row.get("recall_at_k", ""),
                qps=row.get("best_qps", ""),
            )
        )

    if sweep_rows:
        target_recall = 0.995
        candidates = [r for r in sweep_rows if as_float(r, "recall_at_k") >= target_recall]
        lines.extend(["", f"## Search candidates recall >= {target_recall}", ""])
        if candidates:
            lines.append(
                "| dataset | task | cfg | policy | leaf | recall | qps | search_s | exact_s | graph_s |"
            )
            lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
            best = {}
            for row in candidates:
                key = (
                    row.get("dataset", ""),
                    row.get("search_config_id", ""),
                    row.get("search_iteration_policy", ""),
                )
                if key not in best or as_float(row, "best_qps") > as_float(best[key], "best_qps"):
                    best[key] = row
            for row in sorted(best.values(), key=lambda r: (r.get("dataset", ""), r.get("search_config_id", ""))):
                lines.append(
                    "| {dataset} | {task} | {cfg} | {policy} | {leaf} | {recall} | {qps} | {search} | {exact} | {graph} |".format(
                        dataset=row.get("dataset", ""),
                        task=row.get("task_label", ""),
                        cfg=row.get("search_config_id", ""),
                        policy=row.get("search_iteration_policy", ""),
                        leaf=row.get("leaf_size", ""),
                        recall=row.get("recall_at_k", ""),
                        qps=row.get("best_qps", ""),
                        search=row.get("best_search_seconds", ""),
                        exact=row.get("exact_seconds", ""),
                        graph=row.get("graph_seconds", ""),
                    )
                )
        else:
            lines.append("No search row reached the recall threshold.")
    (run_root / "analysis.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run range-CAGRA segment-tree parameter sweeps.")
    parser.add_argument("--run-id", default=time.strftime("segment_tree_param_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--sweep", choices=["degree", "leaf", "combined"], required=True)
    parser.add_argument("--datasets", nargs="+", default=["msong"])
    parser.add_argument("--workload", default="pos_w50")
    parser.add_argument(
        "--workload-sweep",
        default="",
        help="Optional workload names searched after one build per dataset/config, e.g. 'pos_w01 pos_w05 ind_w01'.",
    )
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--build-dir", default=os.environ.get("RANGE_CAGRA_BUILD_DIR", "cpp/build"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--query-root", default=str(DEFAULT_QUERY_ROOT))
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--leaf-size", type=int, default=1000)
    parser.add_argument("--leaf-sizes", default="512 768 1000 1500 2000 3000 4000")
    parser.add_argument("--config-label", default="adaptive_d32_i96_min8_i24")
    parser.add_argument("--build-algo", choices=["flat_gnnd", "segmented_gnnd"], default="flat_gnnd")
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--intermediate-graph-degree", type=int, default=96)
    parser.add_argument("--layer-adaptive-degree", action="store_true")
    parser.add_argument("--min-graph-degree", type=int, default=8)
    parser.add_argument("--min-intermediate-graph-degree", type=int, default=24)
    parser.add_argument("--degree-granularity", type=int, default=8)
    parser.add_argument("--degree-configs", default=DEFAULT_DEGREE_CONFIGS)
    parser.add_argument("--nn-descent-iters", type=int, default=20)
    parser.add_argument("--search-repeats", type=int, default=1)
    parser.add_argument("--exact-threads", type=int, default=128)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--search-sweep", default="64:48:1:32;128:96:2:64")
    parser.add_argument("--search-schedule-sweep", default="")
    parser.add_argument(
        "--search-iteration-policy",
        choices=["", "uniform", "lower_layers", "upper_layers", "layer_adaptive"],
        default="",
    )
    parser.add_argument("--search-iteration-policy-sweep", default="")
    parser.add_argument("--low-layer-search-layers", type=int, default=0)
    parser.add_argument("--low-layer-graph-iterations", type=int, default=0)
    parser.add_argument("--upper-layer-search-layers", type=int, default=0)
    parser.add_argument("--upper-layer-graph-iterations", type=int, default=0)
    parser.add_argument("--adaptive-min-graph-iterations", type=int, default=0)
    parser.add_argument("--adaptive-max-graph-iterations", type=int, default=0)
    parser.add_argument("--adaptive-iteration-granularity", type=int, default=1)
    parser.add_argument("--scratch-guard-gib", type=float, default=3.5)
    parser.add_argument("--max-est-peak-gib", type=float, default=10.0)
    parser.add_argument("--gpu-fraction", type=float, default=0.9)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.data_root = str(pathlib.Path(args.data_root).expanduser().resolve())
    args.query_root = str(pathlib.Path(args.query_root).expanduser().resolve())

    workload_sweep = parse_workload_list(args.workload_sweep)
    if workload_sweep:
        args.workload_sweep = " ".join(workload_sweep)

    if not RUNNER.exists():
        raise SystemExit(f"missing runner: {RUNNER}")
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi not found")

    run_root = ROOT / "results" / "range_cagra" / "segment_tree_param_sweep" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    if workload_sweep:
        for task in tasks:
            task["workload_sweep"] = ";".join(
                display_path(workload_path(args, task["dataset"], workload))
                for workload in workload_sweep
            )
    write_csv(run_root / "plan.csv", tasks, PLAN_KEYS)
    print(f"run_root={run_root.relative_to(ROOT)}", flush=True)
    print((run_root / "plan.csv").relative_to(ROOT), flush=True)

    status_rows = []
    for task in tasks:
        status = dict(task)
        if task["status"] != "pending":
            status.update({"final_status": task["status"], "final_reason": task["reason"], "out_dir": ""})
            print(f"skip {task['task_label']} {task['dataset']}: {task['reason']}", flush=True)
        elif args.dry_run:
            status.update({"final_status": "dry_run", "final_reason": "", "out_dir": ""})
            print(
                f"would_run {task['task_label']} {task['dataset']} leaf={task['leaf_size']} "
                f"leaf_dim_work={task.get('leaf_dim_work', '')} est_peak_gib={task.get('est_peak_gib', '')}",
                flush=True,
            )
        else:
            print(
                f"run {task['task_label']} {task['dataset']} leaf={task['leaf_size']} "
                f"dim={task.get('dim', '')} est_peak_gib={task.get('est_peak_gib', '')}",
                flush=True,
            )
            final_status, final_reason, out_dir = run_task(args, run_root, task)
            status.update(
                {
                    "final_status": final_status,
                    "final_reason": final_reason,
                    "out_dir": str(out_dir.relative_to(ROOT)),
                }
            )
            print(f"{final_status} {task['task_label']} {task['dataset']}: {final_reason}", flush=True)
        status_rows.append(status)
        write_csv(
            run_root / "status.csv",
            status_rows,
            PLAN_KEYS + ["final_status", "final_reason", "out_dir"],
        )
        aggregate(run_root, status_rows)

    aggregate(run_root, status_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
