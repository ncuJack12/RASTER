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


def fvecs_shape(path: pathlib.Path):
    with path.open("rb") as f:
        dim = struct.unpack("<i", f.read(4))[0]
    rows = path.stat().st_size // ((dim + 1) * 4)
    return rows, dim


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


def metadata_value(workload: pathlib.Path, key: str):
    meta = workload / "metadata.json"
    if not meta.exists():
        return ""
    try:
        return json.loads(meta.read_text()).get(key, "")
    except json.JSONDecodeError:
        return ""


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


def edge_degree_sum(args, leaf_blocks: int):
    if leaf_blocks <= 1:
        return 0
    if not args.layer_adaptive_degree:
        return args.graph_degree * max(1, math.ceil(math.log2(leaf_blocks)))

    leaf_base = next_power_of_two(leaf_blocks)
    max_layer = max(0, math.ceil(math.log2(leaf_base)) - 1)
    min_degree = args.min_graph_degree or default_min_degree(args.graph_degree)
    return sum(
        layer_degree(args.graph_degree, min_degree, layer, max_layer, args.degree_granularity)
        for layer in range(max_layer + 1)
    )


def dataset_plan(args):
    total_gib, free_gib = gpu_memory_gib(args.gpu_id)
    selected = args.datasets or list(DATASETS)
    rows = []
    for name in selected:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset: {name}")
        base = ROOT / DATASETS[name]
        workload = ROOT / "generated_queries" / "order_range_raw_attr" / name / args.workload
        if not base.exists():
            status, reason = "skip", "missing_base"
            n_rows = dim = 0
            base_gib = edge_gib = est_peak_gib = 0.0
        elif not workload.exists():
            status, reason = "skip", "missing_workload"
            n_rows, dim = fvecs_shape(base)
            base_gib = n_rows * dim * 4 / 1024**3
            edge_gib = est_peak_gib = 0.0
        else:
            n_rows, dim = fvecs_shape(base)
            leaf_blocks = max(1, math.ceil(n_rows / args.leaf_size))
            base_gib = n_rows * dim * 4 / 1024**3
            edge_gib = n_rows * edge_degree_sum(args, leaf_blocks) * 4 / 1024**3
            est_peak_gib = base_gib + edge_gib + args.scratch_guard_gib
            safe_limit = min(args.max_est_peak_gib, total_gib * args.gpu_fraction)
            if args.include_risky or est_peak_gib <= safe_limit:
                status, reason = "pending", ""
            else:
                status, reason = "skip", f"est_peak_gib>{safe_limit:.2f}"
        rows.append(
            {
                "dataset": name,
                "status": status,
                "reason": reason,
                "base": str(base.relative_to(ROOT)) if base.exists() else DATASETS[name],
                "workload": str(workload.relative_to(ROOT)),
                "rows": n_rows,
                "dim": dim,
                "base_gib": f"{base_gib:.6f}",
                "edge_gib_est": f"{edge_gib:.6f}",
                "est_peak_gib": f"{est_peak_gib:.6f}",
                "layer_adaptive_degree": int(args.layer_adaptive_degree),
                "min_graph_degree": args.min_graph_degree,
                "min_intermediate_graph_degree": args.min_intermediate_graph_degree,
                "degree_granularity": args.degree_granularity,
                "target_width": metadata_value(workload, "target_width"),
                "topk": args.topk,
                "build_algo": args.build_algo,
                "exact_threads": args.exact_threads,
                "gpu_total_gib": f"{total_gib:.3f}",
                "gpu_free_gib_at_plan": f"{free_gib:.3f}",
            }
        )
    return rows


def run_one(args, run_root: pathlib.Path, row):
    dataset = row["dataset"]
    out_dir = run_root / f"{dataset}_{args.workload}"
    if out_dir.exists() and not args.force:
        return "skip_existing", "out_dir_exists", out_dir
    env = os.environ.copy()
    env.update(
        {
            "GPU_ID": args.gpu_id,
            "SAMPLE_INTERVAL": str(args.sample_interval),
            "OUT_DIR": str(out_dir),
            "RUN_ID": f"{dataset}_{args.workload}",
            "RANGE_CAGRA_BUILD_DIR": args.build_dir,
            "RANGE_CAGRA_SEGMENT_BASE": row["base"],
            "RANGE_CAGRA_SEGMENT_WORKLOAD": row["workload"],
            "RANGE_CAGRA_SEGMENT_MAX_QUERIES": str(args.max_queries),
            "RANGE_CAGRA_SEGMENT_TOPK": str(args.topk),
            "RANGE_CAGRA_SEGMENT_LEAF_SIZE": str(args.leaf_size),
            "RANGE_CAGRA_SEGMENT_BUILD_ALGO": args.build_algo,
            "RANGE_CAGRA_SEGMENT_GRAPH_DEGREE": str(args.graph_degree),
            "RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE": str(args.intermediate_graph_degree),
            "RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS": str(args.nn_descent_iters),
            "RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE": str(int(args.layer_adaptive_degree)),
            "RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE": str(args.min_graph_degree),
            "RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE": str(args.min_intermediate_graph_degree),
            "RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY": str(args.degree_granularity),
            "RANGE_CAGRA_SEGMENT_SEARCH_REPEATS": str(args.search_repeats),
            "RANGE_CAGRA_SEGMENT_SEARCH_SWEEP": args.search_sweep,
            "RANGE_CAGRA_SEGMENT_EXACT_THREADS": str(args.exact_threads),
            "RANGE_CAGRA_SEGMENT_GRAPH_THREADS": str(args.graph_threads),
        }
    )
    started = time.time()
    proc = subprocess.run(["bash", str(RUNNER)], cwd=ROOT, env=env, text=True)
    elapsed = time.time() - started
    if proc.returncode == 0:
        return "done", f"elapsed_seconds={elapsed:.3f}", out_dir
    return "failed", f"exit_code={proc.returncode};elapsed_seconds={elapsed:.3f}", out_dir


def aggregate(run_root: pathlib.Path, status_rows):
    sweep_rows = []
    summary_rows = []
    for row in status_rows:
        out_dir = pathlib.Path(row.get("out_dir", ""))
        if not out_dir.exists():
            continue
        for item in read_csv(out_dir / "sweep_summary.csv"):
            item = {"dataset": row["dataset"], **item}
            sweep_rows.append(item)
        for item in read_csv(out_dir / "summary.csv"):
            item = {"dataset": row["dataset"], **item}
            summary_rows.append(item)

    if sweep_rows:
        keys = ["dataset"] + [k for k in sweep_rows[0] if k != "dataset"]
        write_csv(run_root / "aggregate_sweep.csv", sweep_rows, keys)
    if summary_rows:
        keys = ["dataset"] + [k for k in summary_rows[0] if k != "dataset"]
        write_csv(run_root / "aggregate_summary.csv", summary_rows, keys)


def main():
    parser = argparse.ArgumentParser(description="Run range-CAGRA on all available RFANN datasets.")
    parser.add_argument("--run-id", default=time.strftime("all_datasets_%Y%m%d_%H%M%S"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workload", default="pos_w50")
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--build-dir", default=os.environ.get("RANGE_CAGRA_BUILD_DIR", "cpp/build"))
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--leaf-size", type=int, default=1000)
    parser.add_argument("--build-algo", choices=["flat_gnnd", "segmented_gnnd"], default="flat_gnnd")
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--intermediate-graph-degree", type=int, default=96)
    parser.add_argument("--nn-descent-iters", type=int, default=20)
    parser.add_argument("--layer-adaptive-degree", action="store_true")
    parser.add_argument("--min-graph-degree", type=int, default=0)
    parser.add_argument("--min-intermediate-graph-degree", type=int, default=0)
    parser.add_argument("--degree-granularity", type=int, default=8)
    parser.add_argument("--search-repeats", type=int, default=1)
    parser.add_argument("--exact-threads", type=int, default=128)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument(
        "--search-sweep",
        default="31:31:1:16;40:36:1:20;64:48:1:32",
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

    run_root = ROOT / "results" / "range_cagra" / "all_dataset_benchmark" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    plan_rows = dataset_plan(args)
    plan_keys = [
        "dataset",
        "status",
        "reason",
        "base",
        "workload",
        "rows",
        "dim",
        "base_gib",
        "edge_gib_est",
        "est_peak_gib",
        "layer_adaptive_degree",
        "min_graph_degree",
        "min_intermediate_graph_degree",
        "degree_granularity",
        "target_width",
        "topk",
        "build_algo",
        "exact_threads",
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
            print(f"skip {row['dataset']}: {row['reason']}", flush=True)
        elif args.dry_run:
            status_row["final_status"] = "dry_run"
            status_row["final_reason"] = ""
            status_row["out_dir"] = ""
            print(f"would_run {row['dataset']}", flush=True)
        else:
            print(
                f"run {row['dataset']}: rows={row['rows']} dim={row['dim']} "
                f"est_peak_gib={row['est_peak_gib']}",
                flush=True,
            )
            final_status, final_reason, out_dir = run_one(args, run_root, row)
            status_row["final_status"] = final_status
            status_row["final_reason"] = final_reason
            status_row["out_dir"] = str(out_dir.relative_to(ROOT))
            print(f"{final_status} {row['dataset']}: {final_reason}", flush=True)
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
