#!/usr/bin/env python3
import argparse
import csv
import os
import pathlib
import struct
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "results" / "range_cagra" / "run_msong_gpu_benchmark.sh"

DATASETS = {
    "audio": "data/audio/audio_base.fvecs",
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


def parse_result_line(line: str):
    row = {}
    for item in line.strip().split(","):
        if "=" not in item:
            row["record"] = item
            continue
        key, value = item.split("=", 1)
        row[key] = value
    return row


def read_result_rows(out_dir: pathlib.Path):
    path = out_dir / "result_lines.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(parse_result_line(line))
    return rows


def write_csv(path: pathlib.Path, rows, preferred_keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(preferred_keys)
    seen = set(keys)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_float(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def write_analysis(path: pathlib.Path, rows, baseline_leaf: int):
    selected = []
    by_key = {}
    for row in rows:
        key = (row.get("dataset", ""), row.get("search_config_id", ""))
        leaf = int(row["leaf_size"])
        if leaf == baseline_leaf:
            by_key[key] = row
        selected.append(row)

    lines = [
        "# Range-CAGRA Leaf Size Sweep",
        "",
        f"run_id={path.parent.name}",
        f"baseline_leaf_size={baseline_leaf}",
        "",
        "Rows below compare each leaf size against the baseline with the same dataset and search_config_id.",
        "",
        "| dataset | cfg | leaf | leaf*dim | build_s | search_s | exact_s | graph_s | qps | recall | build_delta | qps_delta | graph_delta | exact_delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        key = (row.get("dataset", ""), row.get("search_config_id", ""))
        base = by_key.get(key)
        build = metric_float(row, "build_seconds")
        qps = metric_float(row, "best_qps")
        graph = metric_float(row, "graph_seconds")
        exact = metric_float(row, "exact_seconds")
        if base is None:
            build_delta = qps_delta = graph_delta = exact_delta = ""
        else:
            base_build = metric_float(base, "build_seconds")
            base_qps = metric_float(base, "best_qps")
            base_graph = metric_float(base, "graph_seconds")
            base_exact = metric_float(base, "exact_seconds")
            build_delta = pct_delta(build, base_build)
            qps_delta = pct_delta(qps, base_qps)
            graph_delta = pct_delta(graph, base_graph)
            exact_delta = pct_delta(exact, base_exact)
        lines.append(
            "| {dataset} | {cfg} | {leaf} | {leaf_work} | {build} | {search} | {exact} | "
            "{graph} | {qps} | {recall} | {build_delta} | {qps_delta} | {graph_delta} | "
            "{exact_delta} |".format(
                dataset=row.get("dataset", ""),
                cfg=row.get("search_config_id", ""),
                leaf=row.get("leaf_size", ""),
                leaf_work=row.get("leaf_dim_work", ""),
                build=fmt(row.get("build_seconds", "")),
                search=fmt(row.get("best_search_seconds", "")),
                exact=fmt(row.get("exact_seconds", "")),
                graph=fmt(row.get("graph_seconds", "")),
                qps=fmt(row.get("best_qps", "")),
                recall=fmt(row.get("recall_at_k", "")),
                build_delta=build_delta,
                qps_delta=qps_delta,
                graph_delta=graph_delta,
                exact_delta=exact_delta,
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def fmt(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:.3f}"
    return f"{number:.6f}"


def pct_delta(value, base):
    if value is None or base is None or base == 0:
        return ""
    return f"{(value / base - 1.0) * 100.0:+.2f}%"


def run_one(args, run_root: pathlib.Path, dataset: str, base: str, workload: str, leaf_size: int):
    out_dir = run_root / f"{dataset}_{args.workload}_leaf{leaf_size}"
    if out_dir.exists() and not args.force:
        return "skip_existing", "out_dir_exists", out_dir

    env = os.environ.copy()
    env.update(
        {
            "GPU_ID": args.gpu_id,
            "SAMPLE_INTERVAL": str(args.sample_interval),
            "OUT_DIR": str(out_dir),
            "RUN_ID": f"{dataset}_{args.workload}_leaf{leaf_size}",
            "RANGE_CAGRA_SEGMENT_BASE": base,
            "RANGE_CAGRA_SEGMENT_WORKLOAD": workload,
            "RANGE_CAGRA_SEGMENT_MAX_QUERIES": str(args.max_queries),
            "RANGE_CAGRA_SEGMENT_LEAF_SIZE": str(leaf_size),
            "RANGE_CAGRA_SEGMENT_GRAPH_DEGREE": str(args.graph_degree),
            "RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE": str(args.intermediate_graph_degree),
            "RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS": str(args.nn_descent_iters),
            "RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE": str(int(args.layer_adaptive_degree)),
            "RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE": str(args.min_graph_degree),
            "RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE": str(
                args.min_intermediate_graph_degree
            ),
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


def aggregate(run_root: pathlib.Path, status_rows):
    result_rows = []
    for status in status_rows:
        out_dir = pathlib.Path(status.get("out_dir", ""))
        if not out_dir.exists():
            continue
        for row in read_result_rows(out_dir):
            rows = row.get("rows", status.get("rows", ""))
            dim = row.get("dim", status.get("dim", ""))
            leaf_size = status["leaf_size"]
            row = {
                "dataset": status["dataset"],
                "leaf_size": leaf_size,
                "leaf_dim_work": int(leaf_size) * int(dim) if str(dim).isdigit() else "",
                "out_dir": status["out_dir"],
                **row,
            }
            if "best_qps" not in row and metric_float(row, "best_search_seconds"):
                row["best_qps"] = str(float(row["nq"]) / float(row["best_search_seconds"]))
            result_rows.append(row)

    preferred = [
        "dataset",
        "leaf_size",
        "leaf_dim_work",
        "search_config_id",
        "rows",
        "dim",
        "nq",
        "graph_count",
        "edge_count",
        "build_seconds",
        "best_search_seconds",
        "best_qps",
        "recall_at_k",
        "filter_violations",
        "exact_seconds",
        "graph_seconds",
        "merge_seconds",
        "exact_vectors_scanned",
        "graph_node_tasks",
        "out_dir",
    ]
    if result_rows:
        write_csv(run_root / "aggregate_result_lines.csv", result_rows, preferred)
        write_analysis(run_root / "analysis.md", result_rows, int(status_rows[0]["leaf_size"]))


def main():
    parser = argparse.ArgumentParser(description="Sweep Range-CAGRA segment-tree leaf sizes.")
    parser.add_argument("--run-id", default=time.strftime("leaf_size_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--datasets", nargs="*", default=["msong"])
    parser.add_argument("--leaf-sizes", nargs="+", type=int, default=[1000, 1100, 1200, 1300])
    parser.add_argument("--workload", default="pos_w50")
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--intermediate-graph-degree", type=int, default=96)
    parser.add_argument("--nn-descent-iters", type=int, default=20)
    parser.add_argument("--layer-adaptive-degree", action="store_true")
    parser.add_argument("--min-graph-degree", type=int, default=0)
    parser.add_argument("--min-intermediate-graph-degree", type=int, default=0)
    parser.add_argument("--degree-granularity", type=int, default=8)
    parser.add_argument("--search-repeats", type=int, default=1)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument(
        "--search-sweep",
        default="31:31:1:16;40:36:1:20;64:48:1:32",
        help="semicolon separated ef:iterations:concurrency:entry_count configs",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not RUNNER.exists():
        raise SystemExit(f"missing runner: {RUNNER}")

    run_root = ROOT / "results" / "range_cagra" / "leaf_size_sweep" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for dataset in args.datasets:
        if dataset not in DATASETS:
            raise SystemExit(f"unknown dataset: {dataset}")
        base_path = ROOT / DATASETS[dataset]
        workload_path = ROOT / "generated_queries" / "order_range_raw_attr" / dataset / args.workload
        if not base_path.exists() or not workload_path.exists():
            reason = "missing_base" if not base_path.exists() else "missing_workload"
            for leaf_size in args.leaf_sizes:
                status_rows.append(
                    {
                        "dataset": dataset,
                        "leaf_size": leaf_size,
                        "rows": 0,
                        "dim": 0,
                        "final_status": "skip",
                        "final_reason": reason,
                        "out_dir": "",
                    }
                )
            continue
        rows, dim = fvecs_shape(base_path)
        for leaf_size in args.leaf_sizes:
            status = {
                "dataset": dataset,
                "leaf_size": leaf_size,
                "rows": rows,
                "dim": dim,
                "final_status": "",
                "final_reason": "",
                "out_dir": "",
            }
            if args.dry_run:
                status["final_status"] = "dry_run"
                print(
                    f"would_run dataset={dataset} leaf_size={leaf_size} "
                    f"leaf_dim_work={leaf_size * dim}",
                    flush=True,
                )
            else:
                print(
                    f"run dataset={dataset} leaf_size={leaf_size} "
                    f"leaf_dim_work={leaf_size * dim}",
                    flush=True,
                )
                final_status, final_reason, out_dir = run_one(
                    args,
                    run_root,
                    dataset,
                    str(base_path.relative_to(ROOT)),
                    str(workload_path.relative_to(ROOT)),
                    leaf_size,
                )
                status["final_status"] = final_status
                status["final_reason"] = final_reason
                status["out_dir"] = str(out_dir.relative_to(ROOT))
                print(f"{final_status} dataset={dataset} leaf_size={leaf_size}: {final_reason}")
            status_rows.append(status)
            write_csv(
                run_root / "status.csv",
                status_rows,
                ["dataset", "leaf_size", "rows", "dim", "final_status", "final_reason", "out_dir"],
            )
            aggregate(run_root, status_rows)

    aggregate(run_root, status_rows)
    print(f"run_root={run_root.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
