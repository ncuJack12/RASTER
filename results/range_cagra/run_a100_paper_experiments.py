#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]
SWEEP_RUNNER = ROOT / "results" / "range_cagra" / "run_segment_tree_param_sweep.py"

DEFAULT_DATASETS = [
    "audio",
    "deep",
    "enron",
    "gist",
    "glove-100",
    "msong",
    "sift",
    "text2image",
    "wit",
    "yt8mAudio",
]
DEFAULT_SENSITIVITY_DATASETS = ["audio", "deep", "enron", "gist", "glove-100", "msong", "sift", "yt8mAudio"]
DEFAULT_SCALABILITY_DATASETS = ["deep", "msong", "sift", "yt8mAudio"]
DEFAULT_LARGE_DATASETS = ["arxiv-for-fanns-large"]

MAIN_BUILD_CONFIGS = ";".join(
    [
        "adaptive_d16_i64_min4_i16_it10:1:16:64:4:16:8:10",
        "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
        "adaptive_d48_i128_min12_i32_it20:1:48:128:12:32:8:20",
        "adaptive_d64_i128_min16_i32_it20:1:64:128:16:32:8:20",
    ]
)
BUILD_ABLATION_CONFIGS = ";".join(
    [
        "uniform_d32_i96_it20:0:32:96:0:0:8:20",
        "adaptive_d32_i96_min16_i48_it20:1:32:96:16:48:8:20",
        "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
        "adaptive_d32_i96_min4_i12_it20:1:32:96:4:12:8:20",
    ]
)
ARXIV_BUILD_CONFIGS = ";".join(
    [
        "d16_i64_it10:0:16:64:0:0:8:10",
        "d32_i96_it20:0:32:96:0:0:8:20",
        "d48_i128_it20:0:48:128:0:0:8:20",
        "d64_i128_it20:0:64:128:0:0:8:20",
    ]
)
DIAGNOSTIC_BUILD_CONFIGS = ";".join(
    [
        "adaptive_d48_i128_min12_i32_it20:1:48:128:12:32:8:20",
        "adaptive_d64_i128_min16_i32_it20:1:64:128:16:32:8:20",
        "adaptive_d96_i192_min24_i48_it30:1:96:192:24:48:8:30",
    ]
)

DEFAULT_SEARCH_SWEEP = "16:12:1:16;24:16:1:16;32:24:1:16;48:32:1:24;64:48:1:32;96:64:2:32;128:96:2:32"
ARXIV_SEARCH_SWEEP = "32:32:1:16;64:40:1:24;96:96:2:32;128:128:2:32"
POLICY_SEARCH_SWEEP = "16:12:1:16;24:16:1:16;32:24:1:16;48:32:1:24;64:48:1:32"
DEFAULT_LEAF_SIZES = "16 32 64 96 128 256 512 1000"


def parse_csv_list(text):
    out = []
    for item in text.replace(",", " ").split():
        item = item.strip()
        if item:
            out.append(item)
    return out


def workload_names(preset):
    widths = ["01", "05", "10", "20", "50"]
    if preset == "pos50":
        return ["pos_w50"]
    if preset == "full":
        widths = ["01", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50"]
    if preset not in ("core", "full"):
        raise ValueError(f"unknown workload preset: {preset}")
    return [f"{mode}_w{width}" for mode in ("pos", "ind", "neg") for width in widths]


def shard_items(items, shard_count):
    shard_count = max(1, int(shard_count))
    if shard_count == 1:
        return [(0, items)]
    shards = [[] for _ in range(shard_count)]
    for i, item in enumerate(items):
        shards[i % shard_count].append(item)
    return [(i, shard) for i, shard in enumerate(shards) if shard]


def gpu_id_for(args, row_index):
    ids = parse_csv_list(args.gpu_ids or args.gpu_id)
    if not ids:
        ids = [args.gpu_id]
    return ids[row_index % len(ids)]


def add_common_args(
    cmd, args, run_id, datasets, workload, sweep, leaf_size, degree_configs, search_sweep, gpu_id
):
    cmd.extend(
        [
            str(SWEEP_RUNNER.relative_to(ROOT)),
            "--run-id",
            run_id,
            "--sweep",
            sweep,
            "--datasets",
            *datasets,
            "--workload",
            workload,
            "--gpu-id",
            gpu_id,
            "--build-dir",
            args.build_dir,
            "--max-queries",
            str(args.max_queries),
            "--topk",
            str(args.topk),
            "--leaf-size",
            str(leaf_size),
            "--build-algo",
            args.build_algo,
            "--degree-configs",
            degree_configs,
            "--search-sweep",
            search_sweep,
            "--search-repeats",
            str(args.search_repeats),
            "--exact-threads",
            str(args.exact_threads),
            "--graph-threads",
            str(args.graph_threads),
            "--sample-interval",
            str(args.sample_interval),
            "--scratch-guard-gib",
            str(args.scratch_guard_gib),
            "--max-est-peak-gib",
            str(args.max_est_peak_gib),
            "--gpu-fraction",
            str(args.gpu_fraction),
        ]
    )
    if args.include_risky:
        cmd.append("--include-risky")
    if args.inner_dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")


def make_command(
    args,
    phase,
    workload,
    datasets,
    sweep,
    leaf_size,
    degree_configs,
    search_sweep,
    extra=None,
    workload_sweep=None,
    workload_label=None,
    gpu_id=None,
):
    sweep_names = workload_sweep or []
    label = workload_label or ("multi" if sweep_names else workload)
    safe_label = label.replace("_", "")
    run_id = f"{args.run_id}_{phase}_{safe_label}"
    cmd = ["python3"]
    assigned_gpu = gpu_id or args.gpu_id
    add_common_args(
        cmd, args, run_id, datasets, workload, sweep, leaf_size, degree_configs, search_sweep, assigned_gpu
    )
    if sweep_names:
        cmd.extend(["--workload-sweep", " ".join(sweep_names)])
    if extra:
        cmd.extend(extra)
    return {
        "phase": phase,
        "workload": label,
        "primary_workload": workload,
        "workload_sweep": " ".join(sweep_names),
        "assigned_gpu": assigned_gpu,
        "datasets": " ".join(datasets),
        "run_id": run_id,
        "expected_run_root": f"results/range_cagra/segment_tree_param_sweep/{run_id}",
        "command": shlex.join(cmd),
    }


def build_plan(args):
    rows = []
    phases = set(args.phases)
    datasets = parse_csv_list(args.datasets)
    sensitivity_datasets = parse_csv_list(args.sensitivity_datasets)
    scalability_datasets = parse_csv_list(args.scalability_datasets)
    large_datasets = parse_csv_list(args.large_datasets)

    if "smoke" in phases:
        rows.append(
            make_command(
                args,
                "smoke",
                "pos_w01",
                datasets[:1],
                "degree",
                min(args.leaf_size, 128),
                "adaptive_d16_i64_min4_i16_it10:1:16:64:4:16:8:10",
                "16:12:1:16;24:16:1:16",
                gpu_id=gpu_id_for(args, len(rows)),
            )
        )

    if "main" in phases:
        workloads = workload_names(args.workload_preset)
        for shard_id, shard in shard_items(datasets, args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "main",
                    workloads[0],
                    shard,
                    "degree",
                    args.leaf_size,
                    args.main_build_configs,
                    args.search_sweep,
                    workload_sweep=workloads,
                    workload_label=args.workload_preset
                    if args.dataset_shards == 1
                    else f"{args.workload_preset}_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    if "arxiv" in phases and large_datasets:
        workloads = workload_names(args.arxiv_workload_preset)
        for shard_id, shard in shard_items(large_datasets, args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "arxiv",
                    workloads[0],
                    shard,
                    "degree",
                    args.arxiv_leaf_size,
                    args.arxiv_build_configs,
                    args.arxiv_search_sweep,
                    workload_sweep=workloads,
                    workload_label=args.arxiv_workload_preset
                    if len(large_datasets) <= 1 or args.dataset_shards == 1
                    else f"{args.arxiv_workload_preset}_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    if "build_ablation" in phases:
        for shard_id, shard in shard_items(datasets, args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "build_ablation",
                    "pos_w50",
                    shard,
                    "degree",
                    args.leaf_size,
                    args.build_ablation_configs,
                    args.search_sweep,
                    workload_label="pos_w50"
                    if args.dataset_shards == 1
                    else f"pos_w50_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    if "leaf_sensitivity" in phases:
        for shard_id, shard in shard_items(sensitivity_datasets, args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "leaf_sensitivity",
                    "pos_w50",
                    shard,
                    "leaf",
                    args.leaf_size,
                    "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
                    args.search_sweep,
                    ["--leaf-sizes", args.leaf_sizes, "--layer-adaptive-degree"],
                    workload_label="pos_w50"
                    if args.dataset_shards == 1
                    else f"pos_w50_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    if "search_policy" in phases:
        for shard_id, shard in shard_items(sensitivity_datasets, args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "search_policy",
                    "pos_w50",
                    shard,
                    "leaf",
                    args.policy_leaf_size,
                    "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
                    args.policy_search_sweep,
                    [
                        "--leaf-sizes",
                        str(args.policy_leaf_size),
                        "--layer-adaptive-degree",
                        "--search-iteration-policy-sweep",
                        "uniform,upper_layers,layer_adaptive",
                        "--upper-layer-search-layers",
                        str(args.upper_layer_count),
                        "--upper-layer-graph-iterations",
                        str(args.upper_layer_iterations),
                        "--adaptive-min-graph-iterations",
                        str(args.policy_adaptive_min_iterations),
                        "--adaptive-max-graph-iterations",
                        str(args.policy_adaptive_max_iterations),
                        "--adaptive-iteration-granularity",
                        str(args.policy_adaptive_granularity),
                    ],
                    workload_label="pos_w50"
                    if args.dataset_shards == 1
                    else f"pos_w50_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    if "scalability" in phases:
        workloads = workload_names("full")
        for shard_id, shard in shard_items(scalability_datasets, args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "scalability",
                    workloads[0],
                    shard,
                    "degree",
                    args.leaf_size,
                    "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
                    args.search_sweep,
                    workload_sweep=workloads,
                    workload_label="full" if args.dataset_shards == 1 else f"full_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    if "diagnostics" in phases:
        for shard_id, shard in shard_items(["gist", "glove-100"], args.dataset_shards):
            rows.append(
                make_command(
                    args,
                    "diagnostics",
                    "pos_w50",
                    shard,
                    "degree",
                    args.leaf_size,
                    args.diagnostic_build_configs,
                    args.diagnostic_search_sweep,
                    workload_label="pos_w50"
                    if args.dataset_shards == 1
                    else f"pos_w50_s{shard_id}",
                    gpu_id=gpu_id_for(args, len(rows)),
                )
            )

    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "phase",
            "workload",
            "primary_workload",
            "workload_sweep",
            "assigned_gpu",
            "datasets",
            "run_id",
            "expected_run_root",
            "command",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_commands(path, rows):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"echo '[{i}/{len(rows)}] gpu={row.get('assigned_gpu', '')} {row['phase']} {row['workload']}'"
        )
        lines.append(row["command"])
        lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def write_gpu_commands(suite_dir, rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.get("assigned_gpu", "0"), []).append(row)
    paths = []
    for gpu_id, gpu_rows in sorted(groups.items()):
        safe_gpu = "".join(ch if ch.isalnum() else "_" for ch in gpu_id)
        path = suite_dir / f"commands_gpu{safe_gpu}.sh"
        write_commands(path, gpu_rows)
        paths.append(path)
    return paths


def write_readme(path, rows, args):
    phase_counts = {}
    for row in rows:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
    multi_gpu = bool(args.gpu_ids) or args.dataset_shards > 1
    lines = [
        "# A100 Range-CAGRA Paper Experiment Suite",
        "",
        f"run_id: `{args.run_id}`",
        f"created_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Commands",
        "",
        "- `suite_plan.csv`: command manifest and expected output roots.",
        "- `commands.sh`: sequential command script.",
        "- `results/range_cagra/collect_a100_paper_results.py --suite-dir <this-dir>`: aggregate completed child runs.",
        "",
    ]
    if multi_gpu:
        lines.extend(
            [
                "- `commands_gpu*.sh`: per-GPU command scripts.",
                "",
                "For multi-GPU runs, generate with `--dataset-shards <num_gpus> --gpu-ids \"0 1 ...\"`, then run each `commands_gpu*.sh` in a separate terminal.",
                "",
            ]
        )
    lines.extend(["## Phase Counts", ""])
    for phase in sorted(phase_counts):
        lines.append(f"- `{phase}`: {phase_counts[phase]} command(s)")
    lines.extend(
        [
            "",
            "## Measurement Contract",
            "",
            "- Build time comes from `build_seconds` in each child run summary.",
            "- GPU memory comes from `phase_gpu_summary.csv`, especially build/search peak memory.",
            "- Search quality and throughput come from `sweep_summary.csv`: `recall_at_k`, `best_qps`, and `avg_qps`.",
            "- Use `filter_violations=0` as a hard validity check before treating a row as usable.",
            "- For frontier plots, select the highest QPS row per dataset/workload at fixed recall thresholds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def execute(rows):
    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['phase']} {row['workload']}", flush=True)
        proc = subprocess.run(row["command"], cwd=ROOT, shell=True)
        if proc.returncode:
            return proc.returncode
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Create or execute the A100 Range-CAGRA paper experiment suite.")
    parser.add_argument("--run-id", default=time.strftime("a100_paper_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["smoke", "main", "arxiv", "build_ablation", "leaf_sensitivity", "search_policy", "scalability", "diagnostics"],
        choices=["smoke", "main", "arxiv", "build_ablation", "leaf_sensitivity", "search_policy", "scalability", "diagnostics"],
    )
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    parser.add_argument("--sensitivity-datasets", default=" ".join(DEFAULT_SENSITIVITY_DATASETS))
    parser.add_argument("--scalability-datasets", default=" ".join(DEFAULT_SCALABILITY_DATASETS))
    parser.add_argument("--large-datasets", default=" ".join(DEFAULT_LARGE_DATASETS))
    parser.add_argument("--workload-preset", choices=["pos50", "core", "full"], default="core")
    parser.add_argument("--arxiv-workload-preset", choices=["pos50", "core", "full"], default="full")
    parser.add_argument("--gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Optional space/comma separated GPU ids for round-robin command assignment.",
    )
    parser.add_argument(
        "--dataset-shards",
        type=int,
        default=1,
        help="Split multi-dataset phases into this many dataset shards for multi-GPU execution.",
    )
    parser.add_argument("--build-dir", default=os.environ.get("RANGE_CAGRA_BUILD_DIR", "cpp/build"))
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument("--arxiv-leaf-size", type=int, default=1000)
    parser.add_argument("--policy-leaf-size", type=int, default=64)
    parser.add_argument("--leaf-sizes", default=DEFAULT_LEAF_SIZES)
    parser.add_argument("--build-algo", choices=["flat_gnnd", "segmented_gnnd"], default="flat_gnnd")
    parser.add_argument("--main-build-configs", default=MAIN_BUILD_CONFIGS)
    parser.add_argument("--build-ablation-configs", default=BUILD_ABLATION_CONFIGS)
    parser.add_argument("--arxiv-build-configs", default=ARXIV_BUILD_CONFIGS)
    parser.add_argument("--diagnostic-build-configs", default=DIAGNOSTIC_BUILD_CONFIGS)
    parser.add_argument("--search-sweep", default=DEFAULT_SEARCH_SWEEP)
    parser.add_argument("--arxiv-search-sweep", default=ARXIV_SEARCH_SWEEP)
    parser.add_argument("--policy-search-sweep", default=POLICY_SEARCH_SWEEP)
    parser.add_argument("--diagnostic-search-sweep", default="64:48:1:32;96:96:2:32;128:128:2:32;256:192:4:64")
    parser.add_argument("--search-repeats", type=int, default=3)
    parser.add_argument("--exact-threads", type=int, default=128)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--scratch-guard-gib", type=float, default=3.5)
    parser.add_argument("--max-est-peak-gib", type=float, default=72.0)
    parser.add_argument("--gpu-fraction", type=float, default=0.9)
    parser.add_argument("--upper-layer-count", type=int, default=6)
    parser.add_argument("--upper-layer-iterations", type=int, default=32)
    parser.add_argument("--policy-adaptive-min-iterations", type=int, default=12)
    parser.add_argument("--policy-adaptive-max-iterations", type=int, default=32)
    parser.add_argument("--policy-adaptive-granularity", type=int, default=4)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--inner-dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not SWEEP_RUNNER.exists():
        raise SystemExit(f"missing runner: {SWEEP_RUNNER}")
    suite_dir = ROOT / "results" / "range_cagra" / "a100_paper_suite" / args.run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    rows = build_plan(args)
    write_csv(suite_dir / "suite_plan.csv", rows)
    write_commands(suite_dir / "commands.sh", rows)
    gpu_scripts = write_gpu_commands(suite_dir, rows) if args.gpu_ids or args.dataset_shards > 1 else []
    write_readme(suite_dir / "README.md", rows, args)
    (suite_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    print(f"suite_dir={suite_dir.relative_to(ROOT)}")
    print((suite_dir / "suite_plan.csv").relative_to(ROOT))
    print((suite_dir / "commands.sh").relative_to(ROOT))
    for path in gpu_scripts:
        print(path.relative_to(ROOT))
    if args.execute:
        return execute(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
