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
SUITE_ROOT = ROOT / "results" / "range_cagra" / "paper_full_suite"

MAIN_DATASETS = [
    "audio",
    "arxiv-for-fanns-large",
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

ABLATION_DATASETS = [
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

FULL_WIDTHS = ["01", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50"]

FIXED_DEGREE_CONFIG = "uniform_d32_i96_it20:0:32:96:0:0:8:20"

DEGREE_ABLATION_CONFIGS = ";".join(
    [
        FIXED_DEGREE_CONFIG,
        "adaptive_d24_i64_min4_i12_it20:1:24:64:4:12:8:20",
        "adaptive_d24_i72_min6_i18_it20:1:24:72:6:18:8:20",
        "adaptive_d32_i96_min4_i12_it20:1:32:96:4:12:8:20",
        "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
        "adaptive_d32_i96_min16_i48_it20:1:32:96:16:48:8:20",
    ]
)

FINAL_DEGREE_BY_DATASET = {
    "audio": "adaptive_d24_i72_min6_i18_it20:1:24:72:6:18:8:20",
    "deep": "adaptive_d32_i96_min4_i12_it20:1:32:96:4:12:8:20",
    "enron": "adaptive_d32_i96_min4_i12_it20:1:32:96:4:12:8:20",
    "gist": FIXED_DEGREE_CONFIG,
    "glove-100": FIXED_DEGREE_CONFIG,
    "msong": "adaptive_d24_i64_min4_i12_it20:1:24:64:4:12:8:20",
    "sift": "adaptive_d32_i96_min4_i12_it20:1:32:96:4:12:8:20",
    "text2image": "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
    "wit": "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
    "yt8mAudio": "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
    "arxiv-for-fanns-large": FIXED_DEGREE_CONFIG,
}

FINAL_LEAF_BY_DATASET = {
    "audio": 64,
    "deep": 128,
    "enron": 32,
    "gist": 64,
    "glove-100": 64,
    "msong": 48,
    "sift": 96,
    "text2image": 64,
    "wit": 64,
    "yt8mAudio": 64,
    "arxiv-for-fanns-large": 1000,
}

FINAL_SEARCH_POLICY_BY_DATASET = {
    "audio": "layer_adaptive:6:32:2",
    "deep": "layer_adaptive:6:32:2",
    "enron": "layer_adaptive:6:32:2",
    "msong": "layer_adaptive:6:32:2",
    "sift": "layer_adaptive:6:32:2",
    "yt8mAudio": "layer_adaptive:6:32:2",
    "gist": "layer_adaptive:8:32:4",
    "glove-100": "layer_adaptive:8:32:4",
    "text2image": "layer_adaptive:8:32:4",
    "wit": "layer_adaptive:8:32:4",
    "arxiv-for-fanns-large": "layer_adaptive:8:32:4",
}

FAST90_SEARCH_SWEEP = (
    "10:4:1:8;"
    "12:6:1:8;"
    "16:8:1:12;"
    "16:10:1:12;"
    "24:12:1:16;"
    "24:16:1:16;"
    "32:16:1:16;"
    "32:24:1:16;"
    "48:24:1:24;"
    "64:32:1:24;"
    "96:48:2:32"
)

ROBUST_SEARCH_SWEEP = (
    FAST90_SEARCH_SWEEP
    + ";96:64:2:32;128:96:2:32;192:128:2:64"
)

SEARCH_POLICY_SWEEP = (
    "uniform;"
    "layer_adaptive:12:32:4;"
    "layer_adaptive:10:32:2;"
    "layer_adaptive:8:32:4;"
    "layer_adaptive:6:32:2;"
    "layer_adaptive:4:32:2"
)

LEAF_SIZES = "8 12 16 24 32 48 64 96 128 256 512 1000"


def parse_list(text):
    return [item for item in text.replace(",", " ").split() if item]


def workloads(preset):
    if preset == "pos50":
        return ["pos_w50"]
    widths = ["01", "05", "10", "20", "50"] if preset == "core" else FULL_WIDTHS
    if preset not in ("core", "full"):
        raise ValueError(f"unknown workload preset: {preset}")
    return [f"{kind}_w{width}" for kind in ("pos", "ind", "neg") for width in widths]


def detect_gpu_ids():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["0"]
    ids = [line.strip() for line in out.splitlines() if line.strip()]
    return ids or ["0"]


def shard_items(items, shard_count):
    shard_count = max(1, int(shard_count))
    shards = [[] for _ in range(shard_count)]
    for i, item in enumerate(items):
        shards[i % shard_count].append(item)
    return [(i, shard) for i, shard in enumerate(shards) if shard]


def shell_join(items):
    return " ".join(shlex.quote(str(item)) for item in items)


def ld_library_prefix(args):
    script_root = pathlib.Path(args.script_root)
    remote_mode = str(script_root) != str(ROOT)
    entries = [
        script_root / args.build_dir,
        script_root / args.build_dir / "_deps" / "rmm-build",
    ]

    def add_conda_paths(prefix):
        if not prefix:
            return
        base = pathlib.Path(prefix)
        entries.extend([base / "lib", base / "targets" / "x86_64-linux" / "lib"])

    add_conda_paths(os.environ.get("CONDA_PREFIX"))
    add_conda_paths("/wjy/conda-envs/cuvs-build-129")
    add_conda_paths("/home/wjy/miniconda3/envs/faiss-cuvs")
    if remote_mode:
        return ":".join(str(path) for path in entries)
    return ":".join(str(path) for path in entries if path.exists())


def fixed_leaf_config_args():
    return [
        "--config-label",
        "uniform_d32_i96_it20",
        "--graph-degree",
        "32",
        "--intermediate-graph-degree",
        "96",
        "--min-graph-degree",
        "0",
        "--min-intermediate-graph-degree",
        "0",
        "--degree-granularity",
        "8",
        "--nn-descent-iters",
        "20",
    ]


def policy_args(policy_spec):
    fields = policy_spec.split(":")
    if fields[0] != "layer_adaptive" or len(fields) != 4:
        raise ValueError(f"unsupported final search policy: {policy_spec}")
    return [
        "--search-iteration-policy",
        "layer_adaptive",
        "--adaptive-min-graph-iterations",
        fields[1],
        "--adaptive-max-graph-iterations",
        fields[2],
        "--adaptive-iteration-granularity",
        fields[3],
    ]


def make_row(args, phase, shard_id, datasets, gpu_id, **kwargs):
    suffix = kwargs.pop("run_suffix", f"s{shard_id}")
    row = {
        "phase": phase,
        "shard_id": shard_id,
        "assigned_gpu": gpu_id,
        "datasets": datasets,
        "workloads": kwargs.pop("workloads"),
        "sweep": kwargs.pop("sweep"),
        "leaf_size": kwargs.pop("leaf_size"),
        "degree_configs": kwargs.pop("degree_configs"),
        "search_sweep": kwargs.pop("search_sweep"),
        "extra_args": kwargs.pop("extra_args", []),
        "max_queries": kwargs.pop("max_queries", args.max_queries),
        "search_repeats": kwargs.pop("search_repeats", args.search_repeats),
        "notes": kwargs.pop("notes", ""),
    }
    row["run_id"] = f"{args.run_id}_{phase}_{suffix}"
    row["expected_run_root"] = f"results/range_cagra/segment_tree_param_sweep/{row['run_id']}"
    return row


def common_cmd(args, row):
    cmd = [
        "python3",
        str(SWEEP_RUNNER.relative_to(ROOT)),
        "--run-id",
        row["run_id"],
        "--sweep",
        row["sweep"],
        "--datasets",
        *row["datasets"],
        "--workload",
        row["workloads"][0],
        "--workload-sweep",
        " ".join(row["workloads"]),
        "--gpu-id",
        row["assigned_gpu"],
        "--build-dir",
        args.build_dir,
        "--data-root",
        args.data_root,
        "--query-root",
        args.query_root,
        "--max-queries",
        str(row["max_queries"]),
        "--topk",
        str(args.topk),
        "--leaf-size",
        str(row["leaf_size"]),
        "--build-algo",
        args.build_algo,
        "--degree-configs",
        row["degree_configs"],
        "--search-sweep",
        row["search_sweep"],
        "--search-schedule-sweep",
        args.search_schedule,
        "--search-repeats",
        str(row["search_repeats"]),
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
        "--skip-build",
    ]
    if args.include_risky:
        cmd.append("--include-risky")
    if args.inner_dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    cmd.extend(row["extra_args"])
    return shell_join(cmd)


def build_plan(args):
    main_datasets = parse_list(args.datasets)
    ablation_datasets = parse_list(args.ablation_datasets)
    gpu_ids = parse_list(args.gpu_ids) or detect_gpu_ids()
    shard_count = args.dataset_shards if args.dataset_shards > 0 else len(gpu_ids)
    full_workloads = workloads(args.workload_preset)
    rows = []

    if "smoke" in args.phases:
        rows.append(
            make_row(
                args,
                "smoke",
                0,
                [args.smoke_dataset],
                gpu_ids[0],
                run_suffix=args.smoke_dataset,
                workloads=[args.smoke_workload],
                sweep="degree",
                leaf_size=args.smoke_leaf_size,
                degree_configs=FIXED_DEGREE_CONFIG,
                search_sweep="12:6:1:8;16:8:1:12",
                extra_args=[
                    "--search-iteration-policy-sweep",
                    "uniform;layer_adaptive:8:32:4;layer_adaptive:6:32:2",
                ],
                max_queries=args.smoke_max_queries,
                search_repeats=1,
                notes="workspace-no-presync smoke plus fixed-degree search-adaptive check",
            )
        )

    if "layer_degree" in args.phases:
        for shard_id, shard in shard_items(ablation_datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "layer_degree",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=full_workloads,
                    sweep="degree",
                    leaf_size=args.ablation_leaf_size,
                    degree_configs=args.degree_ablation_configs,
                    search_sweep=args.ablation_search_sweep,
                    extra_args=["--search-iteration-policy", "uniform"],
                    notes="fixed leaf and fixed uniform search; only degree allocation changes",
                )
            )

    if "layer_search" in args.phases:
        for shard_id, shard in shard_items(ablation_datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "layer_search",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=full_workloads,
                    sweep="degree",
                    leaf_size=args.ablation_leaf_size,
                    degree_configs=FIXED_DEGREE_CONFIG,
                    search_sweep=args.search_ablation_sweep,
                    extra_args=["--search-iteration-policy-sweep", args.search_policy_sweep],
                    notes="fixed leaf and fixed degree; only search-effort allocation changes",
                )
            )

    if "leaf_size" in args.phases:
        for shard_id, shard in shard_items(ablation_datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "leaf_size",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=full_workloads,
                    sweep="leaf",
                    leaf_size=args.ablation_leaf_size,
                    degree_configs=FIXED_DEGREE_CONFIG,
                    search_sweep=args.leaf_ablation_sweep,
                    extra_args=[
                        "--leaf-sizes",
                        args.leaf_sizes,
                        "--search-iteration-policy",
                        "uniform",
                        *fixed_leaf_config_args(),
                    ],
                    notes="fixed degree and fixed uniform search; only leaf size changes",
                )
            )

    if "main_algo" in args.phases:
        for i, dataset in enumerate(main_datasets):
            if dataset not in FINAL_DEGREE_BY_DATASET:
                raise SystemExit(f"missing final degree config for dataset: {dataset}")
            search_sweep = args.arxiv_search_sweep if dataset == "arxiv-for-fanns-large" else args.final_search_sweep
            rows.append(
                make_row(
                    args,
                    "main_algo",
                    i,
                    [dataset],
                    gpu_ids[i % len(gpu_ids)],
                    run_suffix=dataset.replace("-", "_"),
                    workloads=full_workloads,
                    sweep="degree",
                    leaf_size=FINAL_LEAF_BY_DATASET[dataset],
                    degree_configs=FINAL_DEGREE_BY_DATASET[dataset],
                    search_sweep=search_sweep,
                    extra_args=policy_args(FINAL_SEARCH_POLICY_BY_DATASET[dataset]),
                    notes="per-dataset Fast90 candidate: local winners where available, conservative A100 fallback otherwise",
                )
            )

    for row in rows:
        row["command"] = common_cmd(args, row)
    return rows, gpu_ids


PLAN_FIELDS = [
    "phase",
    "shard_id",
    "assigned_gpu",
    "datasets",
    "workload",
    "primary_workload",
    "workload_sweep",
    "sweep",
    "leaf_size",
    "degree_configs",
    "search_sweep",
    "max_queries",
    "search_repeats",
    "run_id",
    "expected_run_root",
    "notes",
    "command",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            serial["datasets"] = " ".join(row["datasets"])
            serial["workload"] = row["workloads"][0]
            serial["primary_workload"] = row["workloads"][0]
            serial["workload_sweep"] = " ".join(row["workloads"])
            serial.pop("workloads", None)
            serial.pop("extra_args", None)
            writer.writerow({key: serial.get(key, "") for key in PLAN_FIELDS})


def script_header(args):
    ld_prefix = ld_library_prefix(args)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(args.script_root))}",
        "export RANGE_CAGRA_SEGMENT_WORKSPACE_MODE=device_no_presync",
    ]
    if ld_prefix:
        lines.append(f"export LD_LIBRARY_PATH={shlex.quote(ld_prefix)}:${{LD_LIBRARY_PATH:-}}")
    lines.append("")
    return lines


def write_build_once(path, args):
    lines = script_header(args)
    lines.append(
        f"cmake --build {shlex.quote(args.build_dir)} --target NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST -j{args.build_jobs}"
    )
    lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def write_command_script(path, args, rows, label, suite_dir=None, build_first=False):
    lines = script_header(args)
    if build_first:
        if suite_dir is None:
            raise ValueError("suite_dir is required when build_first=True")
        lines.append(f"bash {shlex.quote(str((suite_dir / 'build_once.sh').relative_to(ROOT)))}")
        lines.append("")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"echo '[{label} {i}/{len(rows)}] gpu={row['assigned_gpu']} phase={row['phase']} shard={row['shard_id']}'"
        )
        lines.append(row["command"])
        lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def write_launcher(path, args, suite_dir, gpu_scripts, build_first=True):
    lines = script_header(args)
    if build_first:
        lines.append(f"bash {shlex.quote(str((suite_dir / 'build_once.sh').relative_to(ROOT)))}")
    lines.extend(
        [
            "mkdir -p " + shlex.quote(str((suite_dir / "logs").relative_to(ROOT))),
            "pids=()",
        ]
    )
    for script in gpu_scripts:
        log = suite_dir / "logs" / f"{script.stem}.log"
        lines.append(
            f"bash {shlex.quote(str(script.relative_to(ROOT)))} >{shlex.quote(str(log.relative_to(ROOT)))} 2>&1 &"
        )
        lines.append('pids+=("$!")')
    lines.extend(
        [
            "status=0",
            'for pid in "${pids[@]}"; do',
            '  if ! wait "$pid"; then status=1; fi',
            "done",
            'exit "$status"',
            "",
        ]
    )
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def write_readme(path, args, rows, gpu_scripts):
    phase_counts = {}
    for row in rows:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
    lines = [
        "# A100 Range-CAGRA Fast90 Paper Suite",
        "",
        f"run_id: `{args.run_id}`",
        f"created_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Contract",
        "",
        "- Objective: maximize QPS subject to `Recall@10 >= 0.90`.",
        "- Execution schedule is fixed to `exact_then_graph`.",
        "- Search uses `RANGE_CAGRA_SEGMENT_WORKSPACE_MODE=device_no_presync`, so the benchmark path stays on the GPU-side workspace launch path.",
        "- Main algorithm uses per-dataset Fast90 candidate configs, then sweeps search parameters.",
        "- Ablations are orthogonal: dynamic degree, dynamic search effort, and leaf size are each compared against a fixed uniform `d32/i96/it20`, `leaf=64`, uniform-search baseline unless the axis itself is being swept.",
        "- Treat any row with `filter_violations != 0` as invalid.",
        "",
        "## Files",
        "",
        "- `suite_plan.csv`: command manifest and expected child output roots.",
        "- `commands.sh`: sequential full execution.",
        "- `commands_gpu*.sh`: per-GPU shards.",
        "- `commands_2gpu.sh`: parallel per-GPU launcher.",
        "- `build_once.sh`: builds `NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST` once.",
        "",
        "## Run",
        "",
        "```bash",
        f"bash {path.parent.relative_to(ROOT)}/commands_2gpu.sh",
        "python3 results/range_cagra/collect_a100_paper_results.py --suite-dir "
        f"{path.parent.relative_to(ROOT)} --thresholds '0.90 0.95 0.98 0.99'",
        "```",
        "",
        "## Phase Counts",
        "",
    ]
    for phase in sorted(phase_counts):
        lines.append(f"- `{phase}`: {phase_counts[phase]} command(s)")
    lines.extend(
        [
            "",
            "## Generated GPU Scripts",
            "",
        ]
    )
    for script in gpu_scripts:
        lines.append(f"- `{script.relative_to(ROOT)}`")
    lines.append("")
    path.write_text("\n".join(lines))


def execute_rows(args, rows):
    env = os.environ.copy()
    env["RANGE_CAGRA_SEGMENT_WORKSPACE_MODE"] = "device_no_presync"
    ld_prefix = ld_library_prefix(args)
    if ld_prefix:
        env["LD_LIBRARY_PATH"] = f"{ld_prefix}:{env.get('LD_LIBRARY_PATH', '')}"
    if args.build_once:
        subprocess.run(
            [
                "cmake",
                "--build",
                args.build_dir,
                "--target",
                "NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST",
                f"-j{args.build_jobs}",
            ],
            cwd=ROOT,
            check=True,
            env=env,
        )
    for i, row in enumerate(rows, start=1):
        print(f"[execute {i}/{len(rows)}] {row['phase']} gpu={row['assigned_gpu']}", flush=True)
        proc = subprocess.run(row["command"], cwd=ROOT, shell=True, env=env)
        if proc.returncode:
            return proc.returncode
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the A100 Fast90 Range-CAGRA paper suite.")
    parser.add_argument("--run-id", default=time.strftime("a100_fast90_full_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["smoke", "layer_degree", "layer_search", "leaf_size", "main_algo"],
        choices=["smoke", "layer_degree", "layer_search", "leaf_size", "main_algo"],
    )
    parser.add_argument("--datasets", default=" ".join(MAIN_DATASETS))
    parser.add_argument("--ablation-datasets", default=" ".join(ABLATION_DATASETS))
    parser.add_argument("--workload-preset", choices=["pos50", "core", "full"], default="full")
    parser.add_argument("--gpu-ids", default=" ".join(detect_gpu_ids()[:2]))
    parser.add_argument("--dataset-shards", type=int, default=0)
    parser.add_argument("--script-root", default=str(ROOT))
    parser.add_argument("--build-dir", default=os.environ.get("RANGE_CAGRA_BUILD_DIR", "cpp/build"))
    parser.add_argument("--data-root", default="")
    parser.add_argument("--query-root", default="")
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--smoke-max-queries", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--build-algo", choices=["flat_gnnd", "segmented_gnnd"], default="flat_gnnd")
    parser.add_argument("--ablation-leaf-size", type=int, default=64)
    parser.add_argument("--smoke-leaf-size", type=int, default=64)
    parser.add_argument("--smoke-dataset", default="audio")
    parser.add_argument("--smoke-workload", default="pos_w01")
    parser.add_argument("--degree-ablation-configs", default=DEGREE_ABLATION_CONFIGS)
    parser.add_argument("--leaf-sizes", default=LEAF_SIZES)
    parser.add_argument("--final-search-sweep", default=FAST90_SEARCH_SWEEP)
    parser.add_argument("--arxiv-search-sweep", default=ROBUST_SEARCH_SWEEP)
    parser.add_argument("--ablation-search-sweep", default=FAST90_SEARCH_SWEEP)
    parser.add_argument("--search-ablation-sweep", default=FAST90_SEARCH_SWEEP)
    parser.add_argument("--leaf-ablation-sweep", default=FAST90_SEARCH_SWEEP)
    parser.add_argument("--search-policy-sweep", default=SEARCH_POLICY_SWEEP)
    parser.add_argument("--search-schedule", default="exact_then_graph")
    parser.add_argument("--search-repeats", type=int, default=3)
    parser.add_argument("--exact-threads", type=int, default=128)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--scratch-guard-gib", type=float, default=3.5)
    parser.add_argument("--max-est-peak-gib", type=float, default=72.0)
    parser.add_argument("--gpu-fraction", type=float, default=0.9)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--inner-dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-build-once", dest="build_once", action="store_false")
    parser.set_defaults(build_once=True)
    parser.add_argument("--build-jobs", type=int, default=2)
    args = parser.parse_args()
    script_root = pathlib.Path(args.script_root)
    if not args.data_root:
        args.data_root = str(script_root / "data")
    if not args.query_root:
        args.query_root = str(script_root / "generated_queries" / "order_range_raw_attr")
    return args


def main():
    args = parse_args()
    if not SWEEP_RUNNER.exists():
        raise SystemExit(f"missing sweep runner: {SWEEP_RUNNER}")
    rows, gpu_ids = build_plan(args)
    suite_dir = SUITE_ROOT / args.run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_csv(suite_dir / "suite_plan.csv", rows)
    write_build_once(suite_dir / "build_once.sh", args)
    write_command_script(
        suite_dir / "commands.sh", args, rows, "all", suite_dir=suite_dir, build_first=args.build_once
    )

    gpu_scripts = []
    for gpu_id in gpu_ids:
        gpu_rows = [row for row in rows if row["assigned_gpu"] == gpu_id]
        if not gpu_rows:
            continue
        safe_gpu = "".join(ch if ch.isalnum() else "_" for ch in gpu_id)
        script = suite_dir / f"commands_gpu{safe_gpu}.sh"
        write_command_script(script, args, gpu_rows, f"gpu{gpu_id}")
        gpu_scripts.append(script)
    write_launcher(suite_dir / "commands_2gpu.sh", args, suite_dir, gpu_scripts, build_first=args.build_once)
    write_readme(suite_dir / "README.md", args, rows, gpu_scripts)
    (suite_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

    print(f"suite_dir={suite_dir.relative_to(ROOT)}")
    print((suite_dir / "suite_plan.csv").relative_to(ROOT))
    print((suite_dir / "commands.sh").relative_to(ROOT))
    for script in gpu_scripts:
        print(script.relative_to(ROOT))
    print((suite_dir / "commands_2gpu.sh").relative_to(ROOT))
    if args.execute:
        return execute_rows(args, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
