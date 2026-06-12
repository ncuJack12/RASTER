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

REGULAR_DATASETS = [
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

FULL_WIDTHS = ["01", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50"]

LAYER_DEGREE_CONFIGS = ";".join(
    [
        "uniform_d32_i96_it20:0:32:96:0:0:8:20",
        "adaptive_d16_i48_min4_i12_it20:1:16:48:4:12:8:20",
        "adaptive_d16_i64_min4_i12_it20:1:16:64:4:12:8:20",
        "adaptive_d24_i64_min4_i12_it20:1:24:64:4:12:8:20",
        "adaptive_d24_i72_min6_i18_it20:1:24:72:6:18:8:20",
        "adaptive_d32_i96_min16_i48_it20:1:32:96:16:48:8:20",
        "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20",
        "adaptive_d32_i96_min4_i12_it20:1:32:96:4:12:8:20",
    ]
)

BASE_ADAPTIVE_CONFIG = "adaptive_d24_i64_min4_i12_it20:1:24:64:4:12:8:20"
BASE_FIXED_CONFIG = "uniform_d32_i96_it20:0:32:96:0:0:8:20"
FINAL_ALGORITHM_CONFIG = "range_cagra_fast90:1:24:64:4:12:8:20"

FULL_SEARCH_SWEEP = (
    "12:8:1:12;"
    "16:8:1:12;"
    "16:10:1:12;"
    "16:12:1:16;"
    "24:12:1:16;"
    "24:16:1:16;"
    "32:16:1:16;"
    "32:24:1:16;"
    "48:24:1:24;"
    "64:32:1:24"
)
LAYER_SEARCH_SWEEP = FULL_SEARCH_SWEEP
LAYER_SEARCH_POLICY_SWEEP = (
    "uniform;"
    "layer_adaptive:12:32:4;"
    "layer_adaptive:10:32:2;"
    "layer_adaptive:8:32:4;"
    "layer_adaptive:6:32:2"
)
LEAF_SIZES = "16 32 64 96 128 256 512 1000"


def parse_list(text):
    return [item for item in text.replace(",", " ").split() if item]


def full_workloads():
    return [f"{kind}_w{width}" for kind in ("pos", "ind", "neg") for width in FULL_WIDTHS]


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
    entries = [
        ROOT / args.build_dir,
        ROOT / args.build_dir / "_deps" / "rmm-build",
    ]

    def add_conda_paths(prefix):
        if not prefix:
            return
        base = pathlib.Path(prefix)
        entries.extend([base / "lib", base / "targets" / "x86_64-linux" / "lib"])

    add_conda_paths(os.environ.get("CONDA_PREFIX"))
    for prefix in os.environ.get("RASTER_EXTRA_CONDA_PREFIXES", "").split(os.pathsep):
        add_conda_paths(prefix)

    return ":".join(str(path) for path in entries if path.exists())


def leaf_base_config_args():
    return [
        "--config-label",
        "adaptive_d24_i64_min4_i12",
        "--graph-degree",
        "24",
        "--intermediate-graph-degree",
        "64",
        "--layer-adaptive-degree",
        "--min-graph-degree",
        "4",
        "--min-intermediate-graph-degree",
        "12",
        "--degree-granularity",
        "8",
        "--nn-descent-iters",
        "20",
    ]


def final_search_policy_args(args):
    policy = args.final_search_iteration_policy
    extra = ["--search-iteration-policy", policy]
    if policy == "upper_layers":
        extra.extend(
            [
                "--upper-layer-search-layers",
                str(args.final_upper_layer_count),
                "--upper-layer-graph-iterations",
                str(args.final_upper_layer_iterations),
            ]
        )
    elif policy == "layer_adaptive":
        extra.extend(
            [
                "--adaptive-min-graph-iterations",
                str(args.final_adaptive_min_iterations),
                "--adaptive-max-graph-iterations",
                str(args.final_adaptive_max_iterations),
                "--adaptive-iteration-granularity",
                str(args.final_adaptive_granularity),
            ]
        )
    else:
        raise ValueError(f"unsupported final search iteration policy: {policy}")
    return extra


def common_cmd(args, row):
    max_queries = row.get("max_queries", args.max_queries)
    cmd = [
        "python3",
        str(SWEEP_RUNNER.relative_to(ROOT)),
        "--run-id",
        row["child_run_id"],
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
        str(max_queries),
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
        str(row.get("search_repeats", args.search_repeats)),
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
    cmd.extend(row.get("extra_args", []))
    return shell_join(cmd)


def make_row(args, phase, shard_id, datasets, gpu_id, **kwargs):
    run_suffix = kwargs.pop("run_suffix", f"s{shard_id}")
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
    }
    row["child_run_id"] = f"{args.run_id}_{phase}_{run_suffix}"
    row["expected_run_root"] = f"results/range_cagra/segment_tree_param_sweep/{row['child_run_id']}"
    return row


def build_plan(args):
    datasets = parse_list(args.datasets)
    gpu_ids = parse_list(args.gpu_ids) or detect_gpu_ids()
    shard_count = args.dataset_shards if args.dataset_shards > 0 else len(gpu_ids)
    workloads = full_workloads()
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
                degree_configs="adaptive_d16_i64_min4_i16_it10:1:16:64:4:16:8:10",
                search_sweep="16:12:1:16;24:16:1:16",
                extra_args=[
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
                max_queries=args.smoke_max_queries,
                search_repeats=1,
            )
        )

    if "main_algo" in args.phases:
        for shard_id, shard in shard_items(datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "main_algo",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=workloads,
                    sweep="degree",
                    leaf_size=args.final_leaf_size,
                    degree_configs=args.final_degree_config,
                    search_sweep=args.final_search_sweep,
                    extra_args=final_search_policy_args(args),
                )
            )

    if "layer_degree" in args.phases:
        for shard_id, shard in shard_items(datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "layer_degree",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=workloads,
                    sweep="degree",
                    leaf_size=args.degree_leaf_size,
                    degree_configs=args.layer_degree_configs,
                    search_sweep=args.degree_search_sweep,
                )
            )

    if "layer_search" in args.phases:
        for shard_id, shard in shard_items(datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "layer_search",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=workloads,
                    sweep="degree",
                    leaf_size=args.search_leaf_size,
                    degree_configs=args.search_degree_config,
                    search_sweep=args.layer_search_sweep,
                    extra_args=[
                        "--search-iteration-policy-sweep",
                        args.layer_search_policy_sweep,
                    ],
                )
            )

    if "leaf_size" in args.phases:
        for shard_id, shard in shard_items(datasets, shard_count):
            rows.append(
                make_row(
                    args,
                    "leaf_size",
                    shard_id,
                    shard,
                    gpu_ids[shard_id % len(gpu_ids)],
                    workloads=workloads,
                    sweep="leaf",
                    leaf_size=args.leaf_size,
                    degree_configs=args.base_adaptive_config,
                    search_sweep=args.leaf_search_sweep,
                    extra_args=["--leaf-sizes", args.leaf_sizes, *leaf_base_config_args()],
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
    "workloads",
    "sweep",
    "leaf_size",
    "degree_configs",
    "search_sweep",
    "max_queries",
    "search_repeats",
    "child_run_id",
    "expected_run_root",
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
            serial["workloads"] = " ".join(row["workloads"])
            serial.pop("extra_args", None)
            writer.writerow({key: serial.get(key, "") for key in PLAN_FIELDS})


def write_build_once(path, args):
    ld_prefix = ld_library_prefix(args)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
    ]
    if ld_prefix:
        lines.append(f"export LD_LIBRARY_PATH={shlex.quote(ld_prefix)}:${{LD_LIBRARY_PATH:-}}")
    lines.extend(
        [
            f"cmake --build {shlex.quote(args.build_dir)} --target NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST -j{args.build_jobs}",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def script_header(args):
    ld_prefix = ld_library_prefix(args)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
    ]
    if ld_prefix:
        lines.append(f"export LD_LIBRARY_PATH={shlex.quote(ld_prefix)}:${{LD_LIBRARY_PATH:-}}")
    lines.append("")
    return lines


def write_command_script(path, rows, label, suite_dir=None, build_first=False):
    lines = script_header(parse_args.cached_args)
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


def write_launcher(path, suite_dir, gpu_scripts, build_first=True):
    lines = script_header(parse_args.cached_args)
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
        lines.append("pids+=(\"$!\")")
    lines.extend(
        [
            "status=0",
            "for pid in \"${pids[@]}\"; do",
            "  if ! wait \"$pid\"; then status=1; fi",
            "done",
            "exit \"$status\"",
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
        "# Range-CAGRA Full Paper Experiment Suite",
        "",
        f"run_id: `{args.run_id}`",
        f"created_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Scope",
        "",
        "- Full dataset set: `audio arxiv-for-fanns-large deep enron gist glove-100 msong sift text2image wit yt8mAudio` unless overridden.",
        "- Full workload set: `pos/ind/neg` x widths `01,05,10,15,20,25,30,35,40,45,50`.",
        "- Main algorithm: `range_cagra_fast90`, which maximizes QPS subject to Recall@10 >= 0.90.",
        "- Fast90 default: layer-adaptive degree `d24/i64 -> min4/i12`, `leaf_size=64`, low-budget search sweep, and layer-adaptive search ratio `8/32/4`.",
        "- Three paper axes: adaptive layer-degree allocation, adaptive layer-search effort, and leaf-size parameter search.",
        "- The `layer_search` ablation uses fixed uniform degree by default, so it isolates search-effort adaptivity instead of measuring only stacked gain on an adaptive-degree index.",
        "- `exact_then_graph` is fixed as the execution schedule; schedule is not a main paper ablation.",
        "",
        "## Files",
        "",
        "- `suite_plan.csv`: one row per generated child sweep command.",
        "- `commands.sh`: sequential execution of all child sweeps.",
        "- `commands_gpu*.sh`: per-GPU shards for parallel execution.",
        "- `commands_2gpu.sh`: runs all per-GPU scripts concurrently; it builds first only when `build_once` is true.",
        "- `build_once.sh`: builds `NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST` once before full execution.",
        f"- `build_once` in generated launchers: `{args.build_once}`.",
        "",
        "## Recommended Execution",
        "",
        "```bash",
        f"bash {path.parent.relative_to(ROOT)}/commands_2gpu.sh",
        "```",
        "",
        "For manual two-terminal execution:",
        "",
        "```bash",
    ]
    if args.build_once:
        lines.append(f"bash {path.parent.relative_to(ROOT)}/build_once.sh")
    for script in gpu_scripts:
        lines.append(f"bash {script.relative_to(ROOT)}")
    lines.extend(["```", "", "## Phase Counts", ""])
    for phase in sorted(phase_counts):
        lines.append(f"- `{phase}`: {phase_counts[phase]} command(s)")
    lines.extend(
        [
            "",
            "## Validity Checks",
            "",
            "- Treat any row with `filter_violations != 0` as invalid.",
            "- Use `aggregate_sweep.csv` from each child run for Recall@10 >= 0.90 QPS frontiers.",
            "- Use `phase_gpu_summary.csv` for build/search peak memory.",
            "- Memory guard may skip `text2image` or `wit` on 11GB GPUs; skipped rows are recorded in each child `status.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def execute_rows(args, rows):
    env = os.environ.copy()
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
    parser = argparse.ArgumentParser(description="Generate the full Range-CAGRA paper experiment suite.")
    parser.add_argument("--run-id", default=time.strftime("paper_full_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["smoke", "main_algo", "layer_degree", "layer_search", "leaf_size"],
        choices=["smoke", "main_algo", "layer_degree", "layer_search", "leaf_size"],
    )
    parser.add_argument("--datasets", default=" ".join(REGULAR_DATASETS))
    parser.add_argument("--gpu-ids", default=" ".join(detect_gpu_ids()[:2]))
    parser.add_argument("--dataset-shards", type=int, default=0)
    parser.add_argument("--build-dir", default=os.environ.get("RANGE_CAGRA_BUILD_DIR", "cpp/build"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--query-root", default=str(ROOT / "generated_queries" / "order_range_raw_attr"))
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--smoke-max-queries", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--build-algo", choices=["flat_gnnd", "segmented_gnnd"], default="flat_gnnd")
    parser.add_argument("--degree-leaf-size", type=int, default=64)
    parser.add_argument("--search-leaf-size", type=int, default=64)
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument("--smoke-leaf-size", type=int, default=128)
    parser.add_argument("--smoke-dataset", default="audio")
    parser.add_argument("--smoke-workload", default="pos_w01")
    parser.add_argument("--leaf-sizes", default=LEAF_SIZES)
    parser.add_argument("--layer-degree-configs", default=LAYER_DEGREE_CONFIGS)
    parser.add_argument("--base-adaptive-config", default=BASE_ADAPTIVE_CONFIG)
    parser.add_argument("--search-degree-config", default=BASE_FIXED_CONFIG)
    parser.add_argument("--final-degree-config", default=FINAL_ALGORITHM_CONFIG)
    parser.add_argument("--final-leaf-size", type=int, default=64)
    parser.add_argument("--final-search-sweep", default=FULL_SEARCH_SWEEP)
    parser.add_argument(
        "--final-search-iteration-policy",
        choices=["upper_layers", "layer_adaptive"],
        default="layer_adaptive",
    )
    parser.add_argument("--final-upper-layer-count", type=int, default=6)
    parser.add_argument("--final-upper-layer-iterations", type=int, default=32)
    parser.add_argument("--final-adaptive-min-iterations", type=int, default=8)
    parser.add_argument("--final-adaptive-max-iterations", type=int, default=32)
    parser.add_argument("--final-adaptive-granularity", type=int, default=4)
    parser.add_argument("--degree-search-sweep", default=FULL_SEARCH_SWEEP)
    parser.add_argument("--layer-search-sweep", default=LAYER_SEARCH_SWEEP)
    parser.add_argument("--leaf-search-sweep", default=FULL_SEARCH_SWEEP)
    parser.add_argument("--search-schedule", default="exact_then_graph")
    parser.add_argument("--search-repeats", type=int, default=3)
    parser.add_argument("--exact-threads", type=int, default=128)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--scratch-guard-gib", type=float, default=3.5)
    parser.add_argument("--max-est-peak-gib", type=float, default=10.0)
    parser.add_argument("--gpu-fraction", type=float, default=0.9)
    parser.add_argument("--upper-layer-count", type=int, default=6)
    parser.add_argument("--upper-layer-iterations", type=int, default=32)
    parser.add_argument("--policy-adaptive-min-iterations", type=int, default=12)
    parser.add_argument("--policy-adaptive-max-iterations", type=int, default=32)
    parser.add_argument("--policy-adaptive-granularity", type=int, default=4)
    parser.add_argument("--layer-search-policy-sweep", default=LAYER_SEARCH_POLICY_SWEEP)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--inner-dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-build-once", dest="build_once", action="store_false")
    parser.set_defaults(build_once=True)
    parser.add_argument("--build-jobs", type=int, default=2)
    args = parser.parse_args()
    parse_args.cached_args = args
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
        suite_dir / "commands.sh", rows, "all", suite_dir=suite_dir, build_first=args.build_once
    )

    gpu_scripts = []
    for gpu_id in gpu_ids:
        gpu_rows = [row for row in rows if row["assigned_gpu"] == gpu_id]
        if not gpu_rows:
            continue
        safe_gpu = "".join(ch if ch.isalnum() else "_" for ch in gpu_id)
        script = suite_dir / f"commands_gpu{safe_gpu}.sh"
        write_command_script(script, gpu_rows, f"gpu{gpu_id}")
        gpu_scripts.append(script)
    write_launcher(suite_dir / "commands_2gpu.sh", suite_dir, gpu_scripts, build_first=args.build_once)
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
