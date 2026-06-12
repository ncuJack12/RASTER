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
ANALYZER = ROOT / "results" / "range_cagra" / "analyze_search_adaptive_ratio_sweep.py"
SUITE_ROOT = ROOT / "results" / "range_cagra" / "paper_full_suite"

DEFAULT_DATASETS = "audio deep enron msong sift yt8mAudio"
DEFAULT_WIDTHS = "01 05 10 20 35 50"
DEFAULT_RATIO_SPECS = "16:32:4 14:32:2 12:32:4 10:32:2 8:32:4 6:32:2"
BASE_FIXED_CONFIG = "uniform_d32_i96_it20:0:32:96:0:0:8:20"
BASE_ADAPTIVE_CONFIG = "adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20"
SEARCH_SWEEP = (
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


def parse_list(text):
    return [item for item in text.replace(",", " ").split() if item]


def build_workloads(widths):
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
    add_conda_paths("/wjy/conda-envs/cuvs-build-129")
    add_conda_paths("/home/wjy/miniconda3/envs/faiss-cuvs")
    return ":".join(str(path) for path in entries if path.exists())


def parse_ratio_specs(text):
    specs = []
    for item in parse_list(text):
        fields = item.replace("/", ":").split(":")
        if len(fields) not in (2, 3):
            raise SystemExit(f"bad ratio spec '{item}', expected min:max[:granularity]")
        min_it = int(fields[0])
        max_it = int(fields[1])
        granularity = int(fields[2]) if len(fields) == 3 else 1
        if min_it <= 0 or max_it <= 0 or granularity <= 0:
            raise SystemExit(f"bad ratio spec '{item}', values must be positive")
        specs.append((min_it, max_it, granularity))
    if not specs:
        raise SystemExit("empty ratio spec list")
    return specs


def policy_sweep_spec(ratios):
    parts = ["uniform"]
    parts.extend(f"layer_adaptive:{mn}:{mx}:{g}" for mn, mx, g in ratios)
    return ";".join(parts)


def make_row(args, shard_id, datasets, gpu_id, workloads, ratio_spec):
    child_run_id = f"{args.run_id}_search_ratio_s{shard_id}"
    row = {
        "phase": "search_ratio",
        "target_recall": args.target_recall,
        "shard_id": shard_id,
        "assigned_gpu": gpu_id,
        "datasets": datasets,
        "workloads": workloads,
        "leaf_size": args.leaf_size,
        "degree_configs": args.degree_config,
        "search_sweep": args.search_sweep,
        "search_schedule_sweep": args.search_schedule_sweep,
        "ratio_specs": args.ratios,
        "policy_sweep": ratio_spec,
        "max_queries": args.max_queries,
        "search_repeats": args.search_repeats,
        "child_run_id": child_run_id,
        "expected_run_root": f"results/range_cagra/segment_tree_param_sweep/{child_run_id}",
    }
    cmd = [
        "python3",
        str(SWEEP_RUNNER.relative_to(ROOT)),
        "--run-id",
        child_run_id,
        "--sweep",
        "degree",
        "--datasets",
        *datasets,
        "--workload",
        workloads[0],
        "--workload-sweep",
        " ".join(workloads),
        "--gpu-id",
        gpu_id,
        "--build-dir",
        args.build_dir,
        "--data-root",
        args.data_root,
        "--query-root",
        args.query_root,
        "--max-queries",
        str(args.max_queries),
        "--topk",
        str(args.topk),
        "--leaf-size",
        str(args.leaf_size),
        "--build-algo",
        args.build_algo,
        "--degree-configs",
        args.degree_config,
        "--search-sweep",
        args.search_sweep,
        "--search-schedule-sweep",
        args.search_schedule_sweep,
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
        "--search-iteration-policy-sweep",
        ratio_spec,
        "--skip-build",
    ]
    if args.include_risky:
        cmd.append("--include-risky")
    if args.inner_dry_run:
        cmd.append("--dry-run")
    if args.force:
        cmd.append("--force")
    row["command"] = shell_join(cmd)
    return row


PLAN_FIELDS = [
    "phase",
    "target_recall",
    "shard_id",
    "assigned_gpu",
    "datasets",
    "workloads",
    "leaf_size",
    "degree_configs",
    "search_sweep",
    "search_schedule_sweep",
    "ratio_specs",
    "policy_sweep",
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
            writer.writerow({key: serial.get(key, "") for key in PLAN_FIELDS})


def script_header(args):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
    ]
    ld_prefix = ld_library_prefix(args)
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
            f"echo '[{label} {i}/{len(rows)}] gpu={row['assigned_gpu']} datasets={','.join(row['datasets'])}'"
        )
        lines.append(row["command"])
        lines.append("")
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def write_launcher(path, args, suite_dir, gpu_scripts):
    lines = script_header(args)
    if args.build_once:
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


def write_readme(path, args, rows):
    if args.degree_baseline == "fixed":
        degree_purpose = (
            "- Degree is fixed to the uniform build baseline, so the suite isolates the "
            "search-iteration adaptive ablation."
        )
    elif args.degree_baseline == "adaptive":
        degree_purpose = (
            "- Degree adaptivity is already enabled, so the suite measures the incremental "
            "gain of search-iteration adaptivity on top of the final adaptive-degree index."
        )
    else:
        degree_purpose = (
            "- A custom degree config is used; interpret the suite as a search-iteration "
            "ablation under that fixed structural setting."
        )
    lines = [
        "# Range-CAGRA Search-Adaptive Ratio Sweep",
        "",
        f"run_id: `{args.run_id}`",
        f"created_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"target_recall_at_10: `{args.target_recall}`",
        "",
        "## Purpose",
        "",
        "- This suite only evaluates search-iteration adaptivity.",
        degree_purpose,
        "- Leaf size is held fixed so the search-effort effect is not mixed with boundary exact-scan granularity.",
        "- Each child run builds one fixed segment-tree index per dataset and then sweeps `uniform` plus all requested `layer_adaptive:min:max:granularity` policies.",
        "- Selection rule: for each dataset/workload/policy label, keep rows with `recall_at_k >= target_recall` and `filter_violations == 0`, then maximize `best_qps` over the search config grid.",
        "",
        "## Fixed Settings",
        "",
        f"- datasets: `{args.datasets}`",
        f"- widths: `{args.widths}`",
        f"- workloads: `{args.workloads}`",
        f"- leaf_size: `{args.leaf_size}`",
        f"- degree_baseline: `{args.degree_baseline}`",
        f"- degree_config: `{args.degree_config}`",
        f"- search_schedule_sweep: `{args.search_schedule_sweep}`",
        f"- search_sweep: `{args.search_sweep}`",
        "",
        "## Ratio Sweep",
        "",
        f"- ratios: `{args.ratios}`",
        f"- policy_sweep: `{rows[0]['policy_sweep'] if rows else ''}`",
        "",
        "## Run",
        "",
        "```bash",
        f"bash {path.parent.relative_to(ROOT)}/commands_2gpu.sh",
        "```",
        "",
        "## Analyze",
        "",
        "```bash",
        f"python3 results/range_cagra/analyze_search_adaptive_ratio_sweep.py --suite-dir {path.parent.relative_to(ROOT)}",
        "```",
        "",
    ]
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
        print(f"[execute {i}/{len(rows)}] gpu={row['assigned_gpu']} datasets={row['datasets']}", flush=True)
        proc = subprocess.run(row["command"], cwd=ROOT, shell=True, env=env)
        if proc.returncode:
            return proc.returncode
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Range-CAGRA search-adaptive ratio sweep with controlled degree and leaf settings."
    )
    parser.add_argument("--run-id", default=time.strftime("search_adaptive_ratio_sweep_%Y%m%d_%H%M%S"))
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--widths", default=DEFAULT_WIDTHS)
    parser.add_argument(
        "--workloads",
        default="",
        help="Optional explicit workload list. If empty, generated from --widths for pos/ind/neg.",
    )
    parser.add_argument("--ratios", default=DEFAULT_RATIO_SPECS)
    parser.add_argument("--gpu-ids", default=" ".join(detect_gpu_ids()[:2]))
    parser.add_argument("--dataset-shards", type=int, default=0)
    parser.add_argument("--build-dir", default=os.environ.get("RANGE_CAGRA_BUILD_DIR", "cpp/build"))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--query-root", default=str(ROOT / "generated_queries" / "order_range_raw_attr"))
    parser.add_argument("--max-queries", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--build-algo", choices=["flat_gnnd", "segmented_gnnd"], default="flat_gnnd")
    parser.add_argument("--leaf-size", type=int, default=64)
    parser.add_argument(
        "--degree-baseline",
        choices=["fixed", "adaptive", "custom"],
        default="fixed",
        help=(
            "fixed isolates search adaptivity against uniform_d32_i96_it20; "
            "adaptive reproduces the stacked final-index experiment; custom requires --degree-config."
        ),
    )
    parser.add_argument(
        "--degree-config",
        default="",
        help="Override degree config. Required with --degree-baseline custom.",
    )
    parser.add_argument("--search-sweep", default=SEARCH_SWEEP)
    parser.add_argument("--search-schedule-sweep", default="exact_then_graph")
    parser.add_argument("--search-repeats", type=int, default=3)
    parser.add_argument("--exact-threads", type=int, default=128)
    parser.add_argument("--graph-threads", type=int, default=128)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--scratch-guard-gib", type=float, default=3.5)
    parser.add_argument("--max-est-peak-gib", type=float, default=10.0)
    parser.add_argument("--gpu-fraction", type=float, default=0.9)
    parser.add_argument("--include-risky", action="store_true")
    parser.add_argument("--inner-dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-build-once", dest="build_once", action="store_false")
    parser.set_defaults(build_once=True)
    parser.add_argument("--build-jobs", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.degree_config:
        args.degree_baseline = "custom"
    elif args.degree_baseline == "fixed":
        args.degree_config = BASE_FIXED_CONFIG
    elif args.degree_baseline == "adaptive":
        args.degree_config = BASE_ADAPTIVE_CONFIG
    else:
        raise SystemExit("--degree-config is required with --degree-baseline custom")

    if not SWEEP_RUNNER.exists():
        raise SystemExit(f"missing sweep runner: {SWEEP_RUNNER}")
    if not ANALYZER.exists():
        print(f"warning: analyzer not found yet: {ANALYZER}", file=sys.stderr)

    datasets = parse_list(args.datasets)
    workloads = parse_list(args.workloads) if args.workloads else build_workloads(parse_list(args.widths))
    args.workloads = " ".join(workloads)
    ratios = parse_ratio_specs(args.ratios)
    ratio_policy_sweep = policy_sweep_spec(ratios)
    gpu_ids = parse_list(args.gpu_ids) or detect_gpu_ids()
    shard_count = args.dataset_shards if args.dataset_shards > 0 else len(gpu_ids)
    dataset_shards = shard_items(datasets, shard_count)

    rows = [
        make_row(args, shard_id, shard, gpu_ids[shard_id % len(gpu_ids)], workloads, ratio_policy_sweep)
        for shard_id, shard in dataset_shards
    ]

    suite_dir = SUITE_ROOT / args.run_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    write_csv(suite_dir / "suite_plan.csv", rows)
    write_build_once(suite_dir / "build_once.sh", args)
    write_command_script(
        suite_dir / "commands.sh",
        args,
        rows,
        "all",
        suite_dir=suite_dir,
        build_first=args.build_once,
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
    write_launcher(suite_dir / "commands_2gpu.sh", args, suite_dir, gpu_scripts)
    write_readme(suite_dir / "README.md", args, rows)
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
