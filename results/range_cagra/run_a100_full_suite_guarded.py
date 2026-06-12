#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[2]


def now_stamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg):
    print(f"[{dt.datetime.now().strftime('%F %T')}] {msg}", flush=True)


def run_text(cmd, **kwargs):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)


def nvidia_snapshot(gpu_id):
    query = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,"
        "utilization.gpu,utilization.memory,clocks.current.sm,clocks.current.memory,pstate",
        "--format=csv,noheader,nounits",
    ]
    proc = run_text(query)
    q_text = proc.stdout.strip()
    perf = run_text(["nvidia-smi", "-q", "-i", str(gpu_id), "-d", "PERFORMANCE,TEMPERATURE,CLOCK"])
    perf_text = perf.stdout
    active_lines = []
    for line in perf_text.splitlines():
        if ":" not in line:
            continue
        name, value = [part.strip() for part in line.split(":", 1)]
        if value == "Active" and (
            "Slowdown" in name or "Thermal" in name or "Power Brake" in name
        ):
            active_lines.append(line.strip())
    return {
        "query": q_text,
        "perf": perf_text,
        "active_lines": active_lines,
        "thermal_active": any("Thermal Slowdown" in line for line in active_lines),
        "slowdown_active": bool(active_lines),
    }


def temp_from_query(query_text):
    if not query_text:
        return None
    parts = [part.strip() for part in query_text.split(",", 4)]
    if len(parts) < 3:
        return None
    try:
        return float(parts[2])
    except ValueError:
        return None


def wait_until_cool(gpu_id, max_start_temp, check_interval):
    while True:
        snap = nvidia_snapshot(gpu_id)
        temp = temp_from_query(snap["query"])
        if not snap["slowdown_active"] and (temp is None or temp <= max_start_temp):
            log(f"GPU ready: {snap['query']}")
            return
        log(
            "waiting for GPU cooldown: "
            f"{snap['query']} active={'; '.join(snap['active_lines']) or 'none'}"
        )
        time.sleep(check_interval)


def archive_path(path, reason, attempt):
    if not path.exists():
        return None
    dst = path.with_name(f"{path.name}_{reason}_{now_stamp()}_attempt{attempt}")
    path.rename(dst)
    return dst


def status_done(run_root):
    status = run_root / "status.csv"
    if not status.exists():
        return False
    with status.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return bool(rows) and all(row.get("final_status") == "done" for row in rows)


def terminate_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)


def command_for_run(row, args):
    command = row["command"]
    if not args.resume:
        return command
    try:
        parts = shlex.split(command)
    except ValueError:
        return command.replace(" --force", "")
    filtered = [part for part in parts if part != "--force"]
    if len(filtered) == len(parts):
        return command
    return shlex.join(filtered)


def run_row(row, args):
    run_root = ROOT / row["expected_run_root"]
    if args.resume and status_done(run_root):
        log(f"skip completed {row['phase']} run_id={row['run_id']}")
        return "skipped"

    attempt = 1
    while True:
        wait_until_cool(args.gpu_id, args.max_start_temp, args.cooldown_check_interval)
        if run_root.exists() and not args.resume:
            archived = archive_path(run_root, "PREEXISTING", attempt)
            log(f"archived pre-existing run root: {archived}")

        log(
            f"start phase={row['phase']} shard={row['shard_id']} "
            f"run_id={row['run_id']} attempt={attempt}"
        )
        env = os.environ.copy()
        env.setdefault("RANGE_CAGRA_SEGMENT_WORKSPACE_MODE", "device_no_presync")
        command = command_for_run(row, args)
        if command != row["command"]:
            log("resume mode: stripped --force from child command")
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            shell=True,
            env=env,
            preexec_fn=os.setsid,
        )
        last_report = 0.0
        thermal_abort = False
        while True:
            rc = proc.poll()
            snap = nvidia_snapshot(args.gpu_id)
            now = time.time()
            if now - last_report >= args.progress_interval:
                log(
                    f"phase={row['phase']} run_id={row['run_id']} "
                    f"pid={proc.pid} gpu={snap['query']} active="
                    f"{'; '.join(snap['active_lines']) or 'none'}"
                )
                last_report = now
            if rc is not None:
                break
            if snap["thermal_active"] or (
                args.abort_on_any_slowdown and snap["slowdown_active"]
            ):
                log(
                    f"thermal/slowdown detected; aborting phase={row['phase']} "
                    f"run_id={row['run_id']} active={'; '.join(snap['active_lines'])}"
                )
                terminate_process_group(proc)
                thermal_abort = True
                if args.archive_thermal_partials:
                    archived = archive_path(run_root, "ABORTED_THERMAL", attempt)
                    if archived:
                        log(f"archived throttled partial output: {archived}")
                else:
                    log(f"preserving partial output for resume: {run_root.relative_to(ROOT)}")
                break
            time.sleep(args.monitor_interval)

        if thermal_abort:
            if attempt >= args.max_retries:
                raise RuntimeError(f"max thermal retries exceeded for {row['run_id']}")
            attempt += 1
            time.sleep(args.cooldown_sleep)
            continue
        if proc.returncode != 0:
            raise RuntimeError(f"command failed rc={proc.returncode}: {row['run_id']}")
        if not status_done(run_root):
            raise RuntimeError(f"missing or incomplete status after command: {run_root}")
        log(f"done phase={row['phase']} run_id={row['run_id']}")
        return "done"


def collect_and_package(suite_dir, run_id, thresholds):
    log("collecting paper results")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "results" / "range_cagra" / "collect_a100_paper_results.py"),
            "--suite-dir",
            str(suite_dir.relative_to(ROOT)),
            "--thresholds",
            thresholds,
        ],
        cwd=ROOT,
        check=True,
    )

    packaged_dir = ROOT / "results" / "range_cagra" / "paper_full_suite" / "packaged"
    packaged_dir.mkdir(parents=True, exist_ok=True)
    package = packaged_dir / f"{run_id}_results_{now_stamp()}.tar.gz"

    manifest = suite_dir / "package_manifest.json"
    roots = [
        ROOT / "results" / "range_cagra" / "segment_tree_param_sweep" / p.name
        for p in (ROOT / "results" / "range_cagra" / "segment_tree_param_sweep").glob(f"{run_id}*")
        if p.is_dir()
    ]
    snapshot = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "suite_dir": str(suite_dir.relative_to(ROOT)),
        "result_roots": [str(path.relative_to(ROOT)) for path in sorted(roots)],
        "package": str(package.relative_to(ROOT)),
    }
    manifest.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    rels = [str(suite_dir.relative_to(ROOT))]
    rels.extend(str(path.relative_to(ROOT)) for path in sorted(roots))
    log(f"packing {len(rels)} paths into {package}")
    subprocess.run(["tar", "-czf", str(package), *rels], cwd=ROOT, check=True)
    log(f"package ready: {package}")
    return package


def parse_args():
    parser = argparse.ArgumentParser(description="Run an A100 paper suite with thermal guarding.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-start-temp", type=float, default=float(os.environ.get("A100_MAX_START_TEMP", "76")))
    parser.add_argument("--monitor-interval", type=float, default=float(os.environ.get("A100_MONITOR_INTERVAL", "15")))
    parser.add_argument("--progress-interval", type=float, default=float(os.environ.get("A100_PROGRESS_INTERVAL", "120")))
    parser.add_argument("--cooldown-check-interval", type=float, default=float(os.environ.get("A100_COOLDOWN_CHECK_INTERVAL", "30")))
    parser.add_argument("--cooldown-sleep", type=float, default=float(os.environ.get("A100_COOLDOWN_SLEEP", "300")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("A100_MAX_THERMAL_RETRIES", "20")))
    parser.add_argument("--thresholds", default="0.90 0.95 0.98 0.99")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--archive-thermal-partials", action="store_true")
    parser.add_argument("--abort-on-any-slowdown", action="store_true", default=True)
    parser.add_argument("--allow-power-cap", dest="abort_on_any_slowdown", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = (ROOT / args.suite_dir).resolve()
    plan_path = suite_dir / "suite_plan.csv"
    if not plan_path.exists():
        raise SystemExit(f"missing suite_plan.csv: {plan_path}")
    run_id = suite_dir.name
    log(f"suite={suite_dir} gpu={args.gpu_id}")
    with plan_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for i, row in enumerate(rows, start=1):
        log(f"row {i}/{len(rows)}")
        run_row(row, args)
    package = collect_and_package(suite_dir, run_id, args.thresholds)
    log(f"complete package={package}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
