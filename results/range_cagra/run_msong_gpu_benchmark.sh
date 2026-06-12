#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU_ID="${GPU_ID:-0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/range_cagra/msong_gpu_benchmark/${RUN_ID}}"
BUILD_DIR="${RANGE_CAGRA_BUILD_DIR:-cpp/build}"
TEST_BIN="${TEST_BIN:-${BUILD_DIR}/gtests/NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST}"
SKIP_BUILD="${RANGE_CAGRA_SKIP_BUILD:-0}"

BASE="${RANGE_CAGRA_SEGMENT_BASE:-data/msong/msong_base.fvecs}"
WORKLOAD="${RANGE_CAGRA_SEGMENT_WORKLOAD:-generated_queries/order_range_raw_attr/msong/pos_w50}"
WORKLOAD_SWEEP="${RANGE_CAGRA_SEGMENT_WORKLOAD_SWEEP:-}"
MAX_QUERIES="${RANGE_CAGRA_SEGMENT_MAX_QUERIES:-10000}"
TOPK="${RANGE_CAGRA_SEGMENT_TOPK:-10}"
LEAF_SIZE="${RANGE_CAGRA_SEGMENT_LEAF_SIZE:-1000}"
BUILD_ALGO="${RANGE_CAGRA_SEGMENT_BUILD_ALGO:-flat_gnnd}"
GRAPH_DEGREE="${RANGE_CAGRA_SEGMENT_GRAPH_DEGREE:-32}"
INTERMEDIATE_GRAPH_DEGREE="${RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE:-96}"
NN_DESCENT_ITERS="${RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS:-20}"
LAYER_ADAPTIVE_DEGREE="${RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE:-0}"
MIN_GRAPH_DEGREE="${RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE:-0}"
MIN_INTERMEDIATE_GRAPH_DEGREE="${RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE:-0}"
DEGREE_GRANULARITY="${RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY:-8}"
SEARCH_CONCURRENCY="${RANGE_CAGRA_SEGMENT_SEARCH_CONCURRENCY:-2}"
SEARCH_REPEATS="${RANGE_CAGRA_SEGMENT_SEARCH_REPEATS:-10}"
EF="${RANGE_CAGRA_SEGMENT_EF:-64}"
GRAPH_ITERATIONS="${RANGE_CAGRA_SEGMENT_GRAPH_ITERATIONS:-64}"
ENTRY_COUNT="${RANGE_CAGRA_SEGMENT_ENTRY_COUNT:-32}"
EXACT_THREADS="${RANGE_CAGRA_SEGMENT_EXACT_THREADS:-128}"
GRAPH_THREADS="${RANGE_CAGRA_SEGMENT_GRAPH_THREADS:-128}"
SEARCH_SCHEDULE="${RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE:-overlap}"
SEARCH_SCHEDULE_SWEEP="${RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE_SWEEP:-}"
WORKSPACE_MODE="${RANGE_CAGRA_SEGMENT_WORKSPACE_MODE:-device_no_presync}"
SEARCH_SWEEP="${RANGE_CAGRA_SEGMENT_SEARCH_SWEEP:-}"
LOW_LAYER_SEARCH_LAYERS="${RANGE_CAGRA_SEGMENT_LOW_LAYER_SEARCH_LAYERS:-0}"
LOW_LAYER_GRAPH_ITERATIONS="${RANGE_CAGRA_SEGMENT_LOW_LAYER_GRAPH_ITERATIONS:-0}"
SEARCH_ITERATION_POLICY="${RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY:-}"
SEARCH_ITERATION_POLICY_SWEEP="${RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY_SWEEP:-}"
UPPER_LAYER_SEARCH_LAYERS="${RANGE_CAGRA_SEGMENT_UPPER_LAYER_SEARCH_LAYERS:-0}"
UPPER_LAYER_GRAPH_ITERATIONS="${RANGE_CAGRA_SEGMENT_UPPER_LAYER_GRAPH_ITERATIONS:-0}"
ADAPTIVE_MIN_GRAPH_ITERATIONS="${RANGE_CAGRA_SEGMENT_ADAPTIVE_MIN_GRAPH_ITERATIONS:-0}"
ADAPTIVE_MAX_GRAPH_ITERATIONS="${RANGE_CAGRA_SEGMENT_ADAPTIVE_MAX_GRAPH_ITERATIONS:-0}"
ADAPTIVE_ITERATION_GRANULARITY="${RANGE_CAGRA_SEGMENT_ADAPTIVE_ITERATION_GRANULARITY:-1}"

mkdir -p "${OUT_DIR}"

GPU_CSV="${OUT_DIR}/gpu_samples.csv"
RAW_LOG="${OUT_DIR}/benchmark_raw.log"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
PHASE_CSV="${OUT_DIR}/phase_gpu_summary.csv"
SWEEP_CSV="${OUT_DIR}/sweep_summary.csv"
META_TXT="${OUT_DIR}/meta.txt"

cat >"${META_TXT}" <<EOF
run_id=${RUN_ID}
gpu_id=${GPU_ID}
sample_interval=${SAMPLE_INTERVAL}
base=${BASE}
workload=${WORKLOAD}
workload_sweep=${WORKLOAD_SWEEP}
max_queries=${MAX_QUERIES}
topk=${TOPK}
leaf_size=${LEAF_SIZE}
build_algo=${BUILD_ALGO}
graph_degree=${GRAPH_DEGREE}
intermediate_graph_degree=${INTERMEDIATE_GRAPH_DEGREE}
nn_descent_iters=${NN_DESCENT_ITERS}
layer_adaptive_degree=${LAYER_ADAPTIVE_DEGREE}
min_graph_degree=${MIN_GRAPH_DEGREE}
min_intermediate_graph_degree=${MIN_INTERMEDIATE_GRAPH_DEGREE}
degree_granularity=${DEGREE_GRANULARITY}
search_concurrency=${SEARCH_CONCURRENCY}
search_repeats=${SEARCH_REPEATS}
ef=${EF}
graph_iterations=${GRAPH_ITERATIONS}
entry_count=${ENTRY_COUNT}
exact_threads=${EXACT_THREADS}
graph_threads=${GRAPH_THREADS}
search_schedule=${SEARCH_SCHEDULE}
search_schedule_sweep=${SEARCH_SCHEDULE_SWEEP}
search_workspace_mode=${WORKSPACE_MODE}
search_sweep=${SEARCH_SWEEP}
search_iteration_policy=${SEARCH_ITERATION_POLICY}
search_iteration_policy_sweep=${SEARCH_ITERATION_POLICY_SWEEP}
low_layer_search_layers=${LOW_LAYER_SEARCH_LAYERS}
low_layer_graph_iterations=${LOW_LAYER_GRAPH_ITERATIONS}
upper_layer_search_layers=${UPPER_LAYER_SEARCH_LAYERS}
upper_layer_graph_iterations=${UPPER_LAYER_GRAPH_ITERATIONS}
adaptive_min_graph_iterations=${ADAPTIVE_MIN_GRAPH_ITERATIONS}
adaptive_max_graph_iterations=${ADAPTIVE_MAX_GRAPH_ITERATIONS}
adaptive_iteration_granularity=${ADAPTIVE_ITERATION_GRANULARITY}
test_bin=${TEST_BIN}
build_dir=${BUILD_DIR}
skip_build=${SKIP_BUILD}
EOF

if [[ "${SKIP_BUILD}" != "1" ]]; then
  cmake --build "${BUILD_DIR}" --target NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST -j2
fi

echo "epoch_ms,index,memory_used_mb,memory_free_mb,gpu_util_pct,mem_util_pct,power_w,sm_clock_mhz,mem_clock_mhz,temp_c,pstate" >"${GPU_CSV}"

monitor_gpu() {
  local target_pid="$1"
  while kill -0 "${target_pid}" 2>/dev/null; do
    local epoch_ms
    epoch_ms="$(date +%s%3N)"
    local row
    row="$(nvidia-smi -i "${GPU_ID}" \
      --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${row}" ]]; then
      echo "${epoch_ms},${row}" >>"${GPU_CSV}"
    fi
    sleep "${SAMPLE_INTERVAL}"
  done
}

(
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  RANGE_CAGRA_SEGMENT_BASE="${BASE}" \
  RANGE_CAGRA_SEGMENT_WORKLOAD="${WORKLOAD}" \
  RANGE_CAGRA_SEGMENT_WORKLOAD_SWEEP="${WORKLOAD_SWEEP}" \
  RANGE_CAGRA_SEGMENT_MAX_QUERIES="${MAX_QUERIES}" \
  RANGE_CAGRA_SEGMENT_TOPK="${TOPK}" \
  RANGE_CAGRA_SEGMENT_LEAF_SIZE="${LEAF_SIZE}" \
  RANGE_CAGRA_SEGMENT_BUILD_ALGO="${BUILD_ALGO}" \
  RANGE_CAGRA_SEGMENT_GRAPH_DEGREE="${GRAPH_DEGREE}" \
  RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE="${INTERMEDIATE_GRAPH_DEGREE}" \
  RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS="${NN_DESCENT_ITERS}" \
  RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE="${LAYER_ADAPTIVE_DEGREE}" \
  RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE="${MIN_GRAPH_DEGREE}" \
  RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE="${MIN_INTERMEDIATE_GRAPH_DEGREE}" \
  RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY="${DEGREE_GRANULARITY}" \
  RANGE_CAGRA_SEGMENT_SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY}" \
  RANGE_CAGRA_SEGMENT_SEARCH_REPEATS="${SEARCH_REPEATS}" \
  RANGE_CAGRA_SEGMENT_EF="${EF}" \
  RANGE_CAGRA_SEGMENT_GRAPH_ITERATIONS="${GRAPH_ITERATIONS}" \
  RANGE_CAGRA_SEGMENT_ENTRY_COUNT="${ENTRY_COUNT}" \
  RANGE_CAGRA_SEGMENT_EXACT_THREADS="${EXACT_THREADS}" \
  RANGE_CAGRA_SEGMENT_GRAPH_THREADS="${GRAPH_THREADS}" \
  RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE="${SEARCH_SCHEDULE}" \
  RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE_SWEEP="${SEARCH_SCHEDULE_SWEEP}" \
  RANGE_CAGRA_SEGMENT_WORKSPACE_MODE="${WORKSPACE_MODE}" \
  RANGE_CAGRA_SEGMENT_SEARCH_SWEEP="${SEARCH_SWEEP}" \
  RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY="${SEARCH_ITERATION_POLICY}" \
  RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY_SWEEP="${SEARCH_ITERATION_POLICY_SWEEP}" \
  RANGE_CAGRA_SEGMENT_LOW_LAYER_SEARCH_LAYERS="${LOW_LAYER_SEARCH_LAYERS}" \
  RANGE_CAGRA_SEGMENT_LOW_LAYER_GRAPH_ITERATIONS="${LOW_LAYER_GRAPH_ITERATIONS}" \
  RANGE_CAGRA_SEGMENT_UPPER_LAYER_SEARCH_LAYERS="${UPPER_LAYER_SEARCH_LAYERS}" \
  RANGE_CAGRA_SEGMENT_UPPER_LAYER_GRAPH_ITERATIONS="${UPPER_LAYER_GRAPH_ITERATIONS}" \
  RANGE_CAGRA_SEGMENT_ADAPTIVE_MIN_GRAPH_ITERATIONS="${ADAPTIVE_MIN_GRAPH_ITERATIONS}" \
  RANGE_CAGRA_SEGMENT_ADAPTIVE_MAX_GRAPH_ITERATIONS="${ADAPTIVE_MAX_GRAPH_ITERATIONS}" \
  RANGE_CAGRA_SEGMENT_ADAPTIVE_ITERATION_GRANULARITY="${ADAPTIVE_ITERATION_GRANULARITY}" \
  "${TEST_BIN}" \
    --gtest_filter='RangeCagraSegmentTree.OptionalRfannWorkloadSmoke' \
    --gtest_brief=1
) >"${RAW_LOG}" 2>&1 &

bench_pid="$!"
monitor_gpu "${bench_pid}" &
monitor_pid="$!"

set +e
wait "${bench_pid}"
bench_status="$?"
wait "${monitor_pid}" 2>/dev/null
set -e

if [[ "${bench_status}" -ne 0 ]]; then
  echo "benchmark failed with status ${bench_status}; see ${RAW_LOG}" >&2
  exit "${bench_status}"
fi

grep '^range_cagra_segment_tree,' "${RAW_LOG}" >"${OUT_DIR}/result_lines.csv" || true
result_line="$(tail -1 "${OUT_DIR}/result_lines.csv")"
if [[ -z "${result_line}" ]]; then
  echo "missing range_cagra_segment_tree result in ${RAW_LOG}" >&2
  exit 1
fi
echo "${result_line}" >"${OUT_DIR}/result_line.csv"

RESULT_LINE="${result_line}" awk -F',' '
function trim(s) { gsub(/^ +| +$/, "", s); return s }
function val(line, key,    n,a,i,b) {
  n = split(line, a, ",")
  for (i = 1; i <= n; ++i) {
    split(a[i], b, "=")
    if (b[1] == key) return b[2]
  }
  return ""
}
BEGIN {
  line = ENVIRON["RESULT_LINE"]
  rows = val(line, "rows") + 0
  dim = val(line, "dim") + 0
  nq = val(line, "nq") + 0
  topk = val(line, "topk") + 0
  base = val(line, "base")
  workload = val(line, "workload")
  workload_parts = split(workload, workload_path_parts, "/")
  workload_name = workload_path_parts[workload_parts]
  edge_count = val(line, "edge_count") + 0
  build_algo = val(line, "build_algo")
  layer_adaptive = val(line, "layer_adaptive_degree") + 0
  graph_min = val(line, "graph_degree_min") + 0
  graph_max = val(line, "graph_degree_max") + 0
  graph_avg = val(line, "graph_degree_avg") + 0
  inter_min = val(line, "intermediate_graph_degree_min") + 0
  inter_max = val(line, "intermediate_graph_degree_max") + 0
  inter_avg = val(line, "intermediate_graph_degree_avg") + 0
  build_s = val(line, "build_seconds") + 0
  best_s = val(line, "best_search_seconds") + 0
  avg_s = val(line, "avg_search_seconds") + 0
  recall = val(line, "recall_at_k") + 0
  violations = val(line, "filter_violations") + 0
  low_layers = val(line, "low_layer_search_layers") + 0
  low_iters = val(line, "low_layer_graph_iterations") + 0
  low_graph_count = val(line, "low_layer_graph_count") + 0
  exact_threads = val(line, "exact_threads") + 0
  search_width = val(line, "search_width") + 0
  search_iter_policy_id = val(line, "search_iteration_policy_id") + 0
  upper_layers = val(line, "upper_layer_search_layers") + 0
  upper_iters = val(line, "upper_layer_graph_iterations") + 0
  upper_graph_count = val(line, "upper_layer_graph_count") + 0
  search_iter_policy = val(line, "search_iteration_policy")
  search_iter_policy_label = val(line, "search_iteration_policy_label")
  if (search_iter_policy_label == "") search_iter_policy_label = search_iter_policy
  search_iter_base = val(line, "search_iteration_base_graph_iterations") + 0
  search_iter_min = val(line, "search_iteration_min_graph_iterations") + 0
  search_iter_max = val(line, "search_iteration_max_graph_iterations") + 0
  search_iter_avg = val(line, "search_iteration_avg_graph_iterations") + 0
  search_iter_max_layer = val(line, "search_iteration_max_graph_layer") + 0
  search_iter_override_count = val(line, "search_iteration_override_graph_count") + 0
  base_gib = rows * dim * 4 / 1073741824
  edge_gib = edge_count * 4 / 1073741824
  print "base,workload,workload_name,rows,dim,nq,topk,edge_count,base_gib,edge_gib,build_algo,layer_adaptive_degree,search_schedule,search_workspace_mode,search_iteration_policy_id,search_iteration_policy,search_iteration_policy_label,search_iteration_base_graph_iterations,search_iteration_min_graph_iterations,search_iteration_max_graph_iterations,search_iteration_avg_graph_iterations,search_iteration_max_graph_layer,search_iteration_override_graph_count,low_layer_search_layers,low_layer_graph_iterations,low_layer_graph_count,upper_layer_search_layers,upper_layer_graph_iterations,upper_layer_graph_count,search_width,exact_threads,graph_degree_min,graph_degree_max,graph_degree_avg,intermediate_graph_degree_min,intermediate_graph_degree_max,intermediate_graph_degree_avg,build_seconds,best_search_seconds,avg_search_seconds,best_qps,avg_qps,recall_at_k,filter_violations"
  printf "%s,%s,%s,%d,%d,%d,%d,%d,%.6f,%.6f,%s,%d,%s,%s,%d,%s,%s,%d,%d,%d,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%d,%d,%.3f,%.6f,%.6f,%.6f,%.3f,%.3f,%.8f,%d\n", \
    base, workload, workload_name, rows, dim, nq, topk, edge_count, base_gib, edge_gib, build_algo, layer_adaptive, val(line, "search_schedule"), val(line, "search_workspace_mode"), search_iter_policy_id, search_iter_policy, search_iter_policy_label, search_iter_base, search_iter_min, search_iter_max, search_iter_avg, search_iter_max_layer, search_iter_override_count, low_layers, low_iters, low_graph_count, upper_layers, upper_iters, upper_graph_count, search_width, exact_threads, graph_min, graph_max, graph_avg, inter_min, inter_max, inter_avg, build_s, best_s, avg_s, nq / best_s, nq / avg_s, recall, violations
}' >"${SUMMARY_CSV}"

awk -F',' '
function val(key,    i,b) {
  for (i = 1; i <= NF; ++i) {
    split($i, b, "=")
    if (b[1] == key) return b[2]
  }
  return ""
}
BEGIN {
  print "base,workload,workload_name,rows,dim,nq,topk,leaf_size,edge_count,base_gib,edge_gib,build_seconds,search_config_id,search_iteration_policy_id,ef,graph_iterations,graph_search_concurrency,search_width,exact_threads,graph_threads,entry_count,search_repeats,build_algo,layer_adaptive_degree,search_schedule,search_workspace_mode,search_iteration_policy,search_iteration_policy_label,search_iteration_base_graph_iterations,search_iteration_min_graph_iterations,search_iteration_max_graph_iterations,search_iteration_avg_graph_iterations,search_iteration_max_graph_layer,search_iteration_override_graph_count,low_layer_search_layers,low_layer_graph_iterations,low_layer_graph_count,upper_layer_search_layers,upper_layer_graph_iterations,upper_layer_graph_count,graph_degree_min,graph_degree_max,graph_degree_avg,intermediate_graph_degree_min,intermediate_graph_degree_max,intermediate_graph_degree_avg,best_search_seconds,avg_search_seconds,best_qps,avg_qps,recall_at_k,filter_violations,exact_seconds,graph_seconds,merge_seconds,exact_vectors_scanned,graph_node_tasks"
}
{
  rows = val("rows") + 0
  dim = val("dim") + 0
  nq = val("nq") + 0
  edge_count = val("edge_count") + 0
  base_gib = rows * dim * 4 / 1073741824
  edge_gib = edge_count * 4 / 1073741824
  best_s = val("best_search_seconds") + 0
  avg_s = val("avg_search_seconds") + 0
  workload = val("workload")
  workload_parts = split(workload, workload_path_parts, "/")
  workload_name = workload_path_parts[workload_parts]
  search_iter_policy = val("search_iteration_policy")
  search_iter_policy_label = val("search_iteration_policy_label")
  if (search_iter_policy_label == "") search_iter_policy_label = search_iter_policy
  printf "%s,%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,%s,%s,%s,%s,%d,%d,%d,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%d,%d,%.3f,%.6f,%.6f,%.3f,%.3f,%.8f,%d,%.6f,%.6f,%.6f,%d,%d\n", \
    val("base"), workload, workload_name, \
    rows, dim, nq, val("topk") + 0, val("leaf_size") + 0, edge_count, \
    base_gib, edge_gib, val("build_seconds") + 0, \
    val("search_config_id"), val("search_iteration_policy_id") + 0, \
    val("ef") + 0, val("graph_iterations") + 0, \
    val("graph_search_concurrency") + 0, val("search_width") + 0, \
    val("exact_threads") + 0, val("graph_threads") + 0, val("entry_count") + 0, \
    val("search_repeats") + 0, val("build_algo"), val("layer_adaptive_degree") + 0, \
    val("search_schedule"), val("search_workspace_mode"), search_iter_policy, search_iter_policy_label, \
    val("search_iteration_base_graph_iterations") + 0, \
    val("search_iteration_min_graph_iterations") + 0, \
    val("search_iteration_max_graph_iterations") + 0, \
    val("search_iteration_avg_graph_iterations") + 0, \
    val("search_iteration_max_graph_layer") + 0, \
    val("search_iteration_override_graph_count") + 0, \
    val("low_layer_search_layers") + 0, val("low_layer_graph_iterations") + 0, \
    val("low_layer_graph_count") + 0, \
    val("upper_layer_search_layers") + 0, val("upper_layer_graph_iterations") + 0, \
    val("upper_layer_graph_count") + 0, \
    val("graph_degree_min") + 0, val("graph_degree_max") + 0, val("graph_degree_avg") + 0, \
    val("intermediate_graph_degree_min") + 0, val("intermediate_graph_degree_max") + 0, \
    val("intermediate_graph_degree_avg") + 0, best_s, avg_s, nq / best_s, nq / avg_s, \
    val("recall_at_k") + 0, val("filter_violations") + 0, \
    val("exact_seconds") + 0, val("graph_seconds") + 0, val("merge_seconds") + 0, \
    val("exact_vectors_scanned") + 0, val("graph_node_tasks") + 0
}' "${OUT_DIR}/result_lines.csv" >"${SWEEP_CSV}"

awk -F',' '
NR == FNR {
  if ($0 ~ /^range_cagra_phase,/) {
    phase = ""; epoch = 0
    for (i = 1; i <= NF; ++i) {
      split($i, kv, "=")
      if (kv[1] == "phase") phase = kv[2]
      if (kv[1] == "epoch_ms") epoch = kv[2] + 0
    }
    if (phase != "") phase_epoch[phase] = epoch
  }
  next
}
FNR == 1 { next }
{
  t = $1 + 0
  mem = $3 + 0
  gpu = $5 + 0
  memutil = $6 + 0
  power = $7 + 0
  update("whole", mem, gpu, memutil, power)
  if (phase_epoch["build_start"] && phase_epoch["build_end"] &&
      t >= phase_epoch["build_start"] && t <= phase_epoch["build_end"]) {
    update("build", mem, gpu, memutil, power)
  }
  if (phase_epoch["search_start"] && phase_epoch["search_end"] &&
      t >= phase_epoch["search_start"] && t <= phase_epoch["search_end"]) {
    update("search", mem, gpu, memutil, power)
  }
}
function update(phase, mem, gpu, memutil, power) {
  count[phase] += 1
  mem_sum[phase] += mem
  gpu_sum[phase] += gpu
  memutil_sum[phase] += memutil
  power_sum[phase] += power
  if (count[phase] == 1 || mem > mem_peak[phase]) mem_peak[phase] = mem
  if (count[phase] == 1 || gpu > gpu_peak[phase]) gpu_peak[phase] = gpu
  if (count[phase] == 1 || memutil > memutil_peak[phase]) memutil_peak[phase] = memutil
  if (count[phase] == 1 || power > power_peak[phase]) power_peak[phase] = power
}
END {
  print "phase,samples,avg_gpu_util_pct,peak_gpu_util_pct,avg_mem_util_pct,peak_mem_util_pct,avg_memory_used_mb,peak_memory_used_mb,avg_power_w,peak_power_w"
  phases[1] = "whole"; phases[2] = "build"; phases[3] = "search"
  for (i = 1; i <= 3; ++i) {
    p = phases[i]
    if (count[p] > 0) {
      printf "%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", \
        p, count[p], gpu_sum[p] / count[p], gpu_peak[p], memutil_sum[p] / count[p], memutil_peak[p], \
        mem_sum[p] / count[p], mem_peak[p], power_sum[p] / count[p], power_peak[p]
    }
  }
}' "${RAW_LOG}" "${GPU_CSV}" >"${PHASE_CSV}"

echo "out_dir=${OUT_DIR}"
cat "${SUMMARY_CSV}"
cat "${SWEEP_CSV}"
cat "${PHASE_CSV}"
