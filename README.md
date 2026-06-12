# RASTER

## 🚀 Quick Start: Download Data and Workloads

RASTER experiments use released base-vector datasets plus generated
order-range query workloads. The public data archive is hosted here:

```text
https://cloud.tsinghua.edu.cn/d/05dab9941ca9418e8894/
```

After cloning this repository, use the downloader script to fetch only the
datasets you need. It downloads split archives, verifies SHA256 checksums,
extracts them, and validates that the base vectors and all 33 workloads are
present.

```bash
git clone https://github.com/ncuJack12/RASTER.git
cd RASTER

# Show available datasets and archive sizes.
scripts/download_raster_data.sh --list

# Download one dataset and its workloads.
scripts/download_raster_data.sh msong

# Download multiple selected datasets.
scripts/download_raster_data.sh msong sift glove-100

# Download all 11 released datasets and workloads.
scripts/download_raster_data.sh --all
```

The extracted layout matches the benchmark defaults:

```text
data/<dataset>/...
generated_queries/order_range_raw_attr/<dataset>/<workload>/
```

To verify an already-downloaded dataset without downloading again:

```bash
scripts/download_raster_data.sh --verify-only msong
```

The script requires `curl`, `python3`, `zstd`, `tar`, `sha256sum`, and `stat`.

## 🧭 Overview

RASTER is a GPU implementation of ordered range-filtered approximate nearest
neighbor search built inside the RAPIDS cuVS C++ codebase. It extends CAGRA with
a segment-tree index over ordered vector ids. A query consists of a vector and an
inclusive id range `[L, R]`; returned neighbors must lie inside that range.

The implementation keeps one global copy of the base vectors on the GPU. Each
reusable segment-tree node stores graph edges and lightweight local-to-global
metadata, so local graph search can return global vector ids without replicating
the high-dimensional vector table across nodes.

## 🧩 Algorithm

RASTER handles each ordered range query in three stages:

1. Build a heap-numbered segment tree over ordered base-vector ids.
2. Decompose each query range into exact boundary fragments and fully covered
   internal segment-tree nodes.
3. Scan boundary fragments exactly on the GPU, search internal nodes with local
   CAGRA-style graph tasks, and merge candidates into the final top-k result.

The main implementation supports three tuning axes:

- `layer_adaptive_degree`: allocates higher graph degree to upper segment-tree
  nodes and lower degree to small lower-layer nodes.
- `search_iteration_policy`: controls how graph-search iterations vary by
  segment-tree layer. Supported policies are `uniform`, `lower_layers`,
  `upper_layers`, and `layer_adaptive`.
- `leaf_size`: controls the tradeoff between boundary exact scanning and graph
  task granularity.

All benchmark rows report `filter_violations`; a valid range-filtered result
must have `filter_violations == 0`.

## 🗂️ Code Layout

Core implementation:

```text
cpp/src/neighbors/detail/range_cagra/
```

Important files:

| File | Purpose |
| --- | --- |
| `range_cagra_types.cuh` | Global dataset view, graph pool, graph metadata |
| `range_cagra_build.cuh` | Local range graph construction from global vectors |
| `range_cagra_search.cuh` | Single-range graph search utilities |
| `range_cagra_segment_tree_types.cuh` | Segment-tree layout and node ranges |
| `range_cagra_segment_tree_layering.cuh` | Shared layer numbering for build and search policies |
| `range_cagra_uniform_build_policy.cuh` | Uniform graph-degree baseline |
| `range_cagra_adaptive_build_policy.cuh` | Layer-adaptive graph-degree policy |
| `range_cagra_segment_tree_search_policy.cuh` | Layer-aware search-iteration policy |
| `range_cagra_segment_tree.cuh` | Index build, range decomposition, exact tasks, graph tasks, merge |
| `range_cagra_segment_tree_workspace_search.cuh` | Reusable-workspace search path |

Tests and benchmark harness:

```text
cpp/tests/neighbors/range_cagra/
```

Experiment drivers:

```text
results/range_cagra/run_msong_gpu_benchmark.sh
results/range_cagra/run_segment_tree_param_sweep.py
results/range_cagra/run_paper_full_experiments.py
results/range_cagra/run_a100_fast90_paper_experiments.py
results/range_cagra/run_a100_full_suite_guarded.py
results/range_cagra/collect_a100_paper_results.py
```

The `results/range_cagra` directory is configured to keep scripts in git while
ignoring generated benchmark output trees.

## 🛠️ Requirements

RASTER is part of a cuVS source tree, so it uses the normal cuVS build
dependencies:

- Linux
- NVIDIA GPU and CUDA toolkit compatible with the cuVS checkout
- CMake and Ninja
- C++20/CUDA20-capable compiler toolchain
- RAPIDS/cuVS dependencies, typically through a RAPIDS development conda
  environment

For a fresh environment, follow the cuVS dependency setup for this checkout
first, then build the targets below.

## 🔨 Build

From the repository root, build cuVS and tests with the repo build script:

```bash
./build.sh libcuvs tests -n -v \
  --limit-tests=NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST
```

If you already have a configured cuVS build tree, the focused target can be
built directly:

```bash
cmake --build cpp/build --target NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST -j$(nproc)
```

On systems where runtime libraries are not found automatically, put the build
library directory and the active conda environment first:

```bash
export LD_LIBRARY_PATH="$PWD/cpp/build:$PWD/cpp/build/_deps/rmm-build:${CONDA_PREFIX:-}/lib:${CONDA_PREFIX:-}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
```

## ✅ Test

Run the focused synthetic tests:

```bash
cpp/build/gtests/NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST \
  --gtest_filter='RangeCagraSegmentTree.DecompositionUsesAtMostFourExactLeafTasks:RangeCagraSegmentTree.LayerAdaptiveDegreeDecreasesTowardLeaves:RangeCagraSegmentTree.LayerAdaptiveSearchIterationsOnlyReduceBelowBase:RangeCagraSegmentTree.SearchIterationPolicySweepParsesLayerAdaptiveRatios:RangeCagraSegmentTree.BuildsAndSearchesSyntheticSegmentTree'
```

To run all RASTER segment-tree tests:

```bash
cpp/build/gtests/NEIGHBORS_RANGE_CAGRA_SEGMENT_TREE_TEST
```

The optional RFANN workload smoke test is skipped unless data and workload
paths are available.

## 📦 Data Layout

Benchmark scripts expect dense base vectors and generated order-range workloads
under these default roots:

```text
data/<dataset>/<dataset>_base.fvecs
generated_queries/order_range_raw_attr/<dataset>/<workload>/
```

The workload directory contains the generated query vectors, range ids, and
ground truth files used by the benchmark harness. The generated data directories
are ignored by git because they are dataset artifacts, not source code.

To download the released RASTER experiment data and queries:

```bash
scripts/download_raster_data.sh --list
scripts/download_raster_data.sh msong
```

Multiple datasets can be selected in one command:

```bash
scripts/download_raster_data.sh msong sift glove-100
```

To download everything used by the paper-scale suite:

```bash
scripts/download_raster_data.sh --all
```

The downloader retrieves split archives from the public data link, verifies
SHA256 checksums from the remote manifest, and extracts the files into the
layout shown above. Use `--output-root <dir>` to extract somewhere other than
the repository root.

After extraction, the script validates that the base vector file and all 33
order-range workloads are present for each selected dataset. To check an
already-downloaded layout without downloading again:

```bash
scripts/download_raster_data.sh --verify-only msong
```

## 🧪 Run A Workload Benchmark

The shell harness runs the segment-tree benchmark executable and records
`summary.csv`, `sweep_summary.csv`, `phase_gpu_summary.csv`, `result_lines.csv`,
and `benchmark_raw.log`.

Example single-workload run:

```bash
export RANGE_CAGRA_BUILD_DIR=cpp/build
export RANGE_CAGRA_SEGMENT_BASE=data/msong/msong_base.fvecs
export RANGE_CAGRA_SEGMENT_WORKLOAD=generated_queries/order_range_raw_attr/msong/pos_w50
export RANGE_CAGRA_SEGMENT_MAX_QUERIES=10000
export RANGE_CAGRA_SEGMENT_TOPK=10
export RANGE_CAGRA_SEGMENT_LEAF_SIZE=64
export RANGE_CAGRA_SEGMENT_BUILD_ALGO=flat_gnnd
export RANGE_CAGRA_SEGMENT_GRAPH_DEGREE=32
export RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE=96
export RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE=1
export RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE=8
export RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE=24
export RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY=8
export RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS=20
export RANGE_CAGRA_SEGMENT_SEARCH_SWEEP='16:8:1:12;24:16:1:16;32:24:1:16;48:32:1:24'
export RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY=layer_adaptive
export RANGE_CAGRA_SEGMENT_ADAPTIVE_MIN_GRAPH_ITERATIONS=8
export RANGE_CAGRA_SEGMENT_ADAPTIVE_MAX_GRAPH_ITERATIONS=32
export RANGE_CAGRA_SEGMENT_ADAPTIVE_ITERATION_GRANULARITY=4
export RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE=exact_then_graph
export RANGE_CAGRA_SEGMENT_WORKSPACE_MODE=device_no_presync
export RANGE_CAGRA_SEGMENT_SEARCH_REPEATS=3
export OUT_DIR=results/range_cagra/msong_example

bash results/range_cagra/run_msong_gpu_benchmark.sh
```

Each search sweep item has the form:

```text
itopk_size:graph_iterations:search_width:entry_count
```

## 📈 Run A Parameter Sweep

Use the Python driver to build each dataset/config once and sweep workloads and
search settings in the same process:

```bash
python3 results/range_cagra/run_segment_tree_param_sweep.py \
  --run-id msong_raster_example \
  --sweep degree \
  --datasets msong \
  --workload pos_w01 \
  --workload-sweep 'pos_w01 pos_w10 pos_w50 ind_w10 neg_w10' \
  --gpu-id 0 \
  --build-dir cpp/build \
  --data-root data \
  --query-root generated_queries/order_range_raw_attr \
  --max-queries 10000 \
  --topk 10 \
  --leaf-size 64 \
  --build-algo flat_gnnd \
  --degree-configs 'adaptive_d32_i96_min8_i24_it20:1:32:96:8:24:8:20' \
  --search-sweep '16:8:1:12;24:16:1:16;32:24:1:16;48:32:1:24' \
  --search-schedule-sweep exact_then_graph \
  --search-iteration-policy layer_adaptive \
  --adaptive-min-graph-iterations 8 \
  --adaptive-max-graph-iterations 32 \
  --adaptive-iteration-granularity 4 \
  --search-repeats 3 \
  --exact-threads 128 \
  --graph-threads 128 \
  --sample-interval 0.5 \
  --scratch-guard-gib 3.5 \
  --max-est-peak-gib 10.0 \
  --gpu-fraction 0.9 \
  --skip-build
```

The output root is:

```text
results/range_cagra/segment_tree_param_sweep/msong_raster_example/
```

Useful output files:

| File | Contents |
| --- | --- |
| `plan.csv` | Planned dataset/config tasks |
| `status.csv` | Per-task completion status |
| `aggregate_sweep.csv` | All per-workload/per-search rows |
| `aggregate_summary.csv` | Per-task summaries |
| `aggregate_phase_gpu.csv` | GPU memory/utilization samples |

## 📚 Paper-Scale Suite

The full-suite generator is:

```bash
python3 results/range_cagra/run_paper_full_experiments.py \
  --run-id raster_full_$(date +%Y%m%d_%H%M%S) \
  --phases smoke layer_degree layer_search leaf_size main_algo \
  --build-dir cpp/build \
  --gpu-ids '0' \
  --workload-preset full
```

This writes a suite directory under:

```text
results/range_cagra/paper_full_suite/<run-id>/
```

Review `suite_plan.csv`, then run the generated command script. On a single GPU:

```bash
bash results/range_cagra/paper_full_suite/<run-id>/commands_gpu0.sh
```

After the child runs finish, aggregate:

```bash
python3 results/range_cagra/collect_a100_paper_results.py \
  --suite-dir results/range_cagra/paper_full_suite/<run-id> \
  --thresholds '0.90 0.95 0.98 0.99'
```

## 📝 Notes

- Query ranges use zero-based inclusive ids `[L, R]`.
- Rows with nonzero `filter_violations` are invalid and should be excluded.
- Build/load time and query QPS are reported separately.
- Large datasets should use `--workload-sweep` so each built graph pool is
  reused across all workload widths.
