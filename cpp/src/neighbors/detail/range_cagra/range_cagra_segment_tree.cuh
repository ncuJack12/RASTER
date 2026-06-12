/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../cagra/jit_lto_kernels/search_single_cta_device_helpers.cuh"
#include "../cagra/topk_by_radix.cuh"
#include "range_cagra_segment_tree_types.cuh"
#include "range_cagra_adaptive_build_policy.cuh"
#include "range_cagra_segment_tree_search_policy.cuh"
#include "range_cagra_build.cuh"
#include "range_cagra_types.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {

constexpr int kRangeCagraMaxTopK                 = 32;
constexpr int kRangeCagraMaxExactTasksPerQuery   = 4;
constexpr int kRangeCagraMaxGraphItopk           = 512;
constexpr int kRangeCagraMaxGraphDegree          = 128;
constexpr int kRangeCagraMaxGraphSearchWidth     = 32;
constexpr int kRangeCagraDistanceTeamSize        = 16;
constexpr int kRangeCagraDistanceDatasetBlockDim = 512;
constexpr int kRangeCagraMinVisitedHashBitlen    = 11;
constexpr int kRangeCagraMaxVisitedHashBitlen    = 13;
constexpr int kRangeCagraTopkWorkspaceWords      = 3;
constexpr float kRangeCagraHashMaxFillRate       = 0.5f;
constexpr std::uint32_t kRangeCagraExpandedBit   = 0x80000000u;
constexpr std::uint32_t kRangeCagraLocalIdMask   = 0x7fffffffu;
constexpr std::uint32_t kRangeCagraInvalidId     = 0xffffffffu;

struct DeviceSegmentTreeLayoutView {
  std::int64_t rows        = 0;
  int leaf_size            = 0;
  std::int64_t leaf_blocks = 0;
  std::int64_t leaf_base   = 0;
  const int* graph_slot    = nullptr;
};

struct ExactSearchTask {
  int query_id       = 0;
  std::int64_t left  = 0;
  std::int64_t right = -1;
};

struct GraphSearchTask {
  int query_id = 0;
  int graph_id = -1;
};

struct HostRangeGraphMeta {
  std::int64_t node_id = 0;
  SegmentNodeRange range;
  int graph_id                  = -1;
  int degree                    = 0;
  int intermediate_degree       = 0;
  std::int64_t edge_offset      = 0;
};

struct GraphSearchProfileCounters {
  unsigned long long graph_tasks                = 0;
  unsigned long long iterations_started         = 0;
  unsigned long long iterations_completed       = 0;
  unsigned long long terminated_iterations      = 0;
  unsigned long long hash_reset_count           = 0;
  unsigned long long initial_candidates         = 0;
  unsigned long long candidate_slots            = 0;
  unsigned long long valid_child_candidates     = 0;
  unsigned long long inserted_child_candidates  = 0;
  unsigned long long duplicate_child_candidates = 0;
  unsigned long long distance_evaluations       = 0;
  unsigned long long cycles_total               = 0;
  unsigned long long cycles_query_init          = 0;
  unsigned long long cycles_initial_prepare     = 0;
  unsigned long long cycles_initial_distance    = 0;
  unsigned long long cycles_initial_merge       = 0;
  unsigned long long cycles_hash_reset          = 0;
  unsigned long long cycles_pickup              = 0;
  unsigned long long cycles_clear_candidates    = 0;
  unsigned long long cycles_expand_prepare      = 0;
  unsigned long long cycles_expand_distance     = 0;
  unsigned long long cycles_iter_merge          = 0;
  unsigned long long cycles_output              = 0;
};

struct SegmentTreeRangeCagraIndex {
  SegmentTreeLayout layout;
  DeviceRangeGraphPool graph_pool;
  std::vector<HostRangeGraphMeta> graph_metas;
  double build_seconds = 0.0;
};

struct SegmentTreeSearchStats {
  std::int64_t exact_task_count      = 0;
  std::int64_t graph_node_task_count = 0;
  std::int64_t exact_vectors_scanned = 0;
  SegmentTreeSearchIterationPolicy search_iteration_policy =
    SegmentTreeSearchIterationPolicy::kUniform;
  int search_iteration_base_graph_iterations = 0;
  int search_iteration_min_graph_iterations  = 0;
  int search_iteration_max_graph_iterations  = 0;
  double search_iteration_avg_graph_iterations = 0.0;
  int search_iteration_max_graph_layer          = 0;
  std::int64_t search_iteration_override_graph_count = 0;
  int low_layer_search_layers                         = 0;
  int low_layer_graph_iterations                      = 0;
  std::int64_t low_layer_graph_count                  = 0;
  int upper_layer_search_layers                       = 0;
  int upper_layer_graph_iterations                    = 0;
  std::int64_t upper_layer_graph_count                = 0;
  double exact_seconds               = 0.0;
  double graph_seconds               = 0.0;
  double merge_seconds               = 0.0;
  double search_seconds              = 0.0;
  bool graph_profile_enabled         = false;
  GraphSearchProfileCounters graph_profile;
};

enum class SegmentTreeSearchSchedule {
  kOverlap,
  kGraphThenExact,
  kExactThenGraph,
};

[[nodiscard]] inline std::int64_t ceil_div_i64(std::int64_t a, std::int64_t b)
{
  return (a + b - 1) / b;
}

[[nodiscard]] inline int round_up_multiple(int value, int multiple)
{
  return ((value + multiple - 1) / multiple) * multiple;
}

[[nodiscard]] __host__ __device__ inline int range_cagra_distance_query_workspace_len(int dim)
{
  return ((dim + kRangeCagraDistanceDatasetBlockDim - 1) / kRangeCagraDistanceDatasetBlockDim) *
         kRangeCagraDistanceDatasetBlockDim;
}

[[nodiscard]] inline int round_cagra_itopk_size(int requested)
{
  return round_up_multiple(std::max(1, requested), 32);
}

[[nodiscard]] inline int range_cagra_max_itopk_bin(int itopk_size)
{
  if (itopk_size <= 256) { return 256; }
  return 512;
}

[[nodiscard]] inline int range_cagra_radix_workspace_words(int max_itopk)
{
  return static_cast<int>(
    cuvs::neighbors::cagra::detail::single_cta_search::topk_by_radix_sort_base::smem_size(
      static_cast<std::uint32_t>(max_itopk)));
}

[[nodiscard]] inline int range_cagra_hash_bitlen(int itopk_size, int max_candidates)
{
  const int max_visited_nodes = itopk_size + max_candidates;
  int bitlen                  = kRangeCagraMinVisitedHashBitlen;
  while (static_cast<float>(1 << bitlen) * kRangeCagraHashMaxFillRate <
         static_cast<float>(max_visited_nodes)) {
    ++bitlen;
  }
  return bitlen;
}

[[nodiscard]] inline int range_cagra_hash_reset_interval(int itopk_size,
                                                         int max_candidates,
                                                         int hash_bitlen)
{
  const int hash_fill_limit =
    static_cast<int>(static_cast<float>(1 << hash_bitlen) * kRangeCagraHashMaxFillRate);
  if (max_candidates <= 0 || itopk_size + max_candidates >= hash_fill_limit) { return 1; }

  int interval = 1;
  while (itopk_size + max_candidates * (interval + 1) <= hash_fill_limit) {
    ++interval;
  }
  return interval;
}

[[nodiscard]] inline std::int64_t next_power_of_two_i64(std::int64_t value)
{
  std::int64_t out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

[[nodiscard]] inline int floor_log2_u64(std::uint64_t value) { return 63 - __builtin_clzll(value); }

[[nodiscard]] inline int max_graph_tasks_per_query(SegmentTreeLayout const& layout)
{
  int height = 0;
  for (std::int64_t width = layout.leaf_base; width > 1; width >>= 1) {
    ++height;
  }
  return std::max(1, 2 * height + 2);
}

[[nodiscard]] inline SegmentTreeLayout make_segment_tree_layout(std::int64_t rows,
                                                                int dim,
                                                                int leaf_size)
{
  RAFT_EXPECTS(rows > 0, "segment tree rows must be positive");
  RAFT_EXPECTS(dim > 0, "segment tree dim must be positive");
  RAFT_EXPECTS(leaf_size > 0, "leaf_size must be positive");

  SegmentTreeLayout layout;
  layout.rows        = rows;
  layout.dim         = dim;
  layout.leaf_size   = leaf_size;
  layout.leaf_blocks = ceil_div_i64(rows, leaf_size);
  layout.leaf_base   = next_power_of_two_i64(layout.leaf_blocks);
  layout.graph_slot.assign(static_cast<std::size_t>(2 * layout.leaf_base), -1);
  return layout;
}

[[nodiscard]] inline SegmentNodeRange range_from_node_id(std::int64_t node_id,
                                                         SegmentTreeLayout const& layout)
{
  if (node_id <= 0 || node_id >= 2 * layout.leaf_base) { return {}; }

  const int depth            = floor_log2_u64(static_cast<std::uint64_t>(node_id));
  const std::int64_t base    = std::int64_t{1} << depth;
  const std::int64_t offset  = node_id - base;
  const std::int64_t blocks  = layout.leaf_base >> depth;
  const std::int64_t block_l = offset * blocks;
  if (blocks <= 0 || block_l >= layout.leaf_blocks) { return {}; }

  const std::int64_t block_r = std::min(layout.leaf_blocks - 1, (offset + 1) * blocks - 1);
  const std::int64_t vec_l   = block_l * static_cast<std::int64_t>(layout.leaf_size);
  if (vec_l >= layout.rows) { return {}; }
  const std::int64_t vec_r =
    std::min(layout.rows - 1, (block_r + 1) * static_cast<std::int64_t>(layout.leaf_size) - 1);
  return {block_l, block_r, vec_l, vec_r};
}

inline void decompose_query_range(std::int64_t query_id,
                                  std::int64_t left,
                                  std::int64_t right,
                                  SegmentTreeLayout const& layout,
                                  std::vector<ExactSearchTask>& exact_tasks,
                                  std::vector<GraphSearchTask>& graph_tasks,
                                  std::int64_t& exact_vectors_scanned)
{
  left  = std::max<std::int64_t>(0, left);
  right = std::min<std::int64_t>(layout.rows - 1, right);
  if (right < left) { return; }

  auto add_exact = [&](std::int64_t l, std::int64_t r) {
    if (r < l) { return; }
    exact_tasks.push_back({static_cast<int>(query_id), l, r});
    exact_vectors_scanned += r - l + 1;
  };

  auto use_node = [&](std::int64_t node_id) {
    if (node_id > 0 && node_id < static_cast<std::int64_t>(layout.graph_slot.size())) {
      const int slot = layout.graph_slot[static_cast<std::size_t>(node_id)];
      if (slot >= 0) {
        graph_tasks.push_back({static_cast<int>(query_id), slot});
        return;
      }
    }
    auto range = range_from_node_id(node_id, layout);
    if (range.valid()) { add_exact(range.vec_l, range.vec_r); }
  };

  const std::int64_t b_l = left / layout.leaf_size;
  const std::int64_t b_r = right / layout.leaf_size;
  if (b_l == b_r) {
    add_exact(left, right);
    return;
  }

  std::int64_t full_l                 = b_l;
  std::int64_t full_r                 = b_r;
  const std::int64_t left_block_start = b_l * static_cast<std::int64_t>(layout.leaf_size);
  const std::int64_t left_block_end =
    std::min(layout.rows - 1, (b_l + 1) * static_cast<std::int64_t>(layout.leaf_size) - 1);
  const std::int64_t right_block_start = b_r * static_cast<std::int64_t>(layout.leaf_size);
  const std::int64_t right_block_end =
    std::min(layout.rows - 1, (b_r + 1) * static_cast<std::int64_t>(layout.leaf_size) - 1);

  if (left > left_block_start) {
    add_exact(left, left_block_end);
    full_l = b_l + 1;
  }
  if (right < right_block_end) {
    add_exact(right_block_start, right);
    full_r = b_r - 1;
  }
  if (full_l > full_r) { return; }

  std::int64_t lo = layout.leaf_base + full_l;
  std::int64_t hi = layout.leaf_base + full_r;
  while (lo <= hi) {
    if (lo % 2 == 1) {
      use_node(lo);
      ++lo;
    }
    if (hi % 2 == 0) {
      use_node(hi);
      --hi;
    }
    lo /= 2;
    hi /= 2;
  }
}

namespace kernels {

struct DeviceTaskCounters {
  int exact_count                    = 0;
  int graph_node_count               = 0;
  int overflow_count                 = 0;
  unsigned long long exact_vec_count = 0;
};

__device__ inline std::int64_t device_min_i64(std::int64_t a, std::int64_t b)
{
  return a < b ? a : b;
}

__device__ inline std::int64_t device_max_i64(std::int64_t a, std::int64_t b)
{
  return a > b ? a : b;
}

__device__ inline SegmentNodeRange device_range_from_node_id(std::int64_t node_id,
                                                             DeviceSegmentTreeLayoutView layout)
{
  if (node_id <= 0 || node_id >= 2 * layout.leaf_base) { return {}; }

  int depth = 0;
  for (std::int64_t cursor = node_id; cursor > 1; cursor >>= 1) {
    ++depth;
  }
  const std::int64_t base    = std::int64_t{1} << depth;
  const std::int64_t offset  = node_id - base;
  const std::int64_t blocks  = layout.leaf_base >> depth;
  const std::int64_t block_l = offset * blocks;
  if (blocks <= 0 || block_l >= layout.leaf_blocks) { return {}; }

  const std::int64_t block_r = device_min_i64(layout.leaf_blocks - 1, (offset + 1) * blocks - 1);
  const std::int64_t vec_l   = block_l * static_cast<std::int64_t>(layout.leaf_size);
  if (vec_l >= layout.rows) { return {}; }
  const std::int64_t vec_r = device_min_i64(
    layout.rows - 1, (block_r + 1) * static_cast<std::int64_t>(layout.leaf_size) - 1);
  return {block_l, block_r, vec_l, vec_r};
}

__device__ inline int append_exact_task(int query_id,
                                        std::int64_t left,
                                        std::int64_t right,
                                        ExactSearchTask* exact_tasks,
                                        int max_exact_tasks,
                                        DeviceTaskCounters* counters)
{
  if (right < left) { return -1; }
  const int slot = atomicAdd(&counters->exact_count, 1);
  if (slot < max_exact_tasks) {
    exact_tasks[slot] = {query_id, left, right};
    atomicAdd(&counters->exact_vec_count, static_cast<unsigned long long>(right - left + 1));
    return slot;
  }
  atomicAdd(&counters->overflow_count, 1);
  return -1;
}

__device__ inline int append_graph_node_task(int query_id,
                                             int graph_id,
                                             GraphSearchTask* graph_tasks,
                                             int max_graph_tasks,
                                             DeviceTaskCounters* counters)
{
  const int slot = atomicAdd(&counters->graph_node_count, 1);
  if (slot < max_graph_tasks) {
    graph_tasks[slot] = {query_id, graph_id};
    return slot;
  }
  atomicAdd(&counters->overflow_count, 1);
  return -1;
}

RAFT_KERNEL decompose_ranges_kernel(const std::int64_t* __restrict__ query_ranges,
                                    int n_queries,
                                    DeviceSegmentTreeLayoutView layout,
                                    ExactSearchTask* __restrict__ exact_tasks,
                                    int max_exact_tasks,
                                    int* __restrict__ exact_task_indices_by_query,
                                    int* __restrict__ exact_task_counts,
                                    GraphSearchTask* __restrict__ graph_tasks,
                                    int max_graph_tasks,
                                    int max_graph_tasks_per_query,
                                    int* __restrict__ graph_task_indices_by_query,
                                    int* __restrict__ graph_task_counts,
                                    DeviceTaskCounters* __restrict__ counters)
{
  const int query_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query_id >= n_queries) { return; }

  int exact_count = 0;
  int graph_count = 0;

  auto add_exact = [&](std::int64_t l, std::int64_t r) {
    if (exact_count >= kRangeCagraMaxExactTasksPerQuery) {
      atomicAdd(&counters->overflow_count, 1);
      return;
    }
    const int slot = append_exact_task(query_id, l, r, exact_tasks, max_exact_tasks, counters);
    if (slot >= 0) {
      exact_task_indices_by_query[query_id * kRangeCagraMaxExactTasksPerQuery + exact_count] = slot;
      ++exact_count;
    }
  };

  auto use_node = [&](std::int64_t node_id) {
    if (node_id > 0 && node_id < 2 * layout.leaf_base) {
      const int slot = layout.graph_slot[node_id];
      if (slot >= 0) {
        if (graph_count >= max_graph_tasks_per_query) {
          atomicAdd(&counters->overflow_count, 1);
          return;
        }
        const int task_slot =
          append_graph_node_task(query_id, slot, graph_tasks, max_graph_tasks, counters);
        if (task_slot >= 0) {
          graph_task_indices_by_query[query_id * max_graph_tasks_per_query + graph_count] =
            task_slot;
          ++graph_count;
        }
        return;
      }
    }
    auto range = device_range_from_node_id(node_id, layout);
    if (range.valid()) { add_exact(range.vec_l, range.vec_r); }
  };

  std::int64_t left = device_max_i64(0, query_ranges[static_cast<std::int64_t>(query_id) * 2]);
  std::int64_t right =
    device_min_i64(layout.rows - 1, query_ranges[static_cast<std::int64_t>(query_id) * 2 + 1]);
  if (right < left) {
    exact_task_counts[query_id] = 0;
    graph_task_counts[query_id] = 0;
    return;
  }

  const std::int64_t b_l = left / layout.leaf_size;
  const std::int64_t b_r = right / layout.leaf_size;
  if (b_l == b_r) {
    add_exact(left, right);
    exact_task_counts[query_id] = exact_count;
    graph_task_counts[query_id] = graph_count;
    return;
  }

  std::int64_t full_l                 = b_l;
  std::int64_t full_r                 = b_r;
  const std::int64_t left_block_start = b_l * static_cast<std::int64_t>(layout.leaf_size);
  const std::int64_t left_block_end =
    device_min_i64(layout.rows - 1, (b_l + 1) * static_cast<std::int64_t>(layout.leaf_size) - 1);
  const std::int64_t right_block_start = b_r * static_cast<std::int64_t>(layout.leaf_size);
  const std::int64_t right_block_end =
    device_min_i64(layout.rows - 1, (b_r + 1) * static_cast<std::int64_t>(layout.leaf_size) - 1);

  if (left > left_block_start) {
    add_exact(left, left_block_end);
    full_l = b_l + 1;
  }
  if (right < right_block_end) {
    add_exact(right_block_start, right);
    full_r = b_r - 1;
  }
  if (full_l > full_r) {
    exact_task_counts[query_id] = exact_count;
    graph_task_counts[query_id] = graph_count;
    return;
  }

  std::int64_t lo = layout.leaf_base + full_l;
  std::int64_t hi = layout.leaf_base + full_r;
  while (lo <= hi) {
    if (lo % 2 == 1) {
      use_node(lo);
      ++lo;
    }
    if (hi % 2 == 0) {
      use_node(hi);
      --hi;
    }
    lo /= 2;
    hi /= 2;
  }
  exact_task_counts[query_id] = exact_count;
  graph_task_counts[query_id] = graph_count;
}

__device__ inline bool better_pair(float lhs_dist,
                                   std::uint32_t lhs_id,
                                   float rhs_dist,
                                   std::uint32_t rhs_id)
{
  return lhs_dist < rhs_dist || (lhs_dist == rhs_dist && lhs_id < rhs_id);
}

__device__ inline void insert_topk(
  float dist, std::uint32_t id, float* best_dist, std::uint32_t* best_id, int topk)
{
  if (!better_pair(dist, id, best_dist[topk - 1], best_id[topk - 1])) { return; }
  int pos = topk - 1;
  while (pos > 0 && better_pair(dist, id, best_dist[pos - 1], best_id[pos - 1])) {
    best_dist[pos] = best_dist[pos - 1];
    best_id[pos]   = best_id[pos - 1];
    --pos;
  }
  best_dist[pos] = dist;
  best_id[pos]   = id;
}

RAFT_KERNEL exact_task_l2_topk_kernel(GlobalDatasetView dataset,
                                      const float* __restrict__ queries,
                                      const ExactSearchTask* __restrict__ tasks,
                                      int task_count,
                                      int topk,
                                      std::uint32_t* __restrict__ out_ids,
                                      float* __restrict__ out_distances)
{
  const int task_id = static_cast<int>(blockIdx.x);
  if (task_id >= task_count) { return; }

  const auto task = tasks[task_id];
  extern __shared__ unsigned char shared_raw[];
  float* shared_dist        = reinterpret_cast<float*>(shared_raw);
  std::uint32_t* shared_ids = reinterpret_cast<std::uint32_t*>(shared_dist + blockDim.x * topk);
  float* shared_query       = reinterpret_cast<float*>(shared_ids + blockDim.x * topk);

  const float* query = queries + static_cast<std::int64_t>(task.query_id) * dataset.dim;
  for (int d = threadIdx.x; d < dataset.dim; d += blockDim.x) {
    shared_query[d] = query[d];
  }
  __syncthreads();

  float local_dist[kRangeCagraMaxTopK];
  std::uint32_t local_id[kRangeCagraMaxTopK];
  for (int k = 0; k < topk; ++k) {
    local_dist[k] = INFINITY;
    local_id[k]   = std::numeric_limits<std::uint32_t>::max();
  }

  for (std::int64_t id = task.left + threadIdx.x; id <= task.right; id += blockDim.x) {
    float dist         = 0.0f;
    const float* point = dataset.row(id);
    for (int d = 0; d < dataset.dim; ++d) {
      const float diff = point[d] - shared_query[d];
      dist += diff * diff;
    }
    insert_topk(dist, static_cast<std::uint32_t>(id), local_dist, local_id, topk);
  }

  const int offset = static_cast<int>(threadIdx.x) * topk;
  for (int k = 0; k < topk; ++k) {
    shared_dist[offset + k] = local_dist[k];
    shared_ids[offset + k]  = local_id[k];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float final_dist[kRangeCagraMaxTopK];
    std::uint32_t final_id[kRangeCagraMaxTopK];
    for (int k = 0; k < topk; ++k) {
      final_dist[k] = INFINITY;
      final_id[k]   = std::numeric_limits<std::uint32_t>::max();
    }
    for (int t = 0; t < blockDim.x; ++t) {
      const int base = t * topk;
      for (int k = 0; k < topk; ++k) {
        const auto id = shared_ids[base + k];
        if (id != std::numeric_limits<std::uint32_t>::max()) {
          insert_topk(shared_dist[base + k], id, final_dist, final_id, topk);
        }
      }
    }
    for (int k = 0; k < topk; ++k) {
      out_ids[static_cast<std::int64_t>(task_id) * topk + k]       = final_id[k];
      out_distances[static_cast<std::int64_t>(task_id) * topk + k] = final_dist[k];
    }
  }
}

__device__ inline void copy_query_to_distance_workspace(const float* __restrict__ query,
                                                        int dim,
                                                        float* __restrict__ shared_query)
{
  constexpr int dataset_block_dim = kRangeCagraDistanceDatasetBlockDim;
  constexpr int team_size         = kRangeCagraDistanceTeamSize;
  using load_t                    = cuvs::neighbors::cagra::detail::device::LOAD_128BIT_T;
  constexpr int vlen            = cuvs::neighbors::cagra::detail::device::get_vlen<load_t, float>();
  const int query_workspace_len = range_cagra_distance_query_workspace_len(dim);
  for (int i = threadIdx.x; i < query_workspace_len; i += blockDim.x) {
    const int swizzled =
      cuvs::neighbors::cagra::detail::device::swizzling<dataset_block_dim, vlen * team_size>(i);
    shared_query[swizzled] = i < dim ? query[i] : 0.0f;
  }
}

__device__ inline float load_distance_query_value(std::uint32_t query_smem_ptr, int dim)
{
  constexpr int dataset_block_dim = kRangeCagraDistanceDatasetBlockDim;
  constexpr int team_size         = kRangeCagraDistanceTeamSize;
  using load_t                    = cuvs::neighbors::cagra::detail::device::LOAD_128BIT_T;
  constexpr int vlen = cuvs::neighbors::cagra::detail::device::get_vlen<load_t, float>();
  const auto swizzled =
    cuvs::neighbors::cagra::detail::device::swizzling<dataset_block_dim, vlen * team_size>(dim);
  float value = 0.0f;
  cuvs::neighbors::cagra::detail::device::lds(
    value, query_smem_ptr + static_cast<std::uint32_t>(sizeof(float) * swizzled));
  return value;
}

template <bool AssumeVectorized>
__device__ inline float team_l2_distance(GlobalDatasetView dataset,
                                         const float* __restrict__ shared_query,
                                         std::int64_t global_id,
                                         bool valid)
{
  constexpr int team_size         = kRangeCagraDistanceTeamSize;
  constexpr int dataset_block_dim = kRangeCagraDistanceDatasetBlockDim;
  using load_t                    = cuvs::neighbors::cagra::detail::device::LOAD_128BIT_T;
  constexpr int vlen      = cuvs::neighbors::cagra::detail::device::get_vlen<load_t, float>();
  constexpr int reg_nelem = (dataset_block_dim + (team_size * vlen) - 1) / (team_size * vlen);
  static_assert((team_size & (team_size - 1)) == 0, "team size must be a power of two");
  static_assert(cuvs::neighbors::cagra::detail::device::warp_size % team_size == 0,
                "team size must divide the warp size");
  static_assert(dataset_block_dim % (team_size * vlen) == 0,
                "dataset block dim must be divisible by the vectorized team width");

  const int lane            = threadIdx.x & (team_size - 1);
  float sum                 = 0.0f;
  const auto query_smem_ptr = static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_query));
  if (valid) {
    const float* point = dataset.row(global_id);
    if constexpr (AssumeVectorized) {
      for (int elem_offset = lane * vlen; elem_offset < dataset.dim;
           elem_offset += dataset_block_dim) {
        alignas(16) float data[reg_nelem][vlen];
#pragma unroll
        for (int e = 0; e < reg_nelem; ++e) {
          const int dim = e * (team_size * vlen) + elem_offset;
          if (dim >= dataset.dim) { break; }
          cuvs::neighbors::cagra::detail::device::ldg_cg(
            reinterpret_cast<load_t&>(data[e]), reinterpret_cast<const load_t*>(point + dim));
        }
#pragma unroll
        for (int e = 0; e < reg_nelem; ++e) {
          const int dim = e * (team_size * vlen) + elem_offset;
          if (dim >= dataset.dim) { break; }
#pragma unroll
          for (int v = 0; v < vlen; ++v) {
            const float query_value = load_distance_query_value(query_smem_ptr, dim + v);
            const float diff        = data[e][v] - query_value;
            sum += diff * diff;
          }
        }
      }
    } else {
      const bool vectorizable =
        ((dataset.dim & (vlen - 1)) == 0) && ((dataset.stride & (vlen - 1)) == 0) &&
        ((reinterpret_cast<std::uintptr_t>(point) & (sizeof(load_t) - 1)) == 0);
      if (vectorizable) {
        for (int elem_offset = lane * vlen; elem_offset < dataset.dim;
             elem_offset += dataset_block_dim) {
          alignas(16) float data[reg_nelem][vlen];
#pragma unroll
          for (int e = 0; e < reg_nelem; ++e) {
            const int dim = e * (team_size * vlen) + elem_offset;
            if (dim >= dataset.dim) { break; }
            cuvs::neighbors::cagra::detail::device::ldg_cg(
              reinterpret_cast<load_t&>(data[e]), reinterpret_cast<const load_t*>(point + dim));
          }
#pragma unroll
          for (int e = 0; e < reg_nelem; ++e) {
            const int dim = e * (team_size * vlen) + elem_offset;
            if (dim >= dataset.dim) { break; }
#pragma unroll
            for (int v = 0; v < vlen; ++v) {
              const float query_value = load_distance_query_value(query_smem_ptr, dim + v);
              const float diff        = data[e][v] - query_value;
              sum += diff * diff;
            }
          }
        }
      } else {
        for (int d = lane; d < dataset.dim; d += team_size) {
          const float query_value = load_distance_query_value(query_smem_ptr, d);
          const float diff        = point[d] - query_value;
          sum += diff * diff;
        }
      }
    }
  }

  return cuvs::neighbors::cagra::detail::device::team_sum<team_size>(sum);
}

__device__ inline std::uint32_t graph_task_hash(std::uint32_t value)
{
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ inline bool visited_hash_insert(std::uint32_t* __restrict__ table,
                                           int hash_bitlen,
                                           std::uint32_t key)
{
  return cuvs::neighbors::cagra::detail::hashmap::insert<std::uint32_t, 0>(
           table, static_cast<std::uint32_t>(hash_bitlen), key) != 0;
}

__device__ inline void visited_hash_init(std::uint32_t* __restrict__ table, int hash_bitlen)
{
  cuvs::neighbors::cagra::detail::hashmap::init<std::uint32_t>(
    table, static_cast<std::uint32_t>(hash_bitlen));
}

__device__ inline void restore_visited_hash_from_itopk(std::uint32_t* __restrict__ table,
                                                       const std::uint32_t* __restrict__ itopk_ids,
                                                       int itopk_size,
                                                       int hash_bitlen)
{
  for (int i = threadIdx.x; i < itopk_size; i += blockDim.x) {
    const auto id = itopk_ids[i];
    if (id != kRangeCagraInvalidId) {
      visited_hash_insert(table, hash_bitlen, id & kRangeCagraLocalIdMask);
    }
  }
}

__device__ inline void insert_graph_candidate(float dist,
                                              std::uint32_t local_id,
                                              float* __restrict__ itopk_dist,
                                              std::uint32_t* __restrict__ itopk_ids,
                                              int itopk_size)
{
  if (local_id == kRangeCagraInvalidId || !(dist < INFINITY)) { return; }
  local_id &= kRangeCagraLocalIdMask;

  for (int i = 0; i < itopk_size; ++i) {
    const auto existing = itopk_ids[i];
    if (existing == kRangeCagraInvalidId) { continue; }
    if ((existing & kRangeCagraLocalIdMask) == local_id) {
      if (dist < itopk_dist[i]) { itopk_dist[i] = dist; }
      return;
    }
  }

  if (!better_pair(dist, local_id, itopk_dist[itopk_size - 1], itopk_ids[itopk_size - 1])) {
    return;
  }

  int pos = itopk_size - 1;
  while (
    pos > 0 &&
    better_pair(dist, local_id, itopk_dist[pos - 1], itopk_ids[pos - 1] & kRangeCagraLocalIdMask)) {
    itopk_dist[pos] = itopk_dist[pos - 1];
    itopk_ids[pos]  = itopk_ids[pos - 1];
    --pos;
  }
  itopk_dist[pos] = dist;
  itopk_ids[pos]  = local_id;
}

__device__ inline void pick_next_graph_parents(std::uint32_t* __restrict__ terminate_flag,
                                               std::uint32_t* __restrict__ parent_slots,
                                               std::uint32_t* __restrict__ itopk_ids,
                                               int itopk_size,
                                               int search_width)
{
  if (threadIdx.x < search_width) { parent_slots[threadIdx.x] = kRangeCagraInvalidId; }
  constexpr int warp_size = 32;
  if (threadIdx.x >= warp_size) { return; }

  const int lane                = threadIdx.x;
  std::uint32_t num_new_parents = 0;
  const int rounded_itopk       = ((itopk_size + warp_size - 1) / warp_size) * warp_size;

  for (int j = lane; j < rounded_itopk; j += warp_size) {
    int new_parent = 0;
    if (j < itopk_size) {
      const auto id = itopk_ids[j];
      new_parent    = (id != kRangeCagraInvalidId && (id & kRangeCagraExpandedBit) == 0) ? 1 : 0;
    }

    const std::uint32_t ballot_mask = __ballot_sync(0xffffffffu, new_parent);
    if (new_parent) {
      const auto dst = __popc(ballot_mask & ((std::uint32_t{1} << lane) - 1)) + num_new_parents;
      if (dst < static_cast<std::uint32_t>(search_width)) {
        parent_slots[dst] = static_cast<std::uint32_t>(j);
        itopk_ids[j] |= kRangeCagraExpandedBit;
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= static_cast<std::uint32_t>(search_width)) { break; }
  }

  if (lane == 0 && num_new_parents == 0) { *terminate_flag = 1; }
}

__device__ inline int device_cagra_bitonic_max_candidates(int candidate_count)
{
  if (candidate_count <= 64) { return 64; }
  if (candidate_count <= 128) { return 128; }
  if (candidate_count <= 256) { return 256; }
  return 0;
}

__device__ inline int device_cagra_bitonic_max_itopk(int itopk_size)
{
  if (itopk_size <= 64) { return 64; }
  if (itopk_size <= 128) { return 128; }
  if (itopk_size <= 256) { return 256; }
  if (itopk_size <= 512) { return 512; }
  return 0;
}

__device__ inline int device_range_cagra_radix_workspace_words(int max_itopk)
{
  return max_itopk * 2 + 2048 + 8;
}

template <bool UseRadixTopk>
__device__ inline void merge_graph_candidates_into_itopk(float* __restrict__ itopk_dist,
                                                         std::uint32_t* __restrict__ itopk_ids,
                                                         int itopk_size,
                                                         float* __restrict__ candidate_dist,
                                                         std::uint32_t* __restrict__ candidate_ids,
                                                         int candidate_count,
                                                         int result_buffer_size,
                                                         std::uint32_t* __restrict__ topk_ws,
                                                         std::uint32_t* __restrict__ radix_smem,
                                                         bool first_bitonic_merge)
{
  if constexpr (UseRadixTopk) {
    const int max_itopk = device_cagra_bitonic_max_itopk(itopk_size);
    cuvs::neighbors::cagra::detail::single_cta_search::topk_by_radix_sort<std::uint32_t>{}(
      static_cast<std::uint32_t>(max_itopk),
      static_cast<std::uint32_t>(itopk_size),
      static_cast<std::uint32_t>(result_buffer_size),
      reinterpret_cast<std::uint32_t*>(itopk_dist),
      itopk_ids,
      reinterpret_cast<std::uint32_t*>(itopk_dist),
      itopk_ids,
      nullptr,
      topk_ws,
      true,
      radix_smem);
    return;
  } else {
    const int max_candidates = device_cagra_bitonic_max_candidates(candidate_count);
    const int max_itopk      = device_cagra_bitonic_max_itopk(itopk_size);
    if (max_candidates > 0 && max_itopk > 0) {
      if (max_itopk <= 256) {
        cuvs::neighbors::cagra::detail::single_cta_search::
          topk_by_bitonic_sort_and_merge<false, std::uint32_t>(
            itopk_dist,
            itopk_ids,
            static_cast<std::uint32_t>(max_itopk),
            static_cast<std::uint32_t>(itopk_size),
            candidate_dist,
            candidate_ids,
            static_cast<std::uint32_t>(max_candidates),
            static_cast<std::uint32_t>(candidate_count),
            topk_ws,
            first_bitonic_merge);
      } else {
        cuvs::neighbors::cagra::detail::single_cta_search::
          topk_by_bitonic_sort_and_merge<true, std::uint32_t>(
            itopk_dist,
            itopk_ids,
            static_cast<std::uint32_t>(max_itopk),
            static_cast<std::uint32_t>(itopk_size),
            candidate_dist,
            candidate_ids,
            static_cast<std::uint32_t>(max_candidates),
            static_cast<std::uint32_t>(candidate_count),
            topk_ws,
            first_bitonic_merge);
      }
      return;
    }

    if (threadIdx.x == 0) {
      for (int i = 0; i < candidate_count; ++i) {
        insert_graph_candidate(
          candidate_dist[i], candidate_ids[i], itopk_dist, itopk_ids, itopk_size);
      }
    }
  }
}

template <bool UseRadixTopk, bool AssumeVectorizedDistance>
RAFT_KERNEL range_graph_task_search_kernel(GlobalDatasetView dataset,
                                           const float* __restrict__ queries,
                                           const GraphSearchTask* __restrict__ tasks,
                                           int task_count,
                                           DeviceRangeGraphPoolView graph_pool,
                                           int topk,
                                           int itopk_size,
                                           int search_width,
                                           int entry_count,
                                           int max_iterations,
                                           const int* __restrict__ graph_iterations_by_graph,
                                           int hash_reset_interval,
                                           int hash_bitlen,
                                           std::uint64_t seed,
                                           GraphSearchProfileCounters* __restrict__ profile,
                                           std::uint32_t* __restrict__ out_ids,
                                           float* __restrict__ out_distances)
{
  const int task_id = static_cast<int>(blockIdx.x);
  if (task_id >= task_count) { return; }

  const auto task    = tasks[task_id];
  const int graph_id = task.graph_id;
  if (graph_id < 0 || graph_id >= graph_pool.graph_count) { return; }

  const int rows                   = graph_pool.rows[graph_id];
  const int degree                 = graph_pool.degrees[graph_id];
  const auto rank_l                = graph_pool.rank_l[graph_id];
  const auto edge_of               = graph_pool.offsets[graph_id];
  const auto* edges                = graph_pool.edges + edge_of;
  const int task_max_iterations =
    graph_iterations_by_graph != nullptr ? max(0, graph_iterations_by_graph[graph_id])
                                         : max_iterations;
  const int initial_count          = min(rows, min(itopk_size, max(entry_count, topk)));
  const int per_iter_candidate_cnt = search_width * degree;
  const int max_merge_candidates   = max(initial_count, per_iter_candidate_cnt);
  const int result_buffer_size     = itopk_size + max_merge_candidates;
  const int max_itopk_for_topk     = device_cagra_bitonic_max_itopk(itopk_size);
  const int radix_smem_words =
    UseRadixTopk ? device_range_cagra_radix_workspace_words(max_itopk_for_topk) : 0;

  unsigned long long profile_total_start = 0;
  unsigned long long profile_stage_start = 0;
  if (profile != nullptr && threadIdx.x == 0) {
    profile_total_start = clock64();
    profile_stage_start = profile_total_start;
  }

  extern __shared__ unsigned char shared_raw[];
  float* shared_query           = reinterpret_cast<float*>(shared_raw);
  const int query_workspace_len = range_cagra_distance_query_workspace_len(dataset.dim);
  float* result_dist            = shared_query + query_workspace_len;
  auto* result_ids     = reinterpret_cast<std::uint32_t*>(result_dist + result_buffer_size);
  float* itopk_dist    = result_dist;
  auto* itopk_ids      = result_ids;
  float* child_dist    = result_dist + itopk_size;
  auto* child_ids      = result_ids + itopk_size;
  auto* parent_slots   = result_ids + result_buffer_size;
  auto* topk_ws        = parent_slots + search_width;
  auto* radix_smem     = topk_ws + kRangeCagraTopkWorkspaceWords;
  auto* visited_hash   = radix_smem + radix_smem_words;
  auto* terminate_flag = visited_hash + (1 << hash_bitlen);

  const float* query = queries + static_cast<std::int64_t>(task.query_id) * dataset.dim;
  copy_query_to_distance_workspace(query, dataset.dim, shared_query);
  for (int i = threadIdx.x; i < itopk_size; i += blockDim.x) {
    itopk_dist[i] = INFINITY;
    itopk_ids[i]  = kRangeCagraInvalidId;
  }
  visited_hash_init(visited_hash, hash_bitlen);
  if (threadIdx.x < kRangeCagraTopkWorkspaceWords) { topk_ws[threadIdx.x] = 0; }
  if (threadIdx.x == 0) {
    topk_ws[0]      = ~0u;
    *terminate_flag = 0;
  }
  __syncthreads();
  if (profile != nullptr && threadIdx.x == 0) {
    atomicAdd(&profile->graph_tasks, 1ULL);
    const auto now = clock64();
    atomicAdd(&profile->cycles_query_init, now - profile_stage_start);
    profile_stage_start = now;
  }

  const int team_id       = threadIdx.x / kRangeCagraDistanceTeamSize;
  const int team_lane     = threadIdx.x & (kRangeCagraDistanceTeamSize - 1);
  const int teams_per_cta = blockDim.x / kRangeCagraDistanceTeamSize;

  for (int i = threadIdx.x; i < initial_count; i += blockDim.x) {
    std::uint32_t local_id = kRangeCagraInvalidId;
    if (rows > 0 && initial_count >= rows) {
      local_id = static_cast<std::uint32_t>(i);
    } else if (rows > 0) {
      const auto hash = graph_task_hash(
        static_cast<std::uint32_t>(i ^ (graph_id * 0x9e3779b9u) ^ (task.query_id * 0x85ebca6bu) ^
                                   static_cast<std::uint32_t>(seed)));
      const std::uint32_t offset = hash % static_cast<std::uint32_t>(rows);
      const std::uint32_t stride =
        rows > 1 ? (graph_task_hash(hash ^ 0x27d4eb2du) % static_cast<std::uint32_t>(rows - 1)) + 1
                 : 1;
      local_id =
        (offset + static_cast<std::uint32_t>(i) * stride) % static_cast<std::uint32_t>(rows);
    }
    bool inserted = false;
    if (local_id != kRangeCagraInvalidId) {
      if (profile != nullptr) { atomicAdd(&profile->initial_candidates, 1ULL); }
      inserted = visited_hash_insert(visited_hash, hash_bitlen, local_id);
      if (profile != nullptr && inserted) { atomicAdd(&profile->distance_evaluations, 1ULL); }
    }
    child_ids[i] = inserted ? local_id : kRangeCagraInvalidId;
  }
  __syncthreads();
  if (profile != nullptr && threadIdx.x == 0) {
    const auto now = clock64();
    atomicAdd(&profile->cycles_initial_prepare, now - profile_stage_start);
    profile_stage_start = now;
  }

  for (int i = team_id; i < initial_count; i += teams_per_cta) {
    const auto local_id = child_ids[i];
    const bool inserted = local_id != kRangeCagraInvalidId;
    const float dist    = team_l2_distance<AssumeVectorizedDistance>(
      dataset, shared_query, rank_l + local_id, rows > 0 && inserted);
    if (team_lane == 0) { child_dist[i] = inserted ? dist : INFINITY; }
  }
  __syncthreads();
  if (profile != nullptr && threadIdx.x == 0) {
    const auto now = clock64();
    atomicAdd(&profile->cycles_initial_distance, now - profile_stage_start);
    profile_stage_start = now;
  }

  merge_graph_candidates_into_itopk<UseRadixTopk>(itopk_dist,
                                                  itopk_ids,
                                                  itopk_size,
                                                  child_dist,
                                                  child_ids,
                                                  initial_count,
                                                  itopk_size + initial_count,
                                                  topk_ws,
                                                  radix_smem,
                                                  true);
  __syncthreads();
  if (profile != nullptr && threadIdx.x == 0) {
    const auto now = clock64();
    atomicAdd(&profile->cycles_initial_merge, now - profile_stage_start);
    profile_stage_start = now;
  }

  for (int iter = 0; iter < task_max_iterations; ++iter) {
    if (profile != nullptr && threadIdx.x == 0) {
      atomicAdd(&profile->iterations_started, 1ULL);
      profile_stage_start = clock64();
    }
    if (iter > 0 && hash_reset_interval > 0 && (iter % hash_reset_interval) == 0) {
      visited_hash_init(visited_hash, hash_bitlen);
      __syncthreads();
      restore_visited_hash_from_itopk(visited_hash, itopk_ids, itopk_size, hash_bitlen);
      __syncthreads();
      if (profile != nullptr && threadIdx.x == 0) {
        atomicAdd(&profile->hash_reset_count, 1ULL);
        const auto now = clock64();
        atomicAdd(&profile->cycles_hash_reset, now - profile_stage_start);
        profile_stage_start = now;
      }
    }

    if (threadIdx.x == 0) { *terminate_flag = 0; }
    __syncthreads();

    if (threadIdx.x < 32) {
      if constexpr (UseRadixTopk) {
        cuvs::neighbors::cagra::detail::single_cta_search::pickup_next_parents<false,
                                                                               std::uint32_t>(
          terminate_flag, parent_slots, itopk_ids, itopk_size, search_width);
      } else {
        cuvs::neighbors::cagra::detail::single_cta_search::pickup_next_parents<true, std::uint32_t>(
          terminate_flag, parent_slots, itopk_ids, itopk_size, search_width);
      }
    }
    __syncthreads();
    if (profile != nullptr && threadIdx.x == 0) {
      const auto now = clock64();
      atomicAdd(&profile->cycles_pickup, now - profile_stage_start);
      profile_stage_start = now;
    }
    if (*terminate_flag != 0) {
      if (profile != nullptr && threadIdx.x == 0) {
        atomicAdd(&profile->terminated_iterations, 1ULL);
      }
      break;
    }

    const int candidate_count = search_width * degree;
    if (profile != nullptr && threadIdx.x == 0) {
      atomicAdd(&profile->candidate_slots, static_cast<unsigned long long>(candidate_count));
    }

    for (int i = threadIdx.x; i < candidate_count; i += blockDim.x) {
      const int parent_id = i / degree;
      const int edge_id   = i - parent_id * degree;
      const auto slot     = parent_slots[parent_id];
      std::uint32_t child = kRangeCagraInvalidId;
      bool valid_child    = false;
      if (slot != kRangeCagraInvalidId) {
        const auto parent_local = itopk_ids[slot] & kRangeCagraLocalIdMask;
        child                   = edges[static_cast<std::int64_t>(parent_local) * degree + edge_id];
        valid_child             = child < static_cast<std::uint32_t>(rows);
      }
      bool inserted = false;
      if (valid_child) {
        if (profile != nullptr) { atomicAdd(&profile->valid_child_candidates, 1ULL); }
        inserted = visited_hash_insert(visited_hash, hash_bitlen, child);
        if (profile != nullptr) {
          if (inserted) {
            atomicAdd(&profile->inserted_child_candidates, 1ULL);
            atomicAdd(&profile->distance_evaluations, 1ULL);
          } else {
            atomicAdd(&profile->duplicate_child_candidates, 1ULL);
          }
        }
        if (!inserted) {
          child       = kRangeCagraInvalidId;
          valid_child = false;
        }
      }
      child_ids[i] = valid_child ? child : kRangeCagraInvalidId;
    }
    __syncthreads();
    if (profile != nullptr && threadIdx.x == 0) {
      const auto now = clock64();
      atomicAdd(&profile->cycles_expand_prepare, now - profile_stage_start);
      profile_stage_start = now;
    }

    for (int i = team_id; i < candidate_count; i += teams_per_cta) {
      const auto child       = child_ids[i];
      const bool valid_child = child != kRangeCagraInvalidId;
      const float dist       = team_l2_distance<AssumeVectorizedDistance>(
        dataset, shared_query, rank_l + static_cast<std::int64_t>(child), valid_child);
      if (team_lane == 0) { child_dist[i] = valid_child ? dist : INFINITY; }
    }
    __syncthreads();
    if (profile != nullptr && threadIdx.x == 0) {
      const auto now = clock64();
      atomicAdd(&profile->cycles_expand_distance, now - profile_stage_start);
      profile_stage_start = now;
    }

    merge_graph_candidates_into_itopk<UseRadixTopk>(itopk_dist,
                                                    itopk_ids,
                                                    itopk_size,
                                                    child_dist,
                                                    child_ids,
                                                    candidate_count,
                                                    itopk_size + candidate_count,
                                                    topk_ws,
                                                    radix_smem,
                                                    false);
    __syncthreads();
    if (profile != nullptr && threadIdx.x == 0) {
      const auto now = clock64();
      atomicAdd(&profile->cycles_iter_merge, now - profile_stage_start);
      atomicAdd(&profile->iterations_completed, 1ULL);
      profile_stage_start = now;
    }
  }

  if (profile != nullptr && threadIdx.x == 0) { profile_stage_start = clock64(); }
  if (threadIdx.x == 0) {
    for (int k = 0; k < topk; ++k) {
      const auto slot       = UseRadixTopk ? static_cast<std::uint32_t>(k)
                                           : cuvs::neighbors::cagra::detail::device::swizzling(k);
      const auto local_id   = itopk_ids[slot];
      const auto out_offset = static_cast<std::int64_t>(task_id) * topk + k;
      if (local_id == kRangeCagraInvalidId) {
        out_ids[out_offset]       = kRangeCagraInvalidId;
        out_distances[out_offset] = INFINITY;
      } else {
        out_ids[out_offset] = static_cast<std::uint32_t>(
          rank_l + static_cast<std::int64_t>(local_id & kRangeCagraLocalIdMask));
        out_distances[out_offset] = itopk_dist[slot];
      }
    }
  }
  if (profile != nullptr && threadIdx.x == 0) {
    const auto now = clock64();
    atomicAdd(&profile->cycles_output, now - profile_stage_start);
    atomicAdd(&profile->cycles_total, now - profile_total_start);
  }
}

RAFT_KERNEL merge_task_topk_kernel(const ExactSearchTask* __restrict__ exact_tasks,
                                   const int* __restrict__ exact_task_indices_by_query,
                                   const int* __restrict__ exact_task_counts,
                                   int max_exact_tasks_per_query,
                                   const std::uint32_t* __restrict__ exact_ids,
                                   const float* __restrict__ exact_distances,
                                   const GraphSearchTask* __restrict__ graph_tasks,
                                   const int* __restrict__ graph_task_indices_by_query,
                                   const int* __restrict__ graph_task_counts,
                                   int max_graph_tasks_per_query,
                                   const std::uint32_t* __restrict__ graph_ids,
                                   const float* __restrict__ graph_distances,
                                   int topk,
                                   std::uint32_t* __restrict__ final_ids,
                                   float* __restrict__ final_distances)
{
  const int query_id = static_cast<int>(blockIdx.x);
  extern __shared__ unsigned char shared_raw[];
  float* shared_dist        = reinterpret_cast<float*>(shared_raw);
  std::uint32_t* shared_ids = reinterpret_cast<std::uint32_t*>(shared_dist + blockDim.x * topk);

  float local_dist[kRangeCagraMaxTopK];
  std::uint32_t local_id[kRangeCagraMaxTopK];
  for (int k = 0; k < topk; ++k) {
    local_dist[k] = INFINITY;
    local_id[k]   = std::numeric_limits<std::uint32_t>::max();
  }

  const int exact_count = exact_task_counts[query_id];
  for (int local = threadIdx.x; local < exact_count; local += blockDim.x) {
    const int t = exact_task_indices_by_query[query_id * max_exact_tasks_per_query + local];
    if (exact_tasks[t].query_id != query_id) { continue; }
    for (int k = 0; k < topk; ++k) {
      insert_topk(exact_distances[static_cast<std::int64_t>(t) * topk + k],
                  exact_ids[static_cast<std::int64_t>(t) * topk + k],
                  local_dist,
                  local_id,
                  topk);
    }
  }
  const int graph_count = graph_task_counts[query_id];
  for (int local = threadIdx.x; local < graph_count; local += blockDim.x) {
    const int t = graph_task_indices_by_query[query_id * max_graph_tasks_per_query + local];
    if (graph_tasks[t].query_id != query_id) { continue; }
    for (int k = 0; k < topk; ++k) {
      insert_topk(graph_distances[static_cast<std::int64_t>(t) * topk + k],
                  graph_ids[static_cast<std::int64_t>(t) * topk + k],
                  local_dist,
                  local_id,
                  topk);
    }
  }

  const int offset = static_cast<int>(threadIdx.x) * topk;
  for (int k = 0; k < topk; ++k) {
    shared_dist[offset + k] = local_dist[k];
    shared_ids[offset + k]  = local_id[k];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float final_dist[kRangeCagraMaxTopK];
    std::uint32_t final_id[kRangeCagraMaxTopK];
    for (int k = 0; k < topk; ++k) {
      final_dist[k] = INFINITY;
      final_id[k]   = std::numeric_limits<std::uint32_t>::max();
    }
    for (int t = 0; t < blockDim.x; ++t) {
      const int base = t * topk;
      for (int k = 0; k < topk; ++k) {
        insert_topk(shared_dist[base + k], shared_ids[base + k], final_dist, final_id, topk);
      }
    }
    for (int k = 0; k < topk; ++k) {
      final_ids[static_cast<std::int64_t>(query_id) * topk + k]       = final_id[k];
      final_distances[static_cast<std::int64_t>(query_id) * topk + k] = final_dist[k];
    }
  }
}

}  // namespace kernels

inline void build_range_graph_pool_on_gpu(raft::resources const& res,
                                          GlobalDatasetView const& dataset,
                                          std::vector<HostRangeGraphMeta>& graph_metas,
                                          RangeGraphBuildParams const& params,
                                          DeviceRangeGraphPool& graph_pool)
{
  graph_pool.release();
  if (graph_metas.empty()) { return; }

  std::int64_t edge_count = 0;
  std::int64_t row_count  = 0;
  std::vector<std::int64_t> h_offsets(graph_metas.size());
  std::vector<std::int64_t> h_rank_l(graph_metas.size());
  std::vector<int> h_rows(graph_metas.size());
  std::vector<int> h_degrees(graph_metas.size());
  std::vector<int> h_intermediate_degrees(graph_metas.size());
  std::vector<std::int64_t> h_row_offsets(graph_metas.size() + 1);
  for (std::size_t i = 0; i < graph_metas.size(); ++i) {
    const auto rows                = static_cast<int>(graph_metas[i].range.size());
    const auto degree              = graph_metas[i].degree;
    const auto intermediate_degree = graph_metas[i].intermediate_degree;
    RAFT_EXPECTS(rows > 1, "range graph metadata must cover at least two vectors");
    RAFT_EXPECTS(degree > 0, "range graph degree must be positive");
    RAFT_EXPECTS(degree <= rows - 1, "range graph degree must fit the graph row count");
    RAFT_EXPECTS(intermediate_degree >= degree,
                 "intermediate graph degree must be at least final graph degree");
    RAFT_EXPECTS(intermediate_degree <= rows - 1,
                 "intermediate graph degree must fit the graph row count");
    h_row_offsets[i]           = row_count;
    graph_metas[i].edge_offset = edge_count;
    h_offsets[i]               = graph_metas[i].edge_offset;
    h_rank_l[i]                = graph_metas[i].range.vec_l;
    h_rows[i]                  = rows;
    h_degrees[i]               = degree;
    h_intermediate_degrees[i]  = intermediate_degree;
    row_count += rows;
    edge_count += static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(degree);
  }
  h_row_offsets[graph_metas.size()] = row_count;

  auto stream            = raft::resource::get_cuda_stream(res);
  graph_pool.graph_count = static_cast<int>(graph_metas.size());
  graph_pool.edge_count  = edge_count;
  RAFT_CUDA_TRY(
    cudaMalloc(&graph_pool.edges, sizeof(std::uint32_t) * static_cast<std::size_t>(edge_count)));
  RAFT_CUDA_TRY(cudaMalloc(
    &graph_pool.offsets, sizeof(std::int64_t) * static_cast<std::size_t>(graph_pool.graph_count)));
  RAFT_CUDA_TRY(cudaMalloc(
    &graph_pool.rank_l, sizeof(std::int64_t) * static_cast<std::size_t>(graph_pool.graph_count)));
  RAFT_CUDA_TRY(
    cudaMalloc(&graph_pool.rows, sizeof(int) * static_cast<std::size_t>(graph_pool.graph_count)));
  RAFT_CUDA_TRY(cudaMalloc(&graph_pool.degrees,
                           sizeof(int) * static_cast<std::size_t>(graph_pool.graph_count)));

  RAFT_CUDA_TRY(cudaMemcpyAsync(graph_pool.offsets,
                                h_offsets.data(),
                                sizeof(std::int64_t) * h_offsets.size(),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(graph_pool.rank_l,
                                h_rank_l.data(),
                                sizeof(std::int64_t) * h_rank_l.size(),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    graph_pool.rows, h_rows.data(), sizeof(int) * h_rows.size(), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(graph_pool.degrees,
                                h_degrees.data(),
                                sizeof(int) * h_degrees.size(),
                                cudaMemcpyHostToDevice,
                                stream));
  raft::resource::sync_stream(res);

  build_range_graph_pool_edges_on_gpu(
    res, dataset, view(graph_pool), h_row_offsets, h_degrees, h_intermediate_degrees, params);
}

inline SegmentTreeRangeCagraIndex build_segment_tree_range_cagra_impl(
  raft::resources const& res,
  GlobalDatasetView const& dataset,
  int leaf_size,
  RangeGraphBuildParams const& params)
{
  RAFT_EXPECTS(dataset.base != nullptr, "GlobalDatasetView::base must not be null");
  RAFT_EXPECTS(dataset.rows > 0, "GlobalDatasetView::rows must be positive");
  RAFT_EXPECTS(dataset.rows <= std::numeric_limits<std::uint32_t>::max(),
               "segment-tree range_cagra currently returns uint32 global ids");

  SegmentTreeRangeCagraIndex out;
  out.layout = make_segment_tree_layout(dataset.rows, dataset.dim, leaf_size);

  out.graph_metas.reserve(static_cast<std::size_t>(out.layout.leaf_blocks));

  const auto build_start      = std::chrono::steady_clock::now();
  const std::int64_t max_node = 2 * out.layout.leaf_base;
  for (std::int64_t node_id = 1; node_id < max_node; ++node_id) {
    auto range = range_from_node_id(node_id, out.layout);
    if (!range.valid() || range.size() <= leaf_size) { continue; }

    const auto degrees = range_graph_degrees_for_node(out.layout, range, params);
    if (degrees.graph_degree <= 0 || degrees.intermediate_graph_degree <= 0) { continue; }

    const int graph_id = static_cast<int>(out.graph_metas.size());
    out.layout.graph_slot[static_cast<std::size_t>(node_id)] = graph_id;
    out.graph_metas.push_back(
      {node_id, range, graph_id, degrees.graph_degree, degrees.intermediate_graph_degree, 0});
  }

  if (!out.graph_metas.empty()) {
    build_range_graph_pool_on_gpu(res, dataset, out.graph_metas, params, out.graph_pool);
  }
  const auto build_stop = std::chrono::steady_clock::now();
  out.build_seconds     = std::chrono::duration<double>(build_stop - build_start).count();

  return out;
}

inline SegmentTreeRangeCagraIndex build_segment_tree_range_cagra(
  raft::resources const& res,
  GlobalDatasetView const& dataset,
  int leaf_size,
  RangeGraphBuildParams const& params)
{
  return build_segment_tree_range_cagra_impl(res, dataset, leaf_size, params);
}

inline void run_segment_tree_range_cagra_search(
  raft::resources const& res,
  GlobalDatasetView const& dataset,
  SegmentTreeRangeCagraIndex const& index,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  std::vector<std::int64_t> const& query_ranges,
  int topk,
  int ef,
  int entry_count,
  int graph_iterations,
  int exact_threads,
  std::vector<std::uint32_t>& final_ids,
  std::vector<float>& final_distances,
  SegmentTreeSearchStats* stats = nullptr,
  int graph_search_concurrency  = 1,
  bool graph_profile_enabled    = false,
  int graph_threads             = 128,
  SegmentTreeSearchSchedule search_schedule = SegmentTreeSearchSchedule::kOverlap,
  SegmentTreeSearchIterationParams search_iteration_params = {})
{
  RAFT_EXPECTS(query_ranges.size() == static_cast<std::size_t>(queries.extent(0)) * 2,
               "query_ranges must be [n_queries, 2]");
  RAFT_EXPECTS(queries.extent(1) == dataset.dim, "query dimension mismatch");
  RAFT_EXPECTS(topk > 0 && topk <= kRangeCagraMaxTopK, "topk must be in [1, 32]");
  RAFT_EXPECTS(ef >= topk, "ef must be >= topk");
  RAFT_EXPECTS(entry_count > 0, "entry_count must be positive");
  RAFT_EXPECTS(graph_iterations >= 0, "graph_iterations must be non-negative");
  RAFT_EXPECTS(exact_threads > 0, "exact_threads must be positive");
  RAFT_EXPECTS(graph_search_concurrency > 0, "graph_search_concurrency must be positive");
  RAFT_EXPECTS(graph_threads >= 32 && graph_threads <= 1024, "graph_threads must be in [32, 1024]");
  RAFT_EXPECTS(graph_threads % kRangeCagraDistanceTeamSize == 0,
               "graph_threads must be divisible by the range-CAGRA distance team size");
  RAFT_EXPECTS(search_iteration_params.lower_layer_count >= 0,
               "lower_layer_count must be non-negative");
  RAFT_EXPECTS(search_iteration_params.lower_layer_iterations >= 0,
               "lower_layer_iterations must be non-negative");
  RAFT_EXPECTS(search_iteration_params.upper_layer_count >= 0,
               "upper_layer_count must be non-negative");
  RAFT_EXPECTS(search_iteration_params.upper_layer_iterations >= 0,
               "upper_layer_iterations must be non-negative");
  RAFT_EXPECTS(search_iteration_params.adaptive_min_iterations >= 0,
               "adaptive_min_iterations must be non-negative");
  RAFT_EXPECTS(search_iteration_params.adaptive_max_iterations >= 0,
               "adaptive_max_iterations must be non-negative");
  RAFT_EXPECTS(search_iteration_params.adaptive_granularity >= 0,
               "adaptive_granularity must be non-negative");
  RAFT_EXPECTS(index.graph_metas.size() == static_cast<std::size_t>(index.graph_pool.graph_count),
               "host graph metadata must match the device graph pool");

  final_ids.assign(static_cast<std::size_t>(queries.extent(0)) * topk,
                   std::numeric_limits<std::uint32_t>::max());
  final_distances.assign(static_cast<std::size_t>(queries.extent(0)) * topk, INFINITY);

  SegmentTreeSearchStats local_stats;
  local_stats.graph_profile_enabled = graph_profile_enabled;
  local_stats.search_iteration_policy                = search_iteration_params.policy;
  local_stats.search_iteration_base_graph_iterations = graph_iterations;
  local_stats.search_iteration_min_graph_iterations  = graph_iterations;
  local_stats.search_iteration_max_graph_iterations  = graph_iterations;
  local_stats.search_iteration_avg_graph_iterations  = static_cast<double>(graph_iterations);
  local_stats.low_layer_search_layers =
    search_iteration_params.policy == SegmentTreeSearchIterationPolicy::kLowerLayers
      ? search_iteration_params.lower_layer_count
      : 0;
  local_stats.low_layer_graph_iterations =
    local_stats.low_layer_search_layers > 0 ? search_iteration_params.lower_layer_iterations
                                            : graph_iterations;
  local_stats.upper_layer_search_layers =
    search_iteration_params.policy == SegmentTreeSearchIterationPolicy::kUpperLayers
      ? search_iteration_params.upper_layer_count
      : 0;
  local_stats.upper_layer_graph_iterations =
    local_stats.upper_layer_search_layers > 0 ? search_iteration_params.upper_layer_iterations
                                              : graph_iterations;
  const auto search_start           = std::chrono::steady_clock::now();
  auto stream                       = raft::resource::get_cuda_stream(res);

  const auto n_queries             = static_cast<int>(queries.extent(0));
  const auto graph_tasks_per_query = max_graph_tasks_per_query(index.layout);
  const auto max_exact_tasks =
    static_cast<int>(static_cast<std::int64_t>(n_queries) * kRangeCagraMaxExactTasksPerQuery);
  const auto max_graph_node_tasks =
    static_cast<int>(static_cast<std::int64_t>(n_queries) * graph_tasks_per_query);

  std::int64_t* d_query_ranges                = nullptr;
  int* d_graph_slot                           = nullptr;
  kernels::DeviceTaskCounters* d_counters     = nullptr;
  ExactSearchTask* d_exact_tasks              = nullptr;
  int* d_exact_task_indices_by_query          = nullptr;
  int* d_exact_task_counts                    = nullptr;
  std::uint32_t* d_exact_ids                  = nullptr;
  float* d_exact_distances                    = nullptr;
  GraphSearchTask* d_graph_tasks              = nullptr;
  int* d_graph_task_indices_by_query          = nullptr;
  int* d_graph_task_counts                    = nullptr;
  std::uint32_t* d_graph_ids                  = nullptr;
  float* d_graph_distances                    = nullptr;
  std::uint32_t* d_final_ids                  = nullptr;
  float* d_final_distances                    = nullptr;
  GraphSearchProfileCounters* d_graph_profile = nullptr;
  int* d_graph_iterations_by_graph            = nullptr;
  const auto final_output_elements            = static_cast<std::size_t>(n_queries) * topk;

  RAFT_CUDA_TRY(
    cudaMalloc(&d_query_ranges, sizeof(std::int64_t) * static_cast<std::size_t>(n_queries) * 2));
  RAFT_CUDA_TRY(cudaMalloc(&d_graph_slot,
                           sizeof(int) * static_cast<std::size_t>(index.layout.graph_slot.size())));
  RAFT_CUDA_TRY(cudaMalloc(&d_counters, sizeof(kernels::DeviceTaskCounters)));
  RAFT_CUDA_TRY(cudaMalloc(&d_exact_tasks,
                           sizeof(ExactSearchTask) * static_cast<std::size_t>(max_exact_tasks)));
  RAFT_CUDA_TRY(cudaMalloc(&d_exact_task_indices_by_query,
                           sizeof(int) * static_cast<std::size_t>(max_exact_tasks)));
  RAFT_CUDA_TRY(
    cudaMalloc(&d_exact_task_counts, sizeof(int) * static_cast<std::size_t>(n_queries)));
  RAFT_CUDA_TRY(cudaMalloc(
    &d_graph_tasks, sizeof(GraphSearchTask) * static_cast<std::size_t>(max_graph_node_tasks)));
  RAFT_CUDA_TRY(cudaMalloc(&d_graph_task_indices_by_query,
                           sizeof(int) * static_cast<std::size_t>(max_graph_node_tasks)));
  RAFT_CUDA_TRY(
    cudaMalloc(&d_graph_task_counts, sizeof(int) * static_cast<std::size_t>(n_queries)));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_query_ranges,
                                query_ranges.data(),
                                sizeof(std::int64_t) * query_ranges.size(),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_graph_slot,
                                index.layout.graph_slot.data(),
                                sizeof(int) * index.layout.graph_slot.size(),
                                cudaMemcpyHostToDevice,
                                stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counters, 0, sizeof(kernels::DeviceTaskCounters), stream));

  DeviceSegmentTreeLayoutView device_layout{index.layout.rows,
                                            index.layout.leaf_size,
                                            index.layout.leaf_blocks,
                                            index.layout.leaf_base,
                                            d_graph_slot};
  constexpr int decompose_threads = 256;
  const int decompose_blocks      = (n_queries + decompose_threads - 1) / decompose_threads;
  kernels::decompose_ranges_kernel<<<decompose_blocks, decompose_threads, 0, stream>>>(
    d_query_ranges,
    n_queries,
    device_layout,
    d_exact_tasks,
    max_exact_tasks,
    d_exact_task_indices_by_query,
    d_exact_task_counts,
    d_graph_tasks,
    max_graph_node_tasks,
    graph_tasks_per_query,
    d_graph_task_indices_by_query,
    d_graph_task_counts,
    d_counters);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  kernels::DeviceTaskCounters h_counters;
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    &h_counters, d_counters, sizeof(kernels::DeviceTaskCounters), cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);
  RAFT_EXPECTS(h_counters.exact_count <= max_exact_tasks,
               "GPU range decomposition exceeded exact task capacity");
  RAFT_EXPECTS(h_counters.graph_node_count <= max_graph_node_tasks,
               "GPU range decomposition exceeded graph-node task capacity");
  RAFT_EXPECTS(h_counters.overflow_count == 0, "GPU range decomposition overflowed task storage");

  const int exact_task_count        = h_counters.exact_count;
  const int graph_node_task_count   = h_counters.graph_node_count;
  local_stats.exact_task_count      = static_cast<std::int64_t>(exact_task_count);
  local_stats.graph_node_task_count = static_cast<std::int64_t>(graph_node_task_count);
  local_stats.exact_vectors_scanned = static_cast<std::int64_t>(h_counters.exact_vec_count);

  RAFT_CUDA_TRY(cudaMalloc(&d_final_ids, sizeof(std::uint32_t) * final_output_elements));
  RAFT_CUDA_TRY(cudaMalloc(&d_final_distances, sizeof(float) * final_output_elements));
  if (graph_profile_enabled) {
    RAFT_CUDA_TRY(cudaMalloc(&d_graph_profile, sizeof(GraphSearchProfileCounters)));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_graph_profile, 0, sizeof(GraphSearchProfileCounters), stream));
  }

  if (graph_search_iteration_policy_needs_per_graph(search_iteration_params, graph_iterations) &&
      index.graph_pool.graph_count > 0) {
    std::vector<int> h_graph_iterations_by_graph(
      static_cast<std::size_t>(index.graph_pool.graph_count), graph_iterations);
    int max_graph_layer = 0;
    for (auto const& meta : index.graph_metas) {
      RAFT_EXPECTS(meta.graph_id >= 0 && meta.graph_id < index.graph_pool.graph_count,
                   "host graph metadata graph_id is out of range");
      max_graph_layer =
        std::max(max_graph_layer, segment_node_layer_from_bottom(meta.range));
    }
    local_stats.search_iteration_max_graph_layer = max_graph_layer;

    std::int64_t low_layer_graph_count = 0;
    std::int64_t upper_layer_graph_count = 0;
    std::int64_t override_graph_count = 0;
    long long iteration_sum = 0;
    int min_graph_iterations = std::numeric_limits<int>::max();
    int max_graph_iterations = 0;
    for (auto const& meta : index.graph_metas) {
      const int layer_from_bottom = segment_node_layer_from_bottom(meta.range);
      const int graph_specific_iterations = graph_search_iterations_for_layer(
        layer_from_bottom, max_graph_layer, graph_iterations, search_iteration_params);
      h_graph_iterations_by_graph[static_cast<std::size_t>(meta.graph_id)] =
        graph_specific_iterations;
      min_graph_iterations = std::min(min_graph_iterations, graph_specific_iterations);
      max_graph_iterations = std::max(max_graph_iterations, graph_specific_iterations);
      iteration_sum += graph_specific_iterations;
      if (graph_specific_iterations != graph_iterations) { ++override_graph_count; }
      if (search_iteration_params.policy == SegmentTreeSearchIterationPolicy::kLowerLayers &&
          search_iteration_params.lower_layer_count > 0 &&
          layer_from_bottom < search_iteration_params.lower_layer_count) {
        ++low_layer_graph_count;
      }
      if (search_iteration_params.policy == SegmentTreeSearchIterationPolicy::kUpperLayers &&
          search_iteration_params.upper_layer_count > 0) {
        const int first_upper_layer =
          std::max(0, max_graph_layer - search_iteration_params.upper_layer_count + 1);
        if (layer_from_bottom >= first_upper_layer) { ++upper_layer_graph_count; }
      }
    }
    local_stats.search_iteration_min_graph_iterations = min_graph_iterations;
    local_stats.search_iteration_max_graph_iterations = max_graph_iterations;
    local_stats.search_iteration_avg_graph_iterations =
      static_cast<double>(iteration_sum) / static_cast<double>(index.graph_pool.graph_count);
    local_stats.search_iteration_override_graph_count = override_graph_count;
    local_stats.low_layer_graph_count                 = low_layer_graph_count;
    local_stats.upper_layer_graph_count               = upper_layer_graph_count;
    if (override_graph_count > 0) {
      RAFT_CUDA_TRY(cudaMalloc(&d_graph_iterations_by_graph,
                               sizeof(int) * h_graph_iterations_by_graph.size()));
      RAFT_CUDA_TRY(cudaMemcpyAsync(d_graph_iterations_by_graph,
                                    h_graph_iterations_by_graph.data(),
                                    sizeof(int) * h_graph_iterations_by_graph.size(),
                                    cudaMemcpyHostToDevice,
                                    stream));
    }
  }

  const bool overlap_exact_and_graph = search_schedule == SegmentTreeSearchSchedule::kOverlap &&
                                       exact_task_count > 0 && graph_node_task_count > 0;
  rmm::cuda_stream exact_stream_holder(rmm::cuda_stream::flags::non_blocking);
  const auto exact_stream = overlap_exact_and_graph ? exact_stream_holder.view() : stream;
  cudaEvent_t exact_start_event = nullptr;
  cudaEvent_t exact_stop_event  = nullptr;
  bool exact_launched           = false;
  auto launch_exact = [&]() {
    if (exact_task_count <= 0) { return; }
    const auto task_count = exact_task_count;
    RAFT_CUDA_TRY(cudaMalloc(
      &d_exact_ids, sizeof(std::uint32_t) * static_cast<std::size_t>(exact_task_count) * topk));
    RAFT_CUDA_TRY(cudaMalloc(&d_exact_distances,
                             sizeof(float) * static_cast<std::size_t>(exact_task_count) * topk));
    RAFT_CUDA_TRY(cudaEventCreate(&exact_start_event));
    RAFT_CUDA_TRY(cudaEventCreate(&exact_stop_event));
    RAFT_CUDA_TRY(cudaEventRecord(exact_start_event, exact_stream));
    const auto shared_bytes =
      static_cast<std::size_t>(exact_threads) * topk * sizeof(float) +
      static_cast<std::size_t>(exact_threads) * topk * sizeof(std::uint32_t) +
      static_cast<std::size_t>(dataset.dim) * sizeof(float);
    kernels::exact_task_l2_topk_kernel<<<task_count, exact_threads, shared_bytes, exact_stream>>>(
      dataset,
      queries.data_handle(),
      d_exact_tasks,
      task_count,
      topk,
      d_exact_ids,
      d_exact_distances);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cudaEventRecord(exact_stop_event, exact_stream));
    exact_launched = true;
  };

  auto run_graph = [&]() {
    if (graph_node_task_count <= 0) { return; }
    const auto graph_start = std::chrono::steady_clock::now();
    RAFT_CUDA_TRY(
      cudaMalloc(&d_graph_ids,
                 sizeof(std::uint32_t) * static_cast<std::size_t>(graph_node_task_count) * topk));
    RAFT_CUDA_TRY(cudaMalloc(
      &d_graph_distances, sizeof(float) * static_cast<std::size_t>(graph_node_task_count) * topk));

    const int graph_itopk_size   = round_cagra_itopk_size(std::max(ef, topk));
    const int graph_search_width = std::min(graph_search_concurrency, graph_itopk_size);
    RAFT_EXPECTS(graph_itopk_size <= kRangeCagraMaxGraphItopk,
                 "range-CAGRA fused graph search currently supports itopk_size <= %d",
                 kRangeCagraMaxGraphItopk);
    RAFT_EXPECTS(graph_search_width <= kRangeCagraMaxGraphSearchWidth,
                 "range-CAGRA fused graph search currently supports search_width <= %d",
                 kRangeCagraMaxGraphSearchWidth);
    int max_graph_degree = 0;
    for (auto const& meta : index.graph_metas) {
      RAFT_EXPECTS(meta.degree <= kRangeCagraMaxGraphDegree,
                   "range-CAGRA fused graph search currently supports graph_degree <= %d",
                   kRangeCagraMaxGraphDegree);
      RAFT_EXPECTS(meta.range.size() < static_cast<std::int64_t>(kRangeCagraExpandedBit),
                   "range-CAGRA fused graph search reserves the uint32 MSB for expanded nodes");
      max_graph_degree = std::max(max_graph_degree, meta.degree);
    }

    const int max_merge_candidates =
      std::max(std::min(graph_itopk_size, std::max(entry_count, topk)),
               graph_search_width * std::max(1, max_graph_degree));
    const int graph_result_buffer_size = graph_itopk_size + max_merge_candidates;
    const int max_itopk_bin            = range_cagra_max_itopk_bin(graph_itopk_size);
    const bool use_radix_topk          = max_merge_candidates > 256;
    const int radix_workspace_words =
      use_radix_topk ? range_cagra_radix_workspace_words(max_itopk_bin) : 0;
    const int graph_hash_bitlen = range_cagra_hash_bitlen(graph_itopk_size, max_merge_candidates);
    RAFT_EXPECTS(graph_hash_bitlen <= kRangeCagraMaxVisitedHashBitlen,
                 "range-CAGRA shared visited hash needs bitlen=%d for itopk=%d and candidates=%d; "
                 "reduce graph_search_concurrency/degree or add global hash support",
                 graph_hash_bitlen,
                 graph_itopk_size,
                 max_merge_candidates);
    const int graph_hash_size = 1 << graph_hash_bitlen;
    const int hash_reset_interval =
      range_cagra_hash_reset_interval(graph_itopk_size, max_merge_candidates, graph_hash_bitlen);
    const int graph_query_workspace_len = range_cagra_distance_query_workspace_len(dataset.dim);
    const auto graph_shared_bytes =
      static_cast<std::size_t>(graph_query_workspace_len) * sizeof(float) +
      static_cast<std::size_t>(graph_result_buffer_size) * (sizeof(float) + sizeof(std::uint32_t)) +
      static_cast<std::size_t>(graph_search_width) * sizeof(std::uint32_t) +
      static_cast<std::size_t>(kRangeCagraTopkWorkspaceWords) * sizeof(std::uint32_t) +
      static_cast<std::size_t>(radix_workspace_words) * sizeof(std::uint32_t) +
      static_cast<std::size_t>(graph_hash_size) * sizeof(std::uint32_t) + sizeof(std::uint32_t);
    int device = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    int default_shared_bytes = 0;
    int optin_shared_bytes   = 0;
    RAFT_CUDA_TRY(
      cudaDeviceGetAttribute(&default_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
    RAFT_CUDA_TRY(
      cudaDeviceGetAttribute(&optin_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const int shared_limit = std::max(default_shared_bytes, optin_shared_bytes);
    const bool use_vectorized_graph_distance =
      ((dataset.dim & 3) == 0) && ((dataset.stride & 3) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(dataset.base) & 15) == 0);
    auto prepare_graph_kernel = [&](auto kernel) {
      cudaFuncAttributes attrs{};
      RAFT_CUDA_TRY(cudaFuncGetAttributes(&attrs, kernel));
      const auto total_kernel_shared_bytes =
        graph_shared_bytes + static_cast<std::size_t>(attrs.sharedSizeBytes);
      RAFT_EXPECTS(total_kernel_shared_bytes <= static_cast<std::size_t>(shared_limit),
                   "range-CAGRA graph search shared memory request %zu dynamic + %zu static "
                   "exceeds device limit %d",
                   graph_shared_bytes,
                   static_cast<std::size_t>(attrs.sharedSizeBytes),
                   shared_limit);
      if (total_kernel_shared_bytes > static_cast<std::size_t>(default_shared_bytes)) {
        RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel,
                                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(graph_shared_bytes)));
      }
    };
    if (use_radix_topk && use_vectorized_graph_distance) {
      prepare_graph_kernel(kernels::range_graph_task_search_kernel<true, true>);
      kernels::range_graph_task_search_kernel<true, true>
        <<<graph_node_task_count, graph_threads, graph_shared_bytes, stream>>>(
          dataset,
          queries.data_handle(),
          d_graph_tasks,
          graph_node_task_count,
          view(index.graph_pool),
          topk,
          graph_itopk_size,
          graph_search_width,
          entry_count,
          graph_iterations,
          d_graph_iterations_by_graph,
          hash_reset_interval,
          graph_hash_bitlen,
          0x128394ULL,
          d_graph_profile,
          d_graph_ids,
          d_graph_distances);
    } else if (use_radix_topk) {
      prepare_graph_kernel(kernels::range_graph_task_search_kernel<true, false>);
      kernels::range_graph_task_search_kernel<true, false>
        <<<graph_node_task_count, graph_threads, graph_shared_bytes, stream>>>(
          dataset,
          queries.data_handle(),
          d_graph_tasks,
          graph_node_task_count,
          view(index.graph_pool),
          topk,
          graph_itopk_size,
          graph_search_width,
          entry_count,
          graph_iterations,
          d_graph_iterations_by_graph,
          hash_reset_interval,
          graph_hash_bitlen,
          0x128394ULL,
          d_graph_profile,
          d_graph_ids,
          d_graph_distances);
    } else if (use_vectorized_graph_distance) {
      prepare_graph_kernel(kernels::range_graph_task_search_kernel<false, true>);
      kernels::range_graph_task_search_kernel<false, true>
        <<<graph_node_task_count, graph_threads, graph_shared_bytes, stream>>>(
          dataset,
          queries.data_handle(),
          d_graph_tasks,
          graph_node_task_count,
          view(index.graph_pool),
          topk,
          graph_itopk_size,
          graph_search_width,
          entry_count,
          graph_iterations,
          d_graph_iterations_by_graph,
          hash_reset_interval,
          graph_hash_bitlen,
          0x128394ULL,
          d_graph_profile,
          d_graph_ids,
          d_graph_distances);
    } else {
      prepare_graph_kernel(kernels::range_graph_task_search_kernel<false, false>);
      kernels::range_graph_task_search_kernel<false, false>
        <<<graph_node_task_count, graph_threads, graph_shared_bytes, stream>>>(
          dataset,
          queries.data_handle(),
          d_graph_tasks,
          graph_node_task_count,
          view(index.graph_pool),
          topk,
          graph_itopk_size,
          graph_search_width,
          entry_count,
          graph_iterations,
          d_graph_iterations_by_graph,
          hash_reset_interval,
          graph_hash_bitlen,
          0x128394ULL,
          d_graph_profile,
          d_graph_ids,
          d_graph_distances);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);

    const auto graph_stop = std::chrono::steady_clock::now();
    local_stats.graph_seconds += std::chrono::duration<double>(graph_stop - graph_start).count();
  };

  switch (search_schedule) {
    case SegmentTreeSearchSchedule::kGraphThenExact:
      run_graph();
      launch_exact();
      if (exact_launched) { raft::resource::sync_stream(res); }
      break;
    case SegmentTreeSearchSchedule::kExactThenGraph:
      launch_exact();
      if (exact_launched) { raft::resource::sync_stream(res); }
      run_graph();
      break;
    case SegmentTreeSearchSchedule::kOverlap:
      launch_exact();
      run_graph();
      break;
  }

  if (exact_launched && exact_stream != stream) {
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream.value(), exact_stop_event, 0));
  }

  const auto merge_start      = std::chrono::steady_clock::now();
  constexpr int merge_threads = 32;
  const auto merge_shared_bytes =
    static_cast<std::size_t>(merge_threads) * topk * (sizeof(float) + sizeof(std::uint32_t));
  kernels::merge_task_topk_kernel<<<n_queries, merge_threads, merge_shared_bytes, stream>>>(
    d_exact_tasks,
    d_exact_task_indices_by_query,
    d_exact_task_counts,
    kRangeCagraMaxExactTasksPerQuery,
    d_exact_ids,
    d_exact_distances,
    d_graph_tasks,
    d_graph_task_indices_by_query,
    d_graph_task_counts,
    graph_tasks_per_query,
    d_graph_ids,
    d_graph_distances,
    topk,
    d_final_ids,
    d_final_distances);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaMemcpyAsync(final_ids.data(),
                                d_final_ids,
                                sizeof(std::uint32_t) * final_ids.size(),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(final_distances.data(),
                                d_final_distances,
                                sizeof(float) * final_distances.size(),
                                cudaMemcpyDeviceToHost,
                                stream));
  raft::resource::sync_stream(res);
  const auto merge_stop = std::chrono::steady_clock::now();
  local_stats.merge_seconds += std::chrono::duration<double>(merge_stop - merge_start).count();
  local_stats.search_seconds += std::chrono::duration<double>(merge_stop - search_start).count();
  if (exact_launched) {
    float exact_milliseconds = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&exact_milliseconds, exact_start_event, exact_stop_event));
    local_stats.exact_seconds += static_cast<double>(exact_milliseconds) / 1000.0;
    RAFT_CUDA_TRY(cudaEventDestroy(exact_start_event));
    RAFT_CUDA_TRY(cudaEventDestroy(exact_stop_event));
  }
  if (d_graph_profile != nullptr) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(&local_stats.graph_profile,
                                  d_graph_profile,
                                  sizeof(GraphSearchProfileCounters),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    raft::resource::sync_stream(res);
  }

  RAFT_CUDA_TRY(cudaFree(d_exact_tasks));
  RAFT_CUDA_TRY(cudaFree(d_exact_task_indices_by_query));
  RAFT_CUDA_TRY(cudaFree(d_exact_task_counts));
  RAFT_CUDA_TRY(cudaFree(d_exact_ids));
  RAFT_CUDA_TRY(cudaFree(d_exact_distances));
  RAFT_CUDA_TRY(cudaFree(d_graph_tasks));
  RAFT_CUDA_TRY(cudaFree(d_graph_task_indices_by_query));
  RAFT_CUDA_TRY(cudaFree(d_graph_task_counts));
  RAFT_CUDA_TRY(cudaFree(d_graph_ids));
  RAFT_CUDA_TRY(cudaFree(d_graph_distances));
  RAFT_CUDA_TRY(cudaFree(d_graph_profile));
  RAFT_CUDA_TRY(cudaFree(d_graph_iterations_by_graph));
  RAFT_CUDA_TRY(cudaFree(d_final_ids));
  RAFT_CUDA_TRY(cudaFree(d_final_distances));
  RAFT_CUDA_TRY(cudaFree(d_query_ranges));
  RAFT_CUDA_TRY(cudaFree(d_graph_slot));
  RAFT_CUDA_TRY(cudaFree(d_counters));

  if (stats != nullptr) { *stats = local_stats; }
}

}  // namespace cuvs::neighbors::range_cagra::detail
