/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_segment_tree.cuh"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {

/**
 * Workspace-backed range-CAGRA segment-tree search path.
 *
 * This header is intentionally separate from range_cagra_segment_tree.cuh so the
 * baseline implementation remains available while this variant reduces
 * repeated host-side work.  It keeps the segment-tree graph_slot table resident on device,
 * reuses all search buffers across calls, and can optionally avoid the
 * pre-search D2H counter sync by launching over the reserved task capacity with
 * invalid task slots masked out.
 *
 * The ANN semantics are unchanged: range decomposition still emits exact leaf
 * fragments plus GraphSearchTask entries, graph tasks still search local-id
 * range graphs, and the final merge still returns global ids.
 */
enum class SegmentTreeWorkspaceLaunchMode {
  /**
   * Copy decomposition counters to host before exact/graph launches.  This is
   * closest to the original search behavior, but still reuses allocations and
   * keeps graph_slot on device.
   */
  kHostCountSync,

  /**
   * Do not copy task counters before search.  Exact tasks use a masked kernel;
   * graph tasks launch over the reserved task array after unused slots are
   * initialized to graph_id=-1.  Overflows are checked after the final sync.
   */
  kDeviceCountNoPreSync,
};

namespace workspace_kernels {

RAFT_KERNEL reset_task_storage_kernel(ExactSearchTask* __restrict__ exact_tasks,
                                      int max_exact_tasks,
                                      GraphSearchTask* __restrict__ graph_tasks,
                                      int max_graph_tasks)
{
  const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < max_exact_tasks) { exact_tasks[tid] = {-1, 0, -1}; }
  if (tid < max_graph_tasks) { graph_tasks[tid] = {-1, -1}; }
}

RAFT_KERNEL exact_task_l2_topk_masked_kernel(GlobalDatasetView dataset,
                                            const float* __restrict__ queries,
                                            const ExactSearchTask* __restrict__ tasks,
                                            const kernels::DeviceTaskCounters* __restrict__ counters,
                                            int topk,
                                            std::uint32_t* __restrict__ out_ids,
                                            float* __restrict__ out_distances)
{
  const int task_id = static_cast<int>(blockIdx.x);
  if (task_id >= counters->exact_count) { return; }

  const auto task = tasks[task_id];
  if (task.query_id < 0 || task.right < task.left) { return; }

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
    kernels::insert_topk(dist, static_cast<std::uint32_t>(id), local_dist, local_id, topk);
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
        if (shared_ids[base + k] != std::numeric_limits<std::uint32_t>::max()) {
          kernels::insert_topk(shared_dist[base + k], shared_ids[base + k], final_dist, final_id, topk);
        }
      }
    }
    for (int k = 0; k < topk; ++k) {
      out_ids[static_cast<std::int64_t>(task_id) * topk + k]       = final_id[k];
      out_distances[static_cast<std::int64_t>(task_id) * topk + k] = final_dist[k];
    }
  }
}

}  // namespace workspace_kernels

struct SegmentTreeRangeCagraSearchWorkspace {
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

  int query_capacity             = 0;
  int max_exact_task_capacity    = 0;
  int max_graph_task_capacity    = 0;
  int topk_capacity              = 0;
  std::size_t graph_slot_capacity = 0;
  std::size_t graph_slot_size     = 0;
  const int* graph_slot_host_ptr  = nullptr;
  std::int64_t graph_slot_rows    = -1;
  std::int64_t graph_slot_base    = -1;
  int graph_slot_leaf_size        = -1;

  SegmentTreeRangeCagraSearchWorkspace() = default;
  SegmentTreeRangeCagraSearchWorkspace(SegmentTreeRangeCagraSearchWorkspace const&) = delete;
  auto operator=(SegmentTreeRangeCagraSearchWorkspace const&)
    -> SegmentTreeRangeCagraSearchWorkspace& = delete;

  SegmentTreeRangeCagraSearchWorkspace(SegmentTreeRangeCagraSearchWorkspace&& other) noexcept
  {
    move_from(other);
  }

  auto operator=(SegmentTreeRangeCagraSearchWorkspace&& other) noexcept
    -> SegmentTreeRangeCagraSearchWorkspace&
  {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }

  ~SegmentTreeRangeCagraSearchWorkspace() { release(); }

  void release() noexcept
  {
    free_ptr(d_query_ranges);
    free_ptr(d_graph_slot);
    free_ptr(d_counters);
    free_ptr(d_exact_tasks);
    free_ptr(d_exact_task_indices_by_query);
    free_ptr(d_exact_task_counts);
    free_ptr(d_exact_ids);
    free_ptr(d_exact_distances);
    free_ptr(d_graph_tasks);
    free_ptr(d_graph_task_indices_by_query);
    free_ptr(d_graph_task_counts);
    free_ptr(d_graph_ids);
    free_ptr(d_graph_distances);
    free_ptr(d_final_ids);
    free_ptr(d_final_distances);
    free_ptr(d_graph_profile);
    free_ptr(d_graph_iterations_by_graph);
    query_capacity          = 0;
    max_exact_task_capacity = 0;
    max_graph_task_capacity = 0;
    topk_capacity           = 0;
    graph_slot_capacity     = 0;
    graph_slot_size         = 0;
    graph_slot_host_ptr     = nullptr;
    graph_slot_rows         = -1;
    graph_slot_base         = -1;
    graph_slot_leaf_size    = -1;
    query_range_capacity_   = 0;
    query_capacity_         = 0;
    graph_count_capacity_   = 0;
    exact_task_capacity_    = 0;
    exact_index_capacity_   = 0;
    graph_task_capacity_    = 0;
    graph_index_capacity_   = 0;
    exact_result_task_capacity_ = 0;
    graph_result_task_capacity_ = 0;
    exact_id_capacity_          = 0;
    exact_distance_capacity_    = 0;
    graph_id_capacity_          = 0;
    graph_distance_capacity_    = 0;
    final_id_capacity_          = 0;
    final_distance_capacity_    = 0;
    counter_capacity_           = 0;
    graph_profile_capacity_     = 0;
    graph_iteration_capacity_   = 0;
  }

  void ensure_capacity(int n_queries,
                       int max_exact_tasks,
                       int max_graph_tasks,
                       int topk,
                       bool need_graph_profile)
  {
    RAFT_EXPECTS(n_queries > 0, "workspace requires at least one query");
    RAFT_EXPECTS(max_exact_tasks >= 0, "max_exact_tasks must be non-negative");
    RAFT_EXPECTS(max_graph_tasks >= 0, "max_graph_tasks must be non-negative");
    RAFT_EXPECTS(topk > 0, "topk must be positive");

    if (n_queries > query_capacity) {
      resize_device_buffer(d_query_ranges,
                           static_cast<std::size_t>(n_queries) * 2,
                           query_range_capacity_);
      resize_device_buffer(d_exact_task_counts, static_cast<std::size_t>(n_queries), query_capacity_);
      resize_device_buffer(d_graph_task_counts, static_cast<std::size_t>(n_queries), graph_count_capacity_);
      query_capacity = n_queries;
    }
    if (max_exact_tasks > max_exact_task_capacity) {
      resize_device_buffer(d_exact_tasks, static_cast<std::size_t>(max_exact_tasks), exact_task_capacity_);
      resize_device_buffer(d_exact_task_indices_by_query,
                           static_cast<std::size_t>(max_exact_tasks),
                           exact_index_capacity_);
      max_exact_task_capacity = max_exact_tasks;
    }
    if (max_graph_tasks > max_graph_task_capacity) {
      resize_device_buffer(d_graph_tasks, static_cast<std::size_t>(max_graph_tasks), graph_task_capacity_);
      resize_device_buffer(d_graph_task_indices_by_query,
                           static_cast<std::size_t>(max_graph_tasks),
                           graph_index_capacity_);
      max_graph_task_capacity = max_graph_tasks;
    }
    if (topk > topk_capacity ||
        static_cast<std::size_t>(max_exact_tasks) > exact_result_task_capacity_) {
      resize_device_buffer(d_exact_ids,
                           static_cast<std::size_t>(max_exact_tasks) * topk,
                           exact_id_capacity_);
      resize_device_buffer(d_exact_distances,
                           static_cast<std::size_t>(max_exact_tasks) * topk,
                           exact_distance_capacity_);
      exact_result_task_capacity_ = max_exact_tasks;
    }
    if (topk > topk_capacity ||
        static_cast<std::size_t>(max_graph_tasks) > graph_result_task_capacity_) {
      resize_device_buffer(d_graph_ids,
                           static_cast<std::size_t>(max_graph_tasks) * topk,
                           graph_id_capacity_);
      resize_device_buffer(d_graph_distances,
                           static_cast<std::size_t>(max_graph_tasks) * topk,
                           graph_distance_capacity_);
      graph_result_task_capacity_ = max_graph_tasks;
    }
    const auto final_elements = static_cast<std::size_t>(n_queries) * topk;
    resize_device_buffer(d_final_ids, final_elements, final_id_capacity_);
    resize_device_buffer(d_final_distances, final_elements, final_distance_capacity_);
    if (topk > topk_capacity) { topk_capacity = topk; }
    resize_device_buffer(d_counters, std::size_t{1}, counter_capacity_);
    if (need_graph_profile) {
      resize_device_buffer(d_graph_profile, std::size_t{1}, graph_profile_capacity_);
    }
  }

  void ensure_graph_slot_resident(raft::resources const& res, SegmentTreeRangeCagraIndex const& index)
  {
    const auto slot_size = index.layout.graph_slot.size();
    if (slot_size > graph_slot_capacity) {
      resize_device_buffer(d_graph_slot, slot_size, graph_slot_capacity);
    }
    const bool same_index = graph_slot_host_ptr == index.layout.graph_slot.data() &&
                            graph_slot_size == slot_size &&
                            graph_slot_rows == index.layout.rows &&
                            graph_slot_base == index.layout.leaf_base &&
                            graph_slot_leaf_size == index.layout.leaf_size;
    if (!same_index) {
      auto stream = raft::resource::get_cuda_stream(res);
      RAFT_CUDA_TRY(cudaMemcpyAsync(d_graph_slot,
                                    index.layout.graph_slot.data(),
                                    sizeof(int) * slot_size,
                                    cudaMemcpyHostToDevice,
                                    stream));
      graph_slot_host_ptr  = index.layout.graph_slot.data();
      graph_slot_size      = slot_size;
      graph_slot_rows      = index.layout.rows;
      graph_slot_base      = index.layout.leaf_base;
      graph_slot_leaf_size = index.layout.leaf_size;
    }
  }

  void ensure_graph_iteration_capacity(std::size_t graph_count)
  {
    resize_device_buffer(d_graph_iterations_by_graph, graph_count, graph_iteration_capacity_);
  }

 private:
  std::size_t query_range_capacity_         = 0;
  std::size_t query_capacity_               = 0;
  std::size_t graph_count_capacity_         = 0;
  std::size_t exact_task_capacity_          = 0;
  std::size_t exact_index_capacity_         = 0;
  std::size_t graph_task_capacity_          = 0;
  std::size_t graph_index_capacity_         = 0;
  std::size_t exact_result_task_capacity_   = 0;
  std::size_t graph_result_task_capacity_   = 0;
  std::size_t exact_id_capacity_            = 0;
  std::size_t exact_distance_capacity_      = 0;
  std::size_t graph_id_capacity_            = 0;
  std::size_t graph_distance_capacity_      = 0;
  std::size_t final_id_capacity_            = 0;
  std::size_t final_distance_capacity_      = 0;
  std::size_t counter_capacity_             = 0;
  std::size_t graph_profile_capacity_       = 0;
  std::size_t graph_iteration_capacity_     = 0;

  template <typename T>
  static void free_ptr(T*& ptr) noexcept
  {
    if (ptr != nullptr) {
      RAFT_CUDA_TRY_NO_THROW(cudaFree(ptr));
      ptr = nullptr;
    }
  }

  template <typename T>
  static void resize_device_buffer(T*& ptr, std::size_t required, std::size_t& capacity)
  {
    if (required <= capacity) { return; }
    free_ptr(ptr);
    if (required > 0) { RAFT_CUDA_TRY(cudaMalloc(&ptr, sizeof(T) * required)); }
    capacity = required;
  }

  void move_from(SegmentTreeRangeCagraSearchWorkspace& other) noexcept
  {
    d_query_ranges                = other.d_query_ranges;
    d_graph_slot                  = other.d_graph_slot;
    d_counters                    = other.d_counters;
    d_exact_tasks                 = other.d_exact_tasks;
    d_exact_task_indices_by_query = other.d_exact_task_indices_by_query;
    d_exact_task_counts           = other.d_exact_task_counts;
    d_exact_ids                   = other.d_exact_ids;
    d_exact_distances             = other.d_exact_distances;
    d_graph_tasks                 = other.d_graph_tasks;
    d_graph_task_indices_by_query = other.d_graph_task_indices_by_query;
    d_graph_task_counts           = other.d_graph_task_counts;
    d_graph_ids                   = other.d_graph_ids;
    d_graph_distances             = other.d_graph_distances;
    d_final_ids                   = other.d_final_ids;
    d_final_distances             = other.d_final_distances;
    d_graph_profile               = other.d_graph_profile;
    d_graph_iterations_by_graph   = other.d_graph_iterations_by_graph;
    query_capacity                = other.query_capacity;
    max_exact_task_capacity       = other.max_exact_task_capacity;
    max_graph_task_capacity       = other.max_graph_task_capacity;
    topk_capacity                 = other.topk_capacity;
    graph_slot_capacity           = other.graph_slot_capacity;
    graph_slot_size               = other.graph_slot_size;
    graph_slot_host_ptr           = other.graph_slot_host_ptr;
    graph_slot_rows               = other.graph_slot_rows;
    graph_slot_base               = other.graph_slot_base;
    graph_slot_leaf_size          = other.graph_slot_leaf_size;
    query_range_capacity_         = other.query_range_capacity_;
    query_capacity_               = other.query_capacity_;
    graph_count_capacity_         = other.graph_count_capacity_;
    exact_task_capacity_          = other.exact_task_capacity_;
    exact_index_capacity_         = other.exact_index_capacity_;
    graph_task_capacity_          = other.graph_task_capacity_;
    graph_index_capacity_         = other.graph_index_capacity_;
    exact_result_task_capacity_   = other.exact_result_task_capacity_;
    graph_result_task_capacity_   = other.graph_result_task_capacity_;
    exact_id_capacity_            = other.exact_id_capacity_;
    exact_distance_capacity_      = other.exact_distance_capacity_;
    graph_id_capacity_            = other.graph_id_capacity_;
    graph_distance_capacity_      = other.graph_distance_capacity_;
    final_id_capacity_            = other.final_id_capacity_;
    final_distance_capacity_      = other.final_distance_capacity_;
    counter_capacity_             = other.counter_capacity_;
    graph_profile_capacity_       = other.graph_profile_capacity_;
    graph_iteration_capacity_     = other.graph_iteration_capacity_;

    other.d_query_ranges                = nullptr;
    other.d_graph_slot                  = nullptr;
    other.d_counters                    = nullptr;
    other.d_exact_tasks                 = nullptr;
    other.d_exact_task_indices_by_query = nullptr;
    other.d_exact_task_counts           = nullptr;
    other.d_exact_ids                   = nullptr;
    other.d_exact_distances             = nullptr;
    other.d_graph_tasks                 = nullptr;
    other.d_graph_task_indices_by_query = nullptr;
    other.d_graph_task_counts           = nullptr;
    other.d_graph_ids                   = nullptr;
    other.d_graph_distances             = nullptr;
    other.d_final_ids                   = nullptr;
    other.d_final_distances             = nullptr;
    other.d_graph_profile               = nullptr;
    other.d_graph_iterations_by_graph   = nullptr;
    other.query_capacity                = 0;
    other.max_exact_task_capacity       = 0;
    other.max_graph_task_capacity       = 0;
    other.topk_capacity                 = 0;
    other.graph_slot_capacity           = 0;
    other.graph_slot_size               = 0;
    other.graph_slot_host_ptr           = nullptr;
    other.graph_slot_rows               = -1;
    other.graph_slot_base               = -1;
    other.graph_slot_leaf_size          = -1;
    other.query_range_capacity_         = 0;
    other.query_capacity_               = 0;
    other.graph_count_capacity_         = 0;
    other.exact_task_capacity_          = 0;
    other.exact_index_capacity_         = 0;
    other.graph_task_capacity_          = 0;
    other.graph_index_capacity_         = 0;
    other.exact_result_task_capacity_   = 0;
    other.graph_result_task_capacity_   = 0;
    other.exact_id_capacity_            = 0;
    other.exact_distance_capacity_      = 0;
    other.graph_id_capacity_            = 0;
    other.graph_distance_capacity_      = 0;
    other.final_id_capacity_            = 0;
    other.final_distance_capacity_      = 0;
    other.counter_capacity_             = 0;
    other.graph_profile_capacity_       = 0;
    other.graph_iteration_capacity_     = 0;
  }
};

inline void run_segment_tree_range_cagra_search_with_workspace(
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
  SegmentTreeRangeCagraSearchWorkspace& workspace,
  SegmentTreeSearchStats* stats = nullptr,
  int graph_search_concurrency  = 1,
  bool graph_profile_enabled    = false,
  int graph_threads             = 128,
  SegmentTreeSearchSchedule search_schedule = SegmentTreeSearchSchedule::kOverlap,
  SegmentTreeWorkspaceLaunchMode launch_mode = SegmentTreeWorkspaceLaunchMode::kHostCountSync,
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

  workspace.ensure_capacity(
    n_queries, max_exact_tasks, max_graph_node_tasks, topk, graph_profile_enabled);
  workspace.ensure_graph_slot_resident(res, index);

  RAFT_CUDA_TRY(cudaMemcpyAsync(workspace.d_query_ranges,
                                query_ranges.data(),
                                sizeof(std::int64_t) * query_ranges.size(),
                                cudaMemcpyHostToDevice,
                                stream));
  const bool host_count_sync = launch_mode == SegmentTreeWorkspaceLaunchMode::kHostCountSync;
  RAFT_CUDA_TRY(cudaMemsetAsync(
    workspace.d_counters, 0, sizeof(kernels::DeviceTaskCounters), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(workspace.d_exact_task_counts, 0, sizeof(int) * n_queries, stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(workspace.d_graph_task_counts, 0, sizeof(int) * n_queries, stream));
  if (graph_profile_enabled) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      workspace.d_graph_profile, 0, sizeof(GraphSearchProfileCounters), stream));
  }

  constexpr int reset_threads = 256;
  const int reset_count       = std::max(max_exact_tasks, max_graph_node_tasks);
  if (!host_count_sync && reset_count > 0) {
    const int reset_blocks = (reset_count + reset_threads - 1) / reset_threads;
    workspace_kernels::reset_task_storage_kernel<<<reset_blocks, reset_threads, 0, stream>>>(
      workspace.d_exact_tasks, max_exact_tasks, workspace.d_graph_tasks, max_graph_node_tasks);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  DeviceSegmentTreeLayoutView device_layout{index.layout.rows,
                                            index.layout.leaf_size,
                                            index.layout.leaf_blocks,
                                            index.layout.leaf_base,
                                            workspace.d_graph_slot};
  constexpr int decompose_threads = 256;
  const int decompose_blocks      = (n_queries + decompose_threads - 1) / decompose_threads;
  kernels::decompose_ranges_kernel<<<decompose_blocks, decompose_threads, 0, stream>>>(
    workspace.d_query_ranges,
    n_queries,
    device_layout,
    workspace.d_exact_tasks,
    max_exact_tasks,
    workspace.d_exact_task_indices_by_query,
    workspace.d_exact_task_counts,
    workspace.d_graph_tasks,
    max_graph_node_tasks,
    graph_tasks_per_query,
    workspace.d_graph_task_indices_by_query,
    workspace.d_graph_task_counts,
    workspace.d_counters);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  kernels::DeviceTaskCounters h_counters{};
  int exact_task_count       = host_count_sync ? 0 : max_exact_tasks;
  int graph_node_task_count  = host_count_sync ? 0 : max_graph_node_tasks;
  if (host_count_sync) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(&h_counters,
                                  workspace.d_counters,
                                  sizeof(kernels::DeviceTaskCounters),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    raft::resource::sync_stream(res);
    RAFT_EXPECTS(h_counters.exact_count <= max_exact_tasks,
                 "GPU range decomposition exceeded exact task capacity");
    RAFT_EXPECTS(h_counters.graph_node_count <= max_graph_node_tasks,
                 "GPU range decomposition exceeded graph-node task capacity");
    RAFT_EXPECTS(h_counters.overflow_count == 0, "GPU range decomposition overflowed task storage");
    exact_task_count                    = h_counters.exact_count;
    graph_node_task_count               = h_counters.graph_node_count;
    local_stats.exact_task_count        = static_cast<std::int64_t>(h_counters.exact_count);
    local_stats.graph_node_task_count   = static_cast<std::int64_t>(h_counters.graph_node_count);
    local_stats.exact_vectors_scanned   = static_cast<std::int64_t>(h_counters.exact_vec_count);
  }

  const int* d_graph_iterations_by_graph = nullptr;
  if (graph_search_iteration_policy_needs_per_graph(search_iteration_params, graph_iterations) &&
      index.graph_pool.graph_count > 0) {
    std::vector<int> h_graph_iterations_by_graph(
      static_cast<std::size_t>(index.graph_pool.graph_count), graph_iterations);
    int max_graph_layer = 0;
    for (auto const& meta : index.graph_metas) {
      RAFT_EXPECTS(meta.graph_id >= 0 && meta.graph_id < index.graph_pool.graph_count,
                   "host graph metadata graph_id is out of range");
      max_graph_layer = std::max(max_graph_layer, segment_node_layer_from_bottom(meta.range));
    }
    local_stats.search_iteration_max_graph_layer = max_graph_layer;

    std::int64_t low_layer_graph_count   = 0;
    std::int64_t upper_layer_graph_count = 0;
    std::int64_t override_graph_count    = 0;
    long long iteration_sum              = 0;
    int min_graph_iterations             = std::numeric_limits<int>::max();
    int max_graph_iterations             = 0;
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
      workspace.ensure_graph_iteration_capacity(h_graph_iterations_by_graph.size());
      RAFT_CUDA_TRY(cudaMemcpyAsync(workspace.d_graph_iterations_by_graph,
                                    h_graph_iterations_by_graph.data(),
                                    sizeof(int) * h_graph_iterations_by_graph.size(),
                                    cudaMemcpyHostToDevice,
                                    stream));
      d_graph_iterations_by_graph = workspace.d_graph_iterations_by_graph;
    }
  }

  const bool overlap_exact_and_graph = search_schedule == SegmentTreeSearchSchedule::kOverlap &&
                                       exact_task_count > 0 && graph_node_task_count > 0;
  std::optional<rmm::cuda_stream> exact_stream_holder;
  rmm::cuda_stream_view exact_stream = stream;
  if (overlap_exact_and_graph) {
    exact_stream_holder.emplace(rmm::cuda_stream::flags::non_blocking);
    exact_stream = exact_stream_holder->view();
  }

  cudaEvent_t decompose_done_event = nullptr;
  if (!host_count_sync && exact_stream != stream) {
    RAFT_CUDA_TRY(cudaEventCreate(&decompose_done_event));
    RAFT_CUDA_TRY(cudaEventRecord(decompose_done_event, stream));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(exact_stream.value(), decompose_done_event, 0));
  }

  cudaEvent_t exact_start_event = nullptr;
  cudaEvent_t exact_stop_event  = nullptr;
  bool exact_launched           = false;
  auto launch_exact             = [&]() {
    if (exact_task_count <= 0) { return; }
    RAFT_CUDA_TRY(cudaEventCreate(&exact_start_event));
    RAFT_CUDA_TRY(cudaEventCreate(&exact_stop_event));
    RAFT_CUDA_TRY(cudaEventRecord(exact_start_event, exact_stream));
    const auto shared_bytes =
      static_cast<std::size_t>(exact_threads) * topk * sizeof(float) +
      static_cast<std::size_t>(exact_threads) * topk * sizeof(std::uint32_t) +
      static_cast<std::size_t>(dataset.dim) * sizeof(float);
    if (host_count_sync) {
      kernels::exact_task_l2_topk_kernel<<<exact_task_count, exact_threads, shared_bytes, exact_stream>>>(
        dataset,
        queries.data_handle(),
        workspace.d_exact_tasks,
        exact_task_count,
        topk,
        workspace.d_exact_ids,
        workspace.d_exact_distances);
    } else {
      workspace_kernels::exact_task_l2_topk_masked_kernel<<<exact_task_count,
                                                            exact_threads,
                                                            shared_bytes,
                                                            exact_stream>>>(
        dataset,
        queries.data_handle(),
        workspace.d_exact_tasks,
        workspace.d_counters,
        topk,
        workspace.d_exact_ids,
        workspace.d_exact_distances);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    RAFT_CUDA_TRY(cudaEventRecord(exact_stop_event, exact_stream));
    exact_launched = true;
  };

  auto run_graph = [&]() {
    if (graph_node_task_count <= 0) { return; }
    const auto graph_start = std::chrono::steady_clock::now();

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
    auto launch_graph_kernel = [&](auto kernel) {
      prepare_graph_kernel(kernel);
      kernel<<<graph_node_task_count, graph_threads, graph_shared_bytes, stream>>>(
        dataset,
        queries.data_handle(),
        workspace.d_graph_tasks,
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
        graph_profile_enabled ? workspace.d_graph_profile : nullptr,
        workspace.d_graph_ids,
        workspace.d_graph_distances);
    };
    if (use_radix_topk && use_vectorized_graph_distance) {
      launch_graph_kernel(kernels::range_graph_task_search_kernel<true, true>);
    } else if (use_radix_topk) {
      launch_graph_kernel(kernels::range_graph_task_search_kernel<true, false>);
    } else if (use_vectorized_graph_distance) {
      launch_graph_kernel(kernels::range_graph_task_search_kernel<false, true>);
    } else {
      launch_graph_kernel(kernels::range_graph_task_search_kernel<false, false>);
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
    workspace.d_exact_tasks,
    workspace.d_exact_task_indices_by_query,
    workspace.d_exact_task_counts,
    kRangeCagraMaxExactTasksPerQuery,
    workspace.d_exact_ids,
    workspace.d_exact_distances,
    workspace.d_graph_tasks,
    workspace.d_graph_task_indices_by_query,
    workspace.d_graph_task_counts,
    graph_tasks_per_query,
    workspace.d_graph_ids,
    workspace.d_graph_distances,
    topk,
    workspace.d_final_ids,
    workspace.d_final_distances);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaMemcpyAsync(final_ids.data(),
                                workspace.d_final_ids,
                                sizeof(std::uint32_t) * final_ids.size(),
                                cudaMemcpyDeviceToHost,
                                stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(final_distances.data(),
                                workspace.d_final_distances,
                                sizeof(float) * final_distances.size(),
                                cudaMemcpyDeviceToHost,
                                stream));
  raft::resource::sync_stream(res);
  const auto merge_stop = std::chrono::steady_clock::now();
  local_stats.merge_seconds += std::chrono::duration<double>(merge_stop - merge_start).count();
  local_stats.search_seconds += std::chrono::duration<double>(merge_stop - search_start).count();

  if (!host_count_sync) {
    RAFT_CUDA_TRY(cudaMemcpy(
      &h_counters, workspace.d_counters, sizeof(kernels::DeviceTaskCounters), cudaMemcpyDeviceToHost));
    RAFT_EXPECTS(h_counters.exact_count <= max_exact_tasks,
                 "GPU range decomposition exceeded exact task capacity");
    RAFT_EXPECTS(h_counters.graph_node_count <= max_graph_node_tasks,
                 "GPU range decomposition exceeded graph-node task capacity");
    RAFT_EXPECTS(h_counters.overflow_count == 0, "GPU range decomposition overflowed task storage");
    local_stats.exact_task_count      = static_cast<std::int64_t>(h_counters.exact_count);
    local_stats.graph_node_task_count = static_cast<std::int64_t>(h_counters.graph_node_count);
    local_stats.exact_vectors_scanned = static_cast<std::int64_t>(h_counters.exact_vec_count);
  }

  if (exact_launched) {
    float exact_milliseconds = 0.0f;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&exact_milliseconds, exact_start_event, exact_stop_event));
    local_stats.exact_seconds += static_cast<double>(exact_milliseconds) / 1000.0;
    RAFT_CUDA_TRY(cudaEventDestroy(exact_start_event));
    RAFT_CUDA_TRY(cudaEventDestroy(exact_stop_event));
  }
  if (decompose_done_event != nullptr) { RAFT_CUDA_TRY(cudaEventDestroy(decompose_done_event)); }
  if (graph_profile_enabled) {
    RAFT_CUDA_TRY(cudaMemcpy(&local_stats.graph_profile,
                             workspace.d_graph_profile,
                             sizeof(GraphSearchProfileCounters),
                             cudaMemcpyDeviceToHost));
  }

  if (stats != nullptr) { *stats = local_stats; }
}

}  // namespace cuvs::neighbors::range_cagra::detail
