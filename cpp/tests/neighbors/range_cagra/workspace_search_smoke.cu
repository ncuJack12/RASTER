/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/range_cagra/range_cagra_segment_tree_workspace_search.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {
namespace {

struct FloatMatrix {
  std::vector<float> data;
  std::int64_t rows = 0;
  int dim           = 0;
};

FloatMatrix make_host_dataset(int rows, int dim)
{
  FloatMatrix out;
  out.rows = rows;
  out.dim  = dim;
  out.data.resize(static_cast<std::size_t>(rows) * dim);
  for (int i = 0; i < rows; ++i) {
    for (int d = 0; d < dim; ++d) {
      out.data[static_cast<std::size_t>(i) * dim + d] =
        std::sin(0.013f * static_cast<float>(i * 17 + d * 3)) +
        0.01f * static_cast<float>((i + d) % 7);
    }
  }
  return out;
}

FloatMatrix make_host_queries(FloatMatrix const& base, int n_queries)
{
  FloatMatrix out;
  out.rows = n_queries;
  out.dim  = base.dim;
  out.data.resize(static_cast<std::size_t>(out.rows) * out.dim);
  for (int q = 0; q < n_queries; ++q) {
    const int src = (q * 37 + 11) % static_cast<int>(base.rows);
    for (int d = 0; d < out.dim; ++d) {
      out.data[static_cast<std::size_t>(q) * out.dim + d] =
        base.data[static_cast<std::size_t>(src) * base.dim + d] +
        0.001f * static_cast<float>((q + d) % 3);
    }
  }
  return out;
}

std::vector<std::int64_t> make_query_ranges(int rows, int leaf_size, int n_queries)
{
  std::vector<std::int64_t> ranges(static_cast<std::size_t>(n_queries) * 2);
  for (int q = 0; q < n_queries; ++q) {
    const int anchor = (q * 53 + 17) % rows;
    int width        = leaf_size * (3 + (q % 6)) + (q % 19);
    int left         = std::max(0, anchor - width / 3);
    int right        = std::min(rows - 1, left + width);
    left             = std::max(0, right - width);
    ranges[static_cast<std::size_t>(q) * 2]     = left;
    ranges[static_cast<std::size_t>(q) * 2 + 1] = right;
  }
  return ranges;
}

auto copy_to_device(raft::resources const& res, FloatMatrix const& matrix)
{
  auto out = raft::make_device_matrix<float, int64_t>(res, matrix.rows, matrix.dim);
  RAFT_CUDA_TRY(cudaMemcpyAsync(out.data_handle(),
                                matrix.data.data(),
                                sizeof(float) * matrix.data.size(),
                                cudaMemcpyHostToDevice,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);
  return out;
}

struct SearchRunSummary {
  double avg_search_seconds = 0.0;
  double avg_wall_seconds   = 0.0;
  SegmentTreeSearchStats stats;
  std::vector<std::uint32_t> ids;
  std::vector<float> distances;
};

bool same_results(SearchRunSummary const& lhs, SearchRunSummary const& rhs)
{
  if (lhs.ids != rhs.ids || lhs.distances.size() != rhs.distances.size()) { return false; }
  for (std::size_t i = 0; i < lhs.distances.size(); ++i) {
    const float a = lhs.distances[i];
    const float b = rhs.distances[i];
    if (std::isinf(a) || std::isinf(b)) {
      if (std::isinf(a) != std::isinf(b)) { return false; }
      continue;
    }
    if (std::abs(a - b) > 1e-5f) { return false; }
  }
  return true;
}

template <typename SearchF>
SearchRunSummary run_repeated(SearchF&& search, int repeats)
{
  SearchRunSummary out;
  double stats_total = 0.0;
  double wall_total  = 0.0;
  for (int r = 0; r < repeats; ++r) {
    SegmentTreeSearchStats stats;
    std::vector<std::uint32_t> ids;
    std::vector<float> distances;
    const auto t0 = std::chrono::steady_clock::now();
    search(ids, distances, stats);
    const auto t1 = std::chrono::steady_clock::now();
    stats_total += stats.search_seconds;
    wall_total += std::chrono::duration<double>(t1 - t0).count();
    if (r + 1 == repeats) {
      out.stats     = stats;
      out.ids       = std::move(ids);
      out.distances = std::move(distances);
    }
  }
  out.avg_search_seconds = stats_total / static_cast<double>(repeats);
  out.avg_wall_seconds   = wall_total / static_cast<double>(repeats);
  return out;
}

void print_summary(char const* mode,
                   SearchRunSummary const& summary,
                   int n_queries,
                   double baseline_seconds)
{
  const double qps     = static_cast<double>(n_queries) / summary.avg_search_seconds;
  const double speedup = baseline_seconds > 0.0 ? baseline_seconds / summary.avg_search_seconds : 1.0;
  std::cout << "workspace_smoke"
            << ",mode=" << mode
            << ",avg_search_seconds=" << summary.avg_search_seconds
            << ",avg_wall_seconds=" << summary.avg_wall_seconds
            << ",qps=" << qps
            << ",speedup_vs_baseline=" << speedup
            << ",exact_tasks=" << summary.stats.exact_task_count
            << ",graph_node_tasks=" << summary.stats.graph_node_task_count
            << ",exact_vectors_scanned=" << summary.stats.exact_vectors_scanned
            << ",exact_seconds=" << summary.stats.exact_seconds
            << ",graph_seconds=" << summary.stats.graph_seconds
            << ",merge_seconds=" << summary.stats.merge_seconds << '\n';
}

}  // namespace
}  // namespace cuvs::neighbors::range_cagra::detail

int main()
{
  using namespace cuvs::neighbors::range_cagra::detail;

  try {
    raft::resources res;

    constexpr int rows           = 4096;
    constexpr int dim            = 32;
    constexpr int leaf_size      = 128;
    constexpr int n_queries      = 512;
    constexpr int topk           = 10;
    constexpr int ef             = 64;
    constexpr int entry_count    = 32;
    constexpr int graph_iters    = 64;
    constexpr int exact_threads  = 128;
    constexpr int graph_threads  = 128;
    constexpr int search_width   = 1;
    constexpr int warmup_repeats = 1;
    constexpr int timed_repeats  = 5;

    auto base          = make_host_dataset(rows, dim);
    auto queries       = make_host_queries(base, n_queries);
    auto query_ranges  = make_query_ranges(rows, leaf_size, n_queries);
    auto d_base        = copy_to_device(res, base);
    auto d_queries     = copy_to_device(res, queries);
    auto queries_view  = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      d_queries.data_handle(), queries.rows, queries.dim);
    GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

    RangeGraphBuildParams build_params;
    build_params.graph_degree              = 16;
    build_params.intermediate_graph_degree = 32;
    build_params.nn_descent_iterations     = 3;

    const auto build_start = std::chrono::steady_clock::now();
    auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);
    const auto build_stop = std::chrono::steady_clock::now();
    const double build_wall_seconds = std::chrono::duration<double>(build_stop - build_start).count();

    auto baseline_search = [&](std::vector<std::uint32_t>& ids,
                               std::vector<float>& distances,
                               SegmentTreeSearchStats& stats) {
      run_segment_tree_range_cagra_search(res,
                                          dataset,
                                          index,
                                          queries_view,
                                          query_ranges,
                                          topk,
                                          ef,
                                          entry_count,
                                          graph_iters,
                                          exact_threads,
                                          ids,
                                          distances,
                                          &stats,
                                          search_width,
                                          false,
                                          graph_threads,
                                          SegmentTreeSearchSchedule::kOverlap);
    };

    SegmentTreeRangeCagraSearchWorkspace workspace;
    auto workspace_host_count_search = [&](std::vector<std::uint32_t>& ids,
                                           std::vector<float>& distances,
                                           SegmentTreeSearchStats& stats) {
      run_segment_tree_range_cagra_search_with_workspace(
        res,
        dataset,
        index,
        queries_view,
        query_ranges,
        topk,
        ef,
        entry_count,
        graph_iters,
        exact_threads,
        ids,
        distances,
        workspace,
        &stats,
        search_width,
        false,
        graph_threads,
        SegmentTreeSearchSchedule::kOverlap,
        SegmentTreeWorkspaceLaunchMode::kHostCountSync);
    };

    auto workspace_no_presync_search = [&](std::vector<std::uint32_t>& ids,
                                           std::vector<float>& distances,
                                           SegmentTreeSearchStats& stats) {
      run_segment_tree_range_cagra_search_with_workspace(
        res,
        dataset,
        index,
        queries_view,
        query_ranges,
        topk,
        ef,
        entry_count,
        graph_iters,
        exact_threads,
        ids,
        distances,
        workspace,
        &stats,
        search_width,
        false,
        graph_threads,
        SegmentTreeSearchSchedule::kOverlap,
        SegmentTreeWorkspaceLaunchMode::kDeviceCountNoPreSync);
    };

    (void)run_repeated(baseline_search, warmup_repeats);
    auto baseline = run_repeated(baseline_search, timed_repeats);

    (void)run_repeated(workspace_host_count_search, warmup_repeats);
    auto workspace_host = run_repeated(workspace_host_count_search, timed_repeats);

    (void)run_repeated(workspace_no_presync_search, warmup_repeats);
    auto workspace_no_presync = run_repeated(workspace_no_presync_search, timed_repeats);

    const bool host_matches     = same_results(baseline, workspace_host);
    const bool no_sync_matches  = same_results(baseline, workspace_no_presync);
    const int max_graph_tasks   = max_graph_tasks_per_query(index.layout) * n_queries;
    const int max_exact_tasks   = kRangeCagraMaxExactTasksPerQuery * n_queries;

    std::cout << "workspace_smoke_config"
              << ",rows=" << rows
              << ",dim=" << dim
              << ",nq=" << n_queries
              << ",leaf_size=" << leaf_size
              << ",graph_count=" << index.graph_pool.graph_count
              << ",edge_count=" << index.graph_pool.edge_count
              << ",build_seconds=" << index.build_seconds
              << ",build_wall_seconds=" << build_wall_seconds
              << ",max_exact_tasks=" << max_exact_tasks
              << ",max_graph_tasks=" << max_graph_tasks
              << ",timed_repeats=" << timed_repeats << '\n';

    print_summary("baseline", baseline, n_queries, baseline.avg_search_seconds);
    print_summary(
      "workspace_host_count", workspace_host, n_queries, baseline.avg_search_seconds);
    print_summary(
      "workspace_no_presync", workspace_no_presync, n_queries, baseline.avg_search_seconds);
    std::cout << "workspace_smoke_check"
              << ",host_count_matches=" << (host_matches ? 1 : 0)
              << ",no_presync_matches=" << (no_sync_matches ? 1 : 0) << '\n';

    if (!host_matches || !no_sync_matches) {
      std::cerr << "workspace smoke result mismatch\n";
      return 2;
    }
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "workspace smoke failed: " << e.what() << '\n';
    return 1;
  }
}
