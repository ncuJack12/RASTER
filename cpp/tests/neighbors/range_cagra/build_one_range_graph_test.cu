/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/range_cagra/range_cagra_build.cuh"

#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {
namespace {

auto make_test_dataset(raft::resources const& res, int rows, int dim)
{
  std::vector<float> host(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim));
  for (int i = 0; i < rows; ++i) {
    for (int d = 0; d < dim; ++d) {
      host[static_cast<std::size_t>(i) * dim + d] =
        std::sin(0.013f * static_cast<float>(i * 17 + d * 3)) +
        0.01f * static_cast<float>((i + d) % 7);
    }
  }

  auto device = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  RAFT_CUDA_TRY(cudaMemcpyAsync(device.data_handle(),
                                host.data(),
                                sizeof(float) * host.size(),
                                cudaMemcpyHostToDevice,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);
  return device;
}

TEST(RangeCagraBuild, BuildsOneCompactRangeGraphWithLocalEdges)
{
  raft::resources res;

  constexpr int rows           = 96;
  constexpr int dim            = 16;
  constexpr std::int64_t left  = 16;
  constexpr std::int64_t right = 79;

  auto base = make_test_dataset(res, rows, dim);
  GlobalDatasetView dataset{base.data_handle(), static_cast<std::int64_t>(rows), dim, dim};

  RangeGraphBuildParams params;
  params.graph_degree              = 8;
  params.intermediate_graph_degree = 16;
  params.nn_descent_iterations     = 5;

  auto graph = build_one_range_graph(res, dataset, left, right, params);

  EXPECT_EQ(graph.rank_l, left);
  EXPECT_EQ(graph.rows, static_cast<int>(right - left + 1));
  EXPECT_GT(graph.degree, 0);
  EXPECT_LE(graph.degree, params.graph_degree);
  ASSERT_NE(graph.edges, nullptr);

  std::vector<std::uint32_t> edges(static_cast<std::size_t>(graph.edge_count()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(edges.data(),
                                graph.edges,
                                sizeof(std::uint32_t) * edges.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);

  for (auto edge : edges) {
    EXPECT_LT(edge, static_cast<std::uint32_t>(graph.rows));
  }
}

TEST(RangeCagraBuild, BuildsOneSegmentedGnndRangeGraph)
{
  raft::resources res;

  constexpr int rows           = 160;
  constexpr int dim            = 32;
  constexpr std::int64_t left  = 16;
  constexpr std::int64_t right = 143;

  auto base = make_test_dataset(res, rows, dim);
  GlobalDatasetView dataset{base.data_handle(), static_cast<std::int64_t>(rows), dim, dim};

  RangeGraphBuildParams params;
  params.graph_degree              = 16;
  params.intermediate_graph_degree = 64;
  params.nn_descent_iterations     = 4;
  params.build_algorithm           = RangeGraphBuildAlgorithm::kSegmentedGnnd;

  auto graph = build_one_range_graph(res, dataset, left, right, params);

  EXPECT_EQ(graph.rank_l, left);
  EXPECT_EQ(graph.rows, static_cast<int>(right - left + 1));
  EXPECT_EQ(graph.degree, params.graph_degree);
  ASSERT_NE(graph.edges, nullptr);

  std::vector<std::uint32_t> edges(static_cast<std::size_t>(graph.edge_count()));
  RAFT_CUDA_TRY(cudaMemcpyAsync(edges.data(),
                                graph.edges,
                                sizeof(std::uint32_t) * edges.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);

  for (auto edge : edges) {
    EXPECT_LT(edge, static_cast<std::uint32_t>(graph.rows));
  }
}

TEST(RangeCagraBuild, RejectsStridedRangeUntilRangeAwareBuildIsAdded)
{
  raft::resources res;

  constexpr int rows = 96;
  constexpr int dim  = 16;
  auto base          = make_test_dataset(res, rows, dim);
  GlobalDatasetView dataset{base.data_handle(), static_cast<std::int64_t>(rows), dim, dim + 1};

  RangeGraphBuildParams params;
  EXPECT_THROW(build_one_range_graph(res, dataset, 16, 79, params), raft::logic_error);
}

}  // namespace
}  // namespace cuvs::neighbors::range_cagra::detail
