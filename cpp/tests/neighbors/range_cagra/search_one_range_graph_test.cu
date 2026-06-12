/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/range_cagra/range_cagra_build.cuh"
#include "../../../src/neighbors/detail/range_cagra/range_cagra_search.cuh"

#include <cuvs/neighbors/cagra.hpp>

#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
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
        std::sin(0.017f * static_cast<float>(i * 11 + d * 5)) +
        0.03f * static_cast<float>((i + 3 * d) % 5);
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

auto make_test_queries(raft::resources const& res, int rows, int dim)
{
  std::vector<float> host(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim));
  for (int i = 0; i < rows; ++i) {
    for (int d = 0; d < dim; ++d) {
      host[static_cast<std::size_t>(i) * dim + d] =
        std::cos(0.019f * static_cast<float>(i * 13 + d * 7)) +
        0.02f * static_cast<float>((2 * i + d) % 7);
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

TEST(RangeCagraSearch, SearchOneRangeGraphMapsLocalIdsToGlobalIds)
{
  raft::resources res;

  constexpr int rows           = 96;
  constexpr int dim            = 16;
  constexpr int n_queries      = 8;
  constexpr int topk           = 5;
  constexpr std::int64_t left  = 16;
  constexpr std::int64_t right = 79;

  auto base    = make_test_dataset(res, rows, dim);
  auto queries = make_test_queries(res, n_queries, dim);
  auto query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    queries.data_handle(), n_queries, dim);

  GlobalDatasetView dataset{base.data_handle(), static_cast<std::int64_t>(rows), dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree              = 8;
  build_params.intermediate_graph_degree = 16;
  build_params.nn_descent_iterations     = 5;
  auto graph = build_one_range_graph(res, dataset, left, right, build_params);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size   = 32;
  search_params.search_width = 1;

  auto global_neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, n_queries, topk);
  auto global_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, topk);
  search_one_range_graph(res,
                         search_params,
                         dataset,
                         graph,
                         query_view,
                         global_neighbors.view(),
                         global_distances.view());

  auto range_dataset_view = raft::make_device_strided_matrix_view<const float, int64_t>(
    dataset.row(left), static_cast<int64_t>(graph.rows), static_cast<int64_t>(dataset.dim),
    static_cast<int64_t>(dataset.stride));
  auto graph_view = raft::make_device_matrix_view<const std::uint32_t, int64_t, raft::row_major>(
    graph.edges, static_cast<int64_t>(graph.rows), static_cast<int64_t>(graph.degree));
  cuvs::neighbors::cagra::index<float, std::uint32_t> cagra_index(
    res, cuvs::distance::DistanceType::L2Expanded);
  cagra_index.update_dataset(res, range_dataset_view);
  cagra_index.update_graph(res, graph_view);

  auto local_neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, n_queries, topk);
  auto local_distances = raft::make_device_matrix<float, int64_t>(res, n_queries, topk);
  cuvs::neighbors::cagra::search(
    res, search_params, cagra_index, query_view, local_neighbors.view(), local_distances.view());

  std::vector<std::uint32_t> global_ids(static_cast<std::size_t>(n_queries) * topk);
  std::vector<std::uint32_t> local_ids(static_cast<std::size_t>(n_queries) * topk);
  std::vector<float> global_dists(static_cast<std::size_t>(n_queries) * topk);
  std::vector<float> local_dists(static_cast<std::size_t>(n_queries) * topk);

  RAFT_CUDA_TRY(cudaMemcpyAsync(global_ids.data(),
                                global_neighbors.data_handle(),
                                sizeof(std::uint32_t) * global_ids.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  RAFT_CUDA_TRY(cudaMemcpyAsync(local_ids.data(),
                                local_neighbors.data_handle(),
                                sizeof(std::uint32_t) * local_ids.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  RAFT_CUDA_TRY(cudaMemcpyAsync(global_dists.data(),
                                global_distances.data_handle(),
                                sizeof(float) * global_dists.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  RAFT_CUDA_TRY(cudaMemcpyAsync(local_dists.data(),
                                local_distances.data_handle(),
                                sizeof(float) * local_dists.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);

  for (std::size_t i = 0; i < global_ids.size(); ++i) {
    EXPECT_EQ(global_ids[i], static_cast<std::uint32_t>(left + local_ids[i]));
    EXPECT_GE(global_ids[i], static_cast<std::uint32_t>(left));
    EXPECT_LE(global_ids[i], static_cast<std::uint32_t>(right));
    EXPECT_NEAR(global_dists[i], local_dists[i], 1e-5f);
  }
}

TEST(RangeCagraSearch, RejectsUnpaddedDatasetToAvoidPerRangeCopies)
{
  raft::resources res;

  constexpr int rows      = 16;
  constexpr int dim       = 10;
  constexpr int n_queries = 1;
  constexpr int topk      = 1;

  auto base    = make_test_dataset(res, rows, dim);
  auto queries = make_test_queries(res, n_queries, dim);
  auto query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    queries.data_handle(), n_queries, dim);

  std::uint32_t* edges = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&edges, sizeof(std::uint32_t) * 4));
  DeviceRangeGraph graph;
  graph.edges  = edges;
  graph.rank_l = 0;
  graph.rows   = 4;
  graph.degree = 1;

  GlobalDatasetView dataset{base.data_handle(), static_cast<std::int64_t>(rows), dim, dim};
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, n_queries, topk);
  auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, topk);

  cuvs::neighbors::cagra::search_params search_params;
  EXPECT_THROW(search_one_range_graph(res,
                                      search_params,
                                      dataset,
                                      graph,
                                      query_view,
                                      neighbors.view(),
                                      distances.view()),
               raft::logic_error);
}

}  // namespace
}  // namespace cuvs::neighbors::range_cagra::detail
