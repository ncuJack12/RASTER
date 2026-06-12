/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_types.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuvs::neighbors::range_cagra::detail {

namespace kernels {

template <typename OutputIdxT>
RAFT_KERNEL add_rank_offset_kernel(OutputIdxT* ids, std::int64_t count, std::int64_t rank_l)
{
  const auto tid = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= count) { return; }

  const auto local_id = static_cast<std::uint64_t>(ids[tid]);
  if constexpr (std::is_unsigned_v<OutputIdxT>) {
    if (local_id == static_cast<std::uint64_t>(std::numeric_limits<OutputIdxT>::max())) { return; }
  }
  ids[tid] = static_cast<OutputIdxT>(static_cast<std::int64_t>(local_id) + rank_l);
}

}  // namespace kernels

[[nodiscard]] inline int cagra_required_float_stride(int dim)
{
  constexpr int floats_per_16b = 16 / static_cast<int>(sizeof(float));
  return ((dim + floats_per_16b - 1) / floats_per_16b) * floats_per_16b;
}

/**
 * Search one range graph without copying the underlying global dataset.
 *
 * This is the single-graph bridge used to validate range-search semantics before the multi-graph
 * fused kernel. The graph stores local ids, while returned neighbors are mapped to global ids by
 * adding graph.rank_l.
 */
template <typename OutputIdxT>
inline void search_one_range_graph(
  raft::resources const& res,
  cuvs::neighbors::cagra::search_params const& params,
  GlobalDatasetView const& dataset,
  DeviceRangeGraph const& graph,
  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  static_assert(std::is_integral_v<OutputIdxT>, "OutputIdxT must be an integer id type");
  RAFT_EXPECTS(dataset.base != nullptr, "GlobalDatasetView::base must not be null");
  RAFT_EXPECTS(dataset.rows > 0, "GlobalDatasetView::rows must be positive");
  RAFT_EXPECTS(dataset.dim > 0, "GlobalDatasetView::dim must be positive");
  RAFT_EXPECTS(dataset.stride >= dataset.dim, "GlobalDatasetView::stride must cover dim elements");
  RAFT_EXPECTS(graph.edges != nullptr, "DeviceRangeGraph::edges must not be null");
  RAFT_EXPECTS(graph.rows > 0, "DeviceRangeGraph::rows must be positive");
  RAFT_EXPECTS(graph.degree > 0, "DeviceRangeGraph::degree must be positive");
  RAFT_EXPECTS(graph.rank_l >= 0 && graph.rank_l + graph.rows <= dataset.rows,
               "range graph is outside the global dataset");
  RAFT_EXPECTS(queries.extent(1) == dataset.dim, "query dimension must match dataset dimension");
  RAFT_EXPECTS(neighbors.extent(0) == queries.extent(0), "neighbors row count must match queries");
  RAFT_EXPECTS(distances.extent(0) == queries.extent(0), "distances row count must match queries");
  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "neighbors and distances must have the same top-k width");

  const auto required_stride = cagra_required_float_stride(dataset.dim);
  RAFT_EXPECTS(dataset.stride == required_stride,
               "range_cagra search requires one globally padded/aligned dataset; passing an "
               "unaligned per-range view would let CAGRA allocate a padded copy");

  if constexpr (std::is_unsigned_v<OutputIdxT>) {
    const auto max_output = static_cast<std::uint64_t>(std::numeric_limits<OutputIdxT>::max());
    const auto max_id     = static_cast<std::uint64_t>(graph.rank_l + graph.rows - 1);
    RAFT_EXPECTS(max_id < max_output, "global ids must fit in OutputIdxT");
  }

  auto range_dataset_view = raft::make_device_strided_matrix_view<const float, int64_t>(
    dataset.row(graph.rank_l),
    static_cast<int64_t>(graph.rows),
    static_cast<int64_t>(dataset.dim),
    static_cast<int64_t>(dataset.stride));
  auto graph_view = raft::make_device_matrix_view<const std::uint32_t, int64_t, raft::row_major>(
    graph.edges, static_cast<int64_t>(graph.rows), static_cast<int64_t>(graph.degree));

  cuvs::neighbors::cagra::index<float, std::uint32_t> cagra_index(
    res, cuvs::distance::DistanceType::L2Expanded);
  cagra_index.update_dataset(res, range_dataset_view);
  cagra_index.update_graph(res, graph_view);
  cuvs::neighbors::cagra::search(res, params, cagra_index, queries, neighbors, distances);

  if (graph.rank_l != 0 && neighbors.size() > 0) {
    constexpr int block_size = 256;
    const auto count         = static_cast<std::int64_t>(neighbors.size());
    const auto grid_size     = static_cast<int>((count + block_size - 1) / block_size);
    kernels::
      add_rank_offset_kernel<<<grid_size, block_size, 0, raft::resource::get_cuda_stream(res)>>>(
        neighbors.data_handle(), count, graph.rank_l);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace cuvs::neighbors::range_cagra::detail
