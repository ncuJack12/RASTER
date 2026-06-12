/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::range_cagra::detail {

struct GlobalDatasetView {
  const float* base = nullptr;
  std::int64_t rows = 0;
  int dim           = 0;
  int stride        = 0;

  __host__ __device__ const float* row(std::int64_t global_id) const
  {
    return base + global_id * static_cast<std::int64_t>(stride);
  }
};

enum class RangeGraphBuildAlgorithm : int {
  kFlatGnnd      = 0,
  kSegmentedGnnd = 1,
};

struct RangeGraphBuildParams {
  int graph_degree                         = 64;
  int intermediate_graph_degree            = 128;
  int nn_descent_iterations                = 20;
  bool guarantee_connectivity              = false;
  RangeGraphBuildAlgorithm build_algorithm = RangeGraphBuildAlgorithm::kFlatGnnd;
  bool layer_adaptive_degree               = false;
  int min_graph_degree                     = 0;
  int min_intermediate_graph_degree        = 0;
  int degree_granularity                   = 8;
};

struct DeviceRangeGraph {
  std::uint32_t* edges = nullptr;
  std::int64_t rank_l  = 0;
  int rows             = 0;
  int degree           = 0;

  DeviceRangeGraph()                        = default;
  DeviceRangeGraph(const DeviceRangeGraph&) = delete;
  auto operator=(const DeviceRangeGraph&) -> DeviceRangeGraph& = delete;

  DeviceRangeGraph(DeviceRangeGraph&& other) noexcept
    : edges(other.edges), rank_l(other.rank_l), rows(other.rows), degree(other.degree)
  {
    other.edges  = nullptr;
    other.rank_l = 0;
    other.rows   = 0;
    other.degree = 0;
  }

  auto operator=(DeviceRangeGraph&& other) noexcept -> DeviceRangeGraph&
  {
    if (this != &other) {
      release();
      edges        = other.edges;
      rank_l       = other.rank_l;
      rows         = other.rows;
      degree       = other.degree;
      other.edges  = nullptr;
      other.rank_l = 0;
      other.rows   = 0;
      other.degree = 0;
    }
    return *this;
  }

  ~DeviceRangeGraph() { release(); }

  void release() noexcept
  {
    if (edges != nullptr) {
      cudaFree(edges);
      edges = nullptr;
    }
    rank_l = 0;
    rows   = 0;
    degree = 0;
  }

  [[nodiscard]] std::int64_t edge_count() const
  {
    return static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(degree);
  }
};

struct DeviceRangeGraphPool {
  std::uint32_t* edges    = nullptr;
  std::int64_t* offsets   = nullptr;
  std::int64_t* rank_l    = nullptr;
  int* rows               = nullptr;
  int* degrees            = nullptr;
  int graph_count         = 0;
  std::int64_t edge_count = 0;

  DeviceRangeGraphPool()                            = default;
  DeviceRangeGraphPool(DeviceRangeGraphPool const&) = delete;
  auto operator=(DeviceRangeGraphPool const&) -> DeviceRangeGraphPool& = delete;

  DeviceRangeGraphPool(DeviceRangeGraphPool&& other) noexcept
    : edges(other.edges),
      offsets(other.offsets),
      rank_l(other.rank_l),
      rows(other.rows),
      degrees(other.degrees),
      graph_count(other.graph_count),
      edge_count(other.edge_count)
  {
    other.edges       = nullptr;
    other.offsets     = nullptr;
    other.rank_l      = nullptr;
    other.rows        = nullptr;
    other.degrees     = nullptr;
    other.graph_count = 0;
    other.edge_count  = 0;
  }

  auto operator=(DeviceRangeGraphPool&& other) noexcept -> DeviceRangeGraphPool&
  {
    if (this != &other) {
      release();
      edges             = other.edges;
      offsets           = other.offsets;
      rank_l            = other.rank_l;
      rows              = other.rows;
      degrees           = other.degrees;
      graph_count       = other.graph_count;
      edge_count        = other.edge_count;
      other.edges       = nullptr;
      other.offsets     = nullptr;
      other.rank_l      = nullptr;
      other.rows        = nullptr;
      other.degrees     = nullptr;
      other.graph_count = 0;
      other.edge_count  = 0;
    }
    return *this;
  }

  ~DeviceRangeGraphPool() { release(); }

  void release() noexcept
  {
    if (edges != nullptr) { cudaFree(edges); }
    if (offsets != nullptr) { cudaFree(offsets); }
    if (rank_l != nullptr) { cudaFree(rank_l); }
    if (rows != nullptr) { cudaFree(rows); }
    if (degrees != nullptr) { cudaFree(degrees); }
    edges       = nullptr;
    offsets     = nullptr;
    rank_l      = nullptr;
    rows        = nullptr;
    degrees     = nullptr;
    graph_count = 0;
    edge_count  = 0;
  }
};

struct DeviceRangeGraphPoolView {
  std::uint32_t* edges        = nullptr;
  const std::int64_t* offsets = nullptr;
  const std::int64_t* rank_l  = nullptr;
  const int* rows             = nullptr;
  const int* degrees          = nullptr;
  int graph_count             = 0;
  std::int64_t edge_count     = 0;
};

[[nodiscard]] inline DeviceRangeGraphPoolView view(DeviceRangeGraphPool const& pool)
{
  return {pool.edges,
          pool.offsets,
          pool.rank_l,
          pool.rows,
          pool.degrees,
          pool.graph_count,
          pool.edge_count};
}

}  // namespace cuvs::neighbors::range_cagra::detail
