/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_segment_tree_types.cuh"
#include "range_cagra_types.cuh"

#include <raft/core/error.hpp>

#include <algorithm>

namespace cuvs::neighbors::range_cagra::detail {

struct RangeGraphNodeDegrees {
  int graph_degree              = 0;
  int intermediate_graph_degree = 0;
};

[[nodiscard]] inline RangeGraphNodeDegrees clamp_range_graph_degrees_to_node(
  SegmentNodeRange const& range, int graph_degree, int intermediate_graph_degree)
{
  const auto rows = static_cast<int>(range.size());
  RAFT_EXPECTS(rows > 1, "range graph node must cover at least two rows");

  graph_degree              = std::min(graph_degree, rows - 1);
  intermediate_graph_degree = std::min(std::max(intermediate_graph_degree, graph_degree), rows - 1);
  return {graph_degree, intermediate_graph_degree};
}

[[nodiscard]] inline RangeGraphNodeDegrees range_graph_uniform_degrees_for_node(
  SegmentNodeRange const& range, RangeGraphBuildParams const& params)
{
  RAFT_EXPECTS(params.graph_degree > 0, "graph_degree must be positive");
  RAFT_EXPECTS(params.intermediate_graph_degree > 0, "intermediate_graph_degree must be positive");
  return clamp_range_graph_degrees_to_node(
    range, params.graph_degree, params.intermediate_graph_degree);
}

}  // namespace cuvs::neighbors::range_cagra::detail
