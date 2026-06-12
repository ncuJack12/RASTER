/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_segment_tree_layering.cuh"
#include "range_cagra_uniform_build_policy.cuh"

#include <raft/core/error.hpp>

#include <algorithm>

namespace cuvs::neighbors::range_cagra::detail {

[[nodiscard]] inline int align_layer_degree(int degree, int granularity)
{
  if (granularity <= 1) { return degree; }
  return ((degree + granularity - 1) / granularity) * granularity;
}

[[nodiscard]] inline int default_min_layer_degree(int max_degree)
{
  return std::max(1, max_degree / 4);
}

[[nodiscard]] inline int interpolate_layer_degree(int max_degree,
                                                  int min_degree,
                                                  int layer_from_bottom,
                                                  int max_layer,
                                                  int granularity)
{
  RAFT_EXPECTS(max_degree > 0, "max degree must be positive");
  RAFT_EXPECTS(min_degree > 0, "min degree must be positive");
  min_degree = std::min(min_degree, max_degree);
  if (max_layer <= 0) { return max_degree; }

  layer_from_bottom = std::clamp(layer_from_bottom, 0, max_layer);
  if (layer_from_bottom == 0) { return min_degree; }
  if (layer_from_bottom == max_layer) { return max_degree; }
  const int delta  = max_degree - min_degree;
  const int degree = min_degree + (delta * layer_from_bottom + max_layer / 2) / max_layer;
  return std::clamp(align_layer_degree(degree, granularity), min_degree, max_degree);
}

[[nodiscard]] inline RangeGraphNodeDegrees range_graph_adaptive_degrees_for_node(
  SegmentTreeLayout const& layout, SegmentNodeRange const& range, RangeGraphBuildParams const& params)
{
  RAFT_EXPECTS(params.graph_degree > 0, "graph_degree must be positive");
  RAFT_EXPECTS(params.intermediate_graph_degree > 0, "intermediate_graph_degree must be positive");
  RAFT_EXPECTS(params.degree_granularity >= 0, "degree_granularity must be non-negative");

  const int min_graph_degree =
    params.min_graph_degree > 0 ? params.min_graph_degree
                                : default_min_layer_degree(params.graph_degree);
  const int min_intermediate_graph_degree =
    params.min_intermediate_graph_degree > 0
      ? params.min_intermediate_graph_degree
      : std::max(min_graph_degree, default_min_layer_degree(params.intermediate_graph_degree));
  const int max_layer         = segment_tree_max_internal_layer(layout);
  const int layer_from_bottom = segment_node_layer_from_bottom(range);
  const int graph_degree      = interpolate_layer_degree(params.graph_degree,
                                                    min_graph_degree,
                                                    layer_from_bottom,
                                                    max_layer,
                                                    params.degree_granularity);
  const int intermediate_graph_degree =
    interpolate_layer_degree(params.intermediate_graph_degree,
                             min_intermediate_graph_degree,
                             layer_from_bottom,
                             max_layer,
                             params.degree_granularity);

  return clamp_range_graph_degrees_to_node(range, graph_degree, intermediate_graph_degree);
}

[[nodiscard]] inline RangeGraphNodeDegrees range_graph_degrees_for_node(
  SegmentTreeLayout const& layout, SegmentNodeRange const& range, RangeGraphBuildParams const& params)
{
  if (params.layer_adaptive_degree) {
    return range_graph_adaptive_degrees_for_node(layout, range, params);
  }
  return range_graph_uniform_degrees_for_node(range, params);
}

}  // namespace cuvs::neighbors::range_cagra::detail
