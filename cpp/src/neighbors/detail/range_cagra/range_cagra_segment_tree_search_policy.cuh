/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_segment_tree_layering.cuh"

#include <raft/core/error.hpp>

#include <algorithm>
#include <utility>

namespace cuvs::neighbors::range_cagra::detail {

enum class SegmentTreeSearchIterationPolicy {
  kUniform,
  kLowerLayers,
  kUpperLayers,
  kLayerAdaptive,
};

struct SegmentTreeSearchIterationParams {
  SegmentTreeSearchIterationPolicy policy = SegmentTreeSearchIterationPolicy::kUniform;

  int lower_layer_count      = 0;
  int lower_layer_iterations = 0;

  int upper_layer_count      = 0;
  int upper_layer_iterations = 0;

  int adaptive_min_iterations = 0;
  int adaptive_max_iterations = 0;
  int adaptive_granularity    = 1;
};

[[nodiscard]] inline const char* search_iteration_policy_name(
  SegmentTreeSearchIterationPolicy policy)
{
  switch (policy) {
    case SegmentTreeSearchIterationPolicy::kLowerLayers: return "lower_layers";
    case SegmentTreeSearchIterationPolicy::kUpperLayers: return "upper_layers";
    case SegmentTreeSearchIterationPolicy::kLayerAdaptive: return "layer_adaptive";
    case SegmentTreeSearchIterationPolicy::kUniform:
    default: return "uniform";
  }
}

[[nodiscard]] inline int align_search_iterations(int iterations, int granularity)
{
  if (granularity <= 1) { return iterations; }
  return ((iterations + granularity - 1) / granularity) * granularity;
}

[[nodiscard]] inline int default_min_layer_iterations(int max_iterations)
{
  return max_iterations > 0 ? std::max(1, max_iterations / 4) : 0;
}

[[nodiscard]] inline int interpolate_layer_iterations(int min_iterations,
                                                      int max_iterations,
                                                      int layer_from_bottom,
                                                      int max_graph_layer,
                                                      int granularity)
{
  RAFT_EXPECTS(min_iterations >= 0, "min_iterations must be non-negative");
  RAFT_EXPECTS(max_iterations >= 0, "max_iterations must be non-negative");
  if (max_iterations < min_iterations) { std::swap(max_iterations, min_iterations); }
  if (max_graph_layer <= 0 || min_iterations == max_iterations) { return max_iterations; }

  layer_from_bottom = std::clamp(layer_from_bottom, 0, max_graph_layer);
  if (layer_from_bottom == 0) { return min_iterations; }
  if (layer_from_bottom == max_graph_layer) { return max_iterations; }

  const int delta      = max_iterations - min_iterations;
  const int iterations = min_iterations +
                         (delta * layer_from_bottom + max_graph_layer / 2) / max_graph_layer;
  return std::clamp(align_search_iterations(iterations, granularity),
                    min_iterations,
                    max_iterations);
}

[[nodiscard]] inline std::pair<int, int> layer_adaptive_iteration_bounds(
  int base_iterations, SegmentTreeSearchIterationParams const& params)
{
  RAFT_EXPECTS(base_iterations >= 0, "base_iterations must be non-negative");
  const int max_iterations = base_iterations;
  int min_iterations       = default_min_layer_iterations(max_iterations);

  if (params.adaptive_min_iterations > 0 && params.adaptive_max_iterations > 0) {
    int reference_min = params.adaptive_min_iterations;
    int reference_max = params.adaptive_max_iterations;
    if (reference_max < reference_min) { std::swap(reference_max, reference_min); }

    // Treat the supplied min/max as a reference ratio, e.g. 6/24 means that
    // leaf-layer graphs use 25% of the active config's graph_iterations.  The
    // root layer remains at base_iterations, matching layer-adaptive degree
    // where graph_degree is the current config's upper bound.
    min_iterations =
      (reference_min * max_iterations + reference_max / 2) / std::max(1, reference_max);
    if (max_iterations > 0) { min_iterations = std::max(1, min_iterations); }
  } else if (params.adaptive_min_iterations > 0) {
    min_iterations = params.adaptive_min_iterations;
  }

  min_iterations = std::min(min_iterations, max_iterations);
  return {min_iterations, max_iterations};
}

[[nodiscard]] inline bool graph_search_iteration_policy_needs_per_graph(
  SegmentTreeSearchIterationParams const& params, int base_iterations)
{
  switch (params.policy) {
    case SegmentTreeSearchIterationPolicy::kLowerLayers:
      return params.lower_layer_count > 0 && params.lower_layer_iterations != base_iterations;
    case SegmentTreeSearchIterationPolicy::kUpperLayers:
      return params.upper_layer_count > 0 && params.upper_layer_iterations != base_iterations;
    case SegmentTreeSearchIterationPolicy::kLayerAdaptive: {
      const auto [min_iterations, max_iterations] =
        layer_adaptive_iteration_bounds(base_iterations, params);
      return max_iterations != min_iterations;
    }
    case SegmentTreeSearchIterationPolicy::kUniform:
    default: return false;
  }
}

[[nodiscard]] inline int graph_search_iterations_for_layer(
  int layer_from_bottom,
  int max_graph_layer,
  int base_iterations,
  SegmentTreeSearchIterationParams const& params)
{
  RAFT_EXPECTS(base_iterations >= 0, "base_iterations must be non-negative");
  layer_from_bottom = std::clamp(layer_from_bottom, 0, std::max(0, max_graph_layer));

  switch (params.policy) {
    case SegmentTreeSearchIterationPolicy::kLowerLayers:
      if (params.lower_layer_count > 0 && layer_from_bottom < params.lower_layer_count) {
        return std::max(0, params.lower_layer_iterations);
      }
      return base_iterations;
    case SegmentTreeSearchIterationPolicy::kUpperLayers: {
      if (params.upper_layer_count <= 0) { return base_iterations; }
      const int first_upper_layer =
        std::max(0, max_graph_layer - params.upper_layer_count + 1);
      return layer_from_bottom >= first_upper_layer ? std::max(0, params.upper_layer_iterations)
                                                    : base_iterations;
    }
    case SegmentTreeSearchIterationPolicy::kLayerAdaptive: {
      const auto [min_iterations, max_iterations] =
        layer_adaptive_iteration_bounds(base_iterations, params);
      return interpolate_layer_iterations(
        min_iterations, max_iterations, layer_from_bottom, max_graph_layer, params.adaptive_granularity);
    }
    case SegmentTreeSearchIterationPolicy::kUniform:
    default: return base_iterations;
  }
}

}  // namespace cuvs::neighbors::range_cagra::detail
