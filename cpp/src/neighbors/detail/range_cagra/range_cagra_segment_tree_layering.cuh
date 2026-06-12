/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_segment_tree_types.cuh"

#include <raft/core/error.hpp>

#include <cstdint>

namespace cuvs::neighbors::range_cagra::detail {

[[nodiscard]] inline int ceil_log2_i64(std::int64_t value)
{
  RAFT_EXPECTS(value > 0, "ceil_log2_i64 value must be positive");
  const auto uvalue = static_cast<std::uint64_t>(value);
  const int floor   = 63 - __builtin_clzll(uvalue);
  return ((std::uint64_t{1} << floor) == uvalue) ? floor : floor + 1;
}

[[nodiscard]] inline int segment_tree_max_internal_layer(SegmentTreeLayout const& layout)
{
  if (layout.leaf_base <= 2) { return 0; }
  return ceil_log2_i64(layout.leaf_base) - 1;
}

[[nodiscard]] inline int segment_node_layer_from_bottom(SegmentNodeRange const& range)
{
  if (!range.valid()) { return 0; }
  const auto block_count = range.block_r - range.block_l + 1;
  if (block_count <= 2) { return 0; }
  return ceil_log2_i64(block_count) - 1;
}

}  // namespace cuvs::neighbors::range_cagra::detail
