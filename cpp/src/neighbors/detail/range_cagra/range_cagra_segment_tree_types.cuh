/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {

struct SegmentNodeRange {
  std::int64_t block_l = 0;
  std::int64_t block_r = -1;
  std::int64_t vec_l   = 0;
  std::int64_t vec_r   = -1;

  [[nodiscard]] __host__ __device__ bool valid() const { return vec_l <= vec_r; }
  [[nodiscard]] __host__ __device__ std::int64_t size() const
  {
    return valid() ? vec_r - vec_l + 1 : 0;
  }
};

struct SegmentTreeLayout {
  std::int64_t rows        = 0;
  int dim                  = 0;
  int leaf_size            = 0;
  std::int64_t leaf_blocks = 0;
  std::int64_t leaf_base   = 0;
  std::vector<int> graph_slot;
};

}  // namespace cuvs::neighbors::range_cagra::detail
