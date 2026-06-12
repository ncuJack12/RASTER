/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_types.cuh"

#include <cstdint>
#include <limits>

namespace cuvs::neighbors::range_cagra::detail {

constexpr int kRangeCagraBuildThreads              = 128;
constexpr int kRangeCagraBuildTeamSize             = 16;
constexpr int kRangeCagraBuildMaxDegree            = 128;
constexpr int kRangeCagraFlatBuildSamples          = 16;
constexpr int kRangeCagraSegmentedBuildSamples     = 32;
constexpr int kRangeCagraBuildMaxInitCandidates    = 512;
constexpr int kRangeCagraFlatLocalJoinThreads      = 256;
constexpr int kRangeCagraSegmentedLocalJoinThreads = 512;
constexpr int kRangeCagraSegmentSize               = 32;
constexpr int kRangeCagraMaxBuildSegments =
  (kRangeCagraBuildMaxDegree + kRangeCagraSegmentSize - 1) / kRangeCagraSegmentSize;
constexpr int kRangeCagraFlatGnndMaxBiSamples      = kRangeCagraFlatBuildSamples * 2;
constexpr int kRangeCagraSegmentedGnndMaxBiSamples = kRangeCagraSegmentedBuildSamples * 2;
constexpr int kRangeCagraGnndTileDim               = 32;
constexpr int kRangeCagraGnndTilePad               = 4;
constexpr int kRangeCagraOptimizeWarps             = 4;
constexpr int kRangeCagraWarpSize                  = 32;
constexpr std::uint32_t kRangeCagraBuildInvalidId = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kRangeCagraBuildOldFlag   = 0x80000000u;

template <int MaxBiSamples>
[[nodiscard]] constexpr int range_cagra_skewed_bi_samples()
{
  return MaxBiSamples + ((MaxBiSamples % kRangeCagraWarpSize == 0) ? 4 : 0);
}

[[nodiscard]] inline bool range_cagra_uses_segmented_build(RangeGraphBuildParams const& params)
{
  return params.build_algorithm == RangeGraphBuildAlgorithm::kSegmentedGnnd;
}

}  // namespace cuvs::neighbors::range_cagra::detail
