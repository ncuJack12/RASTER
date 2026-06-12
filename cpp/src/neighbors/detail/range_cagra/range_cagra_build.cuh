/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "range_cagra_build_config.cuh"
#include "range_cagra_types.cuh"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {

namespace build_kernels {

__device__ inline std::uint32_t hash_u32(std::uint32_t value)
{
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

__device__ inline int graph_id_from_flat_row(const std::int64_t* __restrict__ row_offsets,
                                             int graph_count,
                                             std::int64_t flat_row)
{
  int lo = 0;
  int hi = graph_count;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (row_offsets[mid] <= flat_row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ inline int graph_id_from_flat_edge(DeviceRangeGraphPoolView graph_pool,
                                              std::int64_t flat_edge)
{
  int lo = 0;
  int hi = graph_pool.graph_count;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (graph_pool.offsets[mid] <= flat_edge) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ inline bool better_build_pair(float lhs_dist,
                                         std::uint32_t lhs_id,
                                         float rhs_dist,
                                         std::uint32_t rhs_id)
{
  return lhs_dist < rhs_dist || (lhs_dist == rhs_dist && lhs_id < rhs_id);
}

__device__ inline std::uint32_t build_local_id(std::uint32_t id) { return id & 0x7fffffffu; }

__device__ inline bool build_is_new(std::uint32_t id)
{
  return id != kRangeCagraBuildInvalidId && ((id & kRangeCagraBuildOldFlag) == 0);
}

__device__ inline std::uint32_t build_mark_old(std::uint32_t id)
{
  return build_local_id(id) | kRangeCagraBuildOldFlag;
}

__device__ inline std::uint32_t warp_reduce_min_u32(std::uint32_t value)
{
#pragma unroll
  for (int offset = kRangeCagraWarpSize >> 1; offset > 0; offset >>= 1) {
    value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ inline void insert_build_candidate(float dist,
                                              std::uint32_t id,
                                              std::uint32_t self,
                                              float* __restrict__ best_dist,
                                              std::uint32_t* __restrict__ best_id,
                                              int degree)
{
  if (id == self || !(dist < INFINITY)) { return; }
  for (int i = 0; i < degree; ++i) {
    if (best_id[i] == id) {
      if (dist < best_dist[i]) { best_dist[i] = dist; }
      return;
    }
  }
  if (!better_build_pair(dist, id, best_dist[degree - 1], best_id[degree - 1])) { return; }

  int pos = degree - 1;
  while (pos > 0 && better_build_pair(dist, id, best_dist[pos - 1], best_id[pos - 1])) {
    best_dist[pos] = best_dist[pos - 1];
    best_id[pos]   = best_id[pos - 1];
    --pos;
  }
  best_dist[pos] = dist;
  best_id[pos]   = id;
}

__device__ inline std::uint32_t initial_candidate(
  std::uint32_t self, int rows, int cand_idx, int cand_count, int graph_id, std::uint64_t seed)
{
  if (rows <= 1) { return std::numeric_limits<std::uint32_t>::max(); }
  if (cand_count >= rows - 1) {
    return static_cast<std::uint32_t>(cand_idx < static_cast<int>(self) ? cand_idx : cand_idx + 1);
  }

  constexpr int local_window = 32;
  if (cand_idx < local_window) {
    const std::uint32_t radius = static_cast<std::uint32_t>((cand_idx >> 1) + 1);
    const auto rows_u          = static_cast<std::uint32_t>(rows);
    if ((cand_idx & 1) == 0) { return (self + radius) % rows_u; }
    return (self + rows_u - (radius % rows_u)) % rows_u;
  }

  const std::uint32_t h =
    hash_u32(static_cast<std::uint32_t>(seed) ^ hash_u32(self + 0x9e3779b9u * cand_idx) ^
             hash_u32(static_cast<std::uint32_t>(graph_id) * 0x85ebca6bu));
  std::uint32_t out = h % static_cast<std::uint32_t>(rows);
  if (out == self) { out = (out + 1) % static_cast<std::uint32_t>(rows); }
  return out;
}

__device__ inline std::uint32_t segmented_initial_candidate(
  std::uint32_t self, int rows, int slot, int degree, int graph_id, std::uint64_t seed)
{
  if (rows <= 1 || degree <= 0) { return kRangeCagraBuildInvalidId; }
  const int num_segments = max(1, (degree + kRangeCagraSegmentSize - 1) / kRangeCagraSegmentSize);
  const int seg_idx      = min(slot / kRangeCagraSegmentSize, num_segments - 1);
  const int seg_slot     = slot - seg_idx * kRangeCagraSegmentSize;
  const int seg_rows     = (rows + num_segments - 1 - seg_idx) / num_segments;
  if (seg_rows <= 0) { return kRangeCagraBuildInvalidId; }

  const std::uint32_t h0 =
    hash_u32(static_cast<std::uint32_t>(seed) ^ hash_u32(self + 0x9e3779b9u * (seg_idx + 1)) ^
             hash_u32(static_cast<std::uint32_t>(graph_id) * 0x85ebca6bu));
  std::uint32_t step = hash_u32(h0 ^ 0x27d4eb2du) % static_cast<std::uint32_t>(seg_rows);
  step               = (step | 1u);
  if (step >= static_cast<std::uint32_t>(seg_rows)) { step = 1; }
  const std::uint32_t pos =
    (h0 + static_cast<std::uint32_t>(seg_slot) * step) % static_cast<std::uint32_t>(seg_rows);
  auto out = pos * static_cast<std::uint32_t>(num_segments) + static_cast<std::uint32_t>(seg_idx);
  if (out >= static_cast<std::uint32_t>(rows)) { out = static_cast<std::uint32_t>(seg_idx); }
  if (out == self) {
    const auto next_pos = (pos + 1) % static_cast<std::uint32_t>(seg_rows);
    out = next_pos * static_cast<std::uint32_t>(num_segments) + static_cast<std::uint32_t>(seg_idx);
    if (out >= static_cast<std::uint32_t>(rows) || out == self) {
      out = (self + 1) % static_cast<std::uint32_t>(rows);
    }
  }
  return out;
}

__device__ inline float team_l2(GlobalDatasetView dataset,
                                std::int64_t rank_l,
                                std::uint32_t lhs_local,
                                std::uint32_t rhs_local,
                                bool valid)
{
  constexpr int team_size = kRangeCagraBuildTeamSize;
  const int lane          = threadIdx.x & (team_size - 1);
  float sum               = 0.0f;
  if (valid) {
    const float* lhs = dataset.row(rank_l + static_cast<std::int64_t>(lhs_local));
    const float* rhs = dataset.row(rank_l + static_cast<std::int64_t>(rhs_local));
    for (int d = lane; d < dataset.dim; d += team_size) {
      const float diff = lhs[d] - rhs[d];
      sum += diff * diff;
    }
  }

#pragma unroll
  for (int offset = team_size >> 1; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset, team_size);
  }
  return sum;
}

__device__ inline void warp_reduce_best_pair(float& dist, std::uint32_t& id)
{
#pragma unroll
  for (int offset = kRangeCagraWarpSize >> 1; offset > 0; offset >>= 1) {
    const float other_dist = __shfl_down_sync(0xffffffffu, dist, offset);
    const auto other_id    = __shfl_down_sync(0xffffffffu, id, offset);
    if (better_build_pair(other_dist, other_id, dist, id)) {
      dist = other_dist;
      id   = other_id;
    }
  }
}

__device__ inline void insert_global_build_candidate(DeviceRangeGraphPoolView graph_pool,
                                                     float* __restrict__ graph_dists,
                                                     int* __restrict__ row_locks,
                                                     const std::int64_t* __restrict__ row_offsets,
                                                     int graph_id,
                                                     std::uint32_t row,
                                                     std::uint32_t candidate,
                                                     float dist)
{
  const int rows = graph_pool.rows[graph_id];
  if (row >= static_cast<std::uint32_t>(rows) || candidate >= static_cast<std::uint32_t>(rows) ||
      row == candidate || !(dist < INFINITY)) {
    return;
  }

  const int degree    = graph_pool.degrees[graph_id];
  const auto row_off  = graph_pool.offsets[graph_id] + static_cast<std::int64_t>(row) * degree;
  const auto flat_row = row_offsets[graph_id] + static_cast<std::int64_t>(row);

  while (atomicCAS(row_locks + flat_row, 0, 1) != 0) {}

  for (int i = 0; i < degree; ++i) {
    const auto existing = build_local_id(graph_pool.edges[row_off + i]);
    if (existing == candidate) {
      if (dist < graph_dists[row_off + i]) { graph_dists[row_off + i] = dist; }
      __threadfence();
      atomicExch(row_locks + flat_row, 0);
      return;
    }
  }

  if (!better_build_pair(dist,
                         candidate,
                         graph_dists[row_off + degree - 1],
                         build_local_id(graph_pool.edges[row_off + degree - 1]))) {
    __threadfence();
    atomicExch(row_locks + flat_row, 0);
    return;
  }

  int pos = degree - 1;
  while (pos > 0 && better_build_pair(dist,
                                      candidate,
                                      graph_dists[row_off + pos - 1],
                                      build_local_id(graph_pool.edges[row_off + pos - 1]))) {
    graph_pool.edges[row_off + pos] = graph_pool.edges[row_off + pos - 1];
    graph_dists[row_off + pos]      = graph_dists[row_off + pos - 1];
    --pos;
  }
  graph_pool.edges[row_off + pos] = candidate;
  graph_dists[row_off + pos]      = dist;

  __threadfence();
  atomicExch(row_locks + flat_row, 0);
}

__device__ inline void insert_segmented_build_candidate(
  DeviceRangeGraphPoolView graph_pool,
  float* __restrict__ graph_dists,
  int* __restrict__ row_locks,
  const std::int64_t* __restrict__ row_offsets,
  int graph_id,
  std::uint32_t row,
  std::uint32_t candidate,
  float dist)
{
  const int rows = graph_pool.rows[graph_id];
  if (row >= static_cast<std::uint32_t>(rows) || candidate >= static_cast<std::uint32_t>(rows) ||
      row == candidate || !(dist < INFINITY)) {
    return;
  }

  const int degree       = graph_pool.degrees[graph_id];
  const int num_segments = max(1, (degree + kRangeCagraSegmentSize - 1) / kRangeCagraSegmentSize);
  const int seg_idx      = static_cast<int>(candidate % static_cast<std::uint32_t>(num_segments));
  const int seg_begin    = seg_idx * kRangeCagraSegmentSize;
  const int seg_end      = min(seg_begin + kRangeCagraSegmentSize, degree);
  if (seg_begin >= seg_end) { return; }

  const auto row_off  = graph_pool.offsets[graph_id] + static_cast<std::int64_t>(row) * degree;
  const auto flat_row = row_offsets[graph_id] + static_cast<std::int64_t>(row);
  const int lock_idx =
    static_cast<int>(flat_row) *
      ((kRangeCagraBuildMaxDegree + kRangeCagraSegmentSize - 1) / kRangeCagraSegmentSize) +
    seg_idx;

  while (atomicCAS(row_locks + lock_idx, 0, 1) != 0) {}

  for (int i = seg_begin; i < seg_end; ++i) {
    const auto existing = build_local_id(graph_pool.edges[row_off + i]);
    if (existing == candidate) {
      if (dist < graph_dists[row_off + i]) { graph_dists[row_off + i] = dist; }
      __threadfence();
      atomicExch(row_locks + lock_idx, 0);
      return;
    }
  }

  if (!better_build_pair(dist,
                         candidate,
                         graph_dists[row_off + seg_end - 1],
                         build_local_id(graph_pool.edges[row_off + seg_end - 1]))) {
    __threadfence();
    atomicExch(row_locks + lock_idx, 0);
    return;
  }

  int pos = seg_end - 1;
  while (pos > seg_begin &&
         better_build_pair(dist,
                           candidate,
                           graph_dists[row_off + pos - 1],
                           build_local_id(graph_pool.edges[row_off + pos - 1]))) {
    graph_pool.edges[row_off + pos] = graph_pool.edges[row_off + pos - 1];
    graph_dists[row_off + pos]      = graph_dists[row_off + pos - 1];
    --pos;
  }
  graph_pool.edges[row_off + pos] = candidate;
  graph_dists[row_off + pos]      = dist;

  __threadfence();
  atomicExch(row_locks + lock_idx, 0);
}

template <int MaxBiSamples>
__device__ inline void append_unique_build_sample(std::uint32_t* __restrict__ list,
                                                  int& size,
                                                  std::uint32_t id,
                                                  int rows)
{
  id = build_local_id(id);
  if (id >= static_cast<std::uint32_t>(rows)) { return; }
  for (int i = 0; i < size; ++i) {
    if (list[i] == id) { return; }
  }
  if (size < MaxBiSamples) {
    list[size] = id;
    ++size;
  }
}

RAFT_KERNEL range_graph_init_kernel(GlobalDatasetView dataset,
                                    DeviceRangeGraphPoolView graph_pool,
                                    float* __restrict__ graph_dists,
                                    const std::int64_t* __restrict__ row_offsets,
                                    const int* __restrict__ init_candidate_counts,
                                    std::int64_t total_rows,
                                    std::uint64_t seed)
{
  const std::int64_t flat_row = static_cast<std::int64_t>(blockIdx.x);
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i  = flat_row - row_offsets[graph_id];
  const auto self                 = static_cast<std::uint32_t>(local_row_i);
  const int rows                  = graph_pool.rows[graph_id];
  const int degree                = graph_pool.degrees[graph_id];
  const int init_candidate_count  = init_candidate_counts[graph_id];
  const std::int64_t rank_l       = graph_pool.rank_l[graph_id];
  const std::int64_t edge_row_off = graph_pool.offsets[graph_id] + local_row_i * degree;

  extern __shared__ unsigned char raw_shared[];
  float* best_dist      = reinterpret_cast<float*>(raw_shared);
  auto* best_id         = reinterpret_cast<std::uint32_t*>(best_dist + kRangeCagraBuildMaxDegree);
  float* candidate_dist = reinterpret_cast<float*>(best_id + kRangeCagraBuildMaxDegree);
  auto* candidate_id =
    reinterpret_cast<std::uint32_t*>(candidate_dist + (blockDim.x / kRangeCagraBuildTeamSize));
  constexpr int team_size  = kRangeCagraBuildTeamSize;
  const int team_id        = threadIdx.x / team_size;
  const int team_lane      = threadIdx.x & (team_size - 1);
  const int teams_per_cta  = blockDim.x / team_size;
  const int candidate_stop = min(rows - 1, init_candidate_count);

  for (int i = threadIdx.x; i < degree; i += blockDim.x) {
    best_dist[i] = INFINITY;
    best_id[i]   = std::numeric_limits<std::uint32_t>::max();
  }
  __syncthreads();

  for (int base = 0; base < candidate_stop; base += teams_per_cta) {
    const int cand_idx = base + team_id;
    bool valid         = cand_idx < candidate_stop;
    std::uint32_t cand = std::numeric_limits<std::uint32_t>::max();
    if (valid) {
      cand  = initial_candidate(self, rows, cand_idx, candidate_stop, graph_id, seed);
      valid = cand < static_cast<std::uint32_t>(rows) && cand != self;
    }
    const float dist = team_l2(dataset, rank_l, self, cand, valid);
    if (team_lane == 0) {
      candidate_id[team_id]   = valid ? cand : std::numeric_limits<std::uint32_t>::max();
      candidate_dist[team_id] = valid ? dist : INFINITY;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const int group_count = min(teams_per_cta, candidate_stop - base);
      for (int i = 0; i < group_count; ++i) {
        insert_build_candidate(
          candidate_dist[i], candidate_id[i], self, best_dist, best_id, degree);
      }
    }
    __syncthreads();
  }

  for (int i = threadIdx.x; i < degree; i += blockDim.x) {
    auto id   = best_id[i];
    auto dist = best_dist[i];
    if (id >= static_cast<std::uint32_t>(rows)) {
      id = static_cast<std::uint32_t>((static_cast<std::int64_t>(self) + i + 1) % rows);
      if (id == self) { id = (id + 1) % static_cast<std::uint32_t>(rows); }
      dist = INFINITY;
    }
    graph_pool.edges[edge_row_off + i] = id;
    graph_dists[edge_row_off + i]      = dist;
  }
}

RAFT_KERNEL range_graph_segmented_init_kernel(DeviceRangeGraphPoolView graph_pool,
                                              float* __restrict__ graph_dists,
                                              const std::int64_t* __restrict__ row_offsets,
                                              std::int64_t total_rows,
                                              std::uint64_t seed)
{
  const std::int64_t flat_row = static_cast<std::int64_t>(blockIdx.x);
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i  = flat_row - row_offsets[graph_id];
  const auto self                 = static_cast<std::uint32_t>(local_row_i);
  const int rows                  = graph_pool.rows[graph_id];
  const int degree                = graph_pool.degrees[graph_id];
  const std::int64_t edge_row_off = graph_pool.offsets[graph_id] + local_row_i * degree;

  for (int slot = threadIdx.x; slot < degree; slot += blockDim.x) {
    auto id = segmented_initial_candidate(self, rows, slot, degree, graph_id, seed);
    if (id >= static_cast<std::uint32_t>(rows) || id == self) {
      id = static_cast<std::uint32_t>((static_cast<std::int64_t>(self) + slot + 1) % rows);
      if (id == self) { id = (id + 1) % static_cast<std::uint32_t>(rows); }
    }
    graph_pool.edges[edge_row_off + slot] = id;
    graph_dists[edge_row_off + slot]      = INFINITY;
  }
}

RAFT_KERNEL range_graph_sample_old_new_kernel(DeviceRangeGraphPoolView graph_pool,
                                              DeviceRangeGraphPoolView new_pool,
                                              DeviceRangeGraphPoolView old_pool,
                                              int* __restrict__ new_counts,
                                              int* __restrict__ old_counts,
                                              const std::int64_t* __restrict__ row_offsets,
                                              std::int64_t total_rows,
                                              int sample_count,
                                              int iteration)
{
  const std::int64_t flat_row = static_cast<std::int64_t>(blockIdx.x);
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i  = flat_row - row_offsets[graph_id];
  const auto self                 = static_cast<std::uint32_t>(local_row_i);
  const int rows                  = graph_pool.rows[graph_id];
  const int degree                = graph_pool.degrees[graph_id];
  const std::int64_t edge_row_off = graph_pool.offsets[graph_id] + local_row_i * degree;
  const int sample_degree         = new_pool.degrees[graph_id];
  const auto sample_row_off       = new_pool.offsets[graph_id] + local_row_i * sample_degree;
  const auto old_sample_row_off   = old_pool.offsets[graph_id] + local_row_i * sample_degree;

  if (threadIdx.x == 0) {
    int new_count = 0;
    int old_count = 0;
    for (int i = 0; i < sample_degree; ++i) {
      new_pool.edges[sample_row_off + i]     = kRangeCagraBuildInvalidId;
      old_pool.edges[old_sample_row_off + i] = kRangeCagraBuildInvalidId;
    }

    const int start = degree == 0 ? 0
                                  : static_cast<int>((hash_u32(self ^ (iteration * 0x9e3779b9u)) %
                                                      static_cast<std::uint32_t>(degree)));
    for (int scan = 0; scan < degree && (new_count < sample_count || old_count < sample_count);
         ++scan) {
      const int slot = (start + scan) % degree;
      const auto raw = graph_pool.edges[edge_row_off + slot];
      const auto id  = build_local_id(raw);
      if (id >= static_cast<std::uint32_t>(rows) || id == self) { continue; }
      if (build_is_new(raw)) {
        if (new_count < sample_count) {
          new_pool.edges[sample_row_off + new_count] = id;
          ++new_count;
          graph_pool.edges[edge_row_off + slot] = build_mark_old(id);
        }
      } else if (old_count < sample_count) {
        old_pool.edges[old_sample_row_off + old_count] = id;
        ++old_count;
      }
    }
    new_counts[flat_row] = new_count;
    old_counts[flat_row] = old_count;
  }
}

RAFT_KERNEL range_graph_segmented_sample_old_new_kernel(
  DeviceRangeGraphPoolView graph_pool,
  DeviceRangeGraphPoolView new_pool,
  DeviceRangeGraphPoolView old_pool,
  int* __restrict__ new_counts,
  int* __restrict__ old_counts,
  const std::int64_t* __restrict__ row_offsets,
  std::int64_t total_rows,
  int sample_count)
{
  const std::int64_t flat_row = static_cast<std::int64_t>(blockIdx.x);
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i  = flat_row - row_offsets[graph_id];
  const auto self                 = static_cast<std::uint32_t>(local_row_i);
  const int rows                  = graph_pool.rows[graph_id];
  const int degree                = graph_pool.degrees[graph_id];
  const std::int64_t edge_row_off = graph_pool.offsets[graph_id] + local_row_i * degree;
  const int sample_degree         = new_pool.degrees[graph_id];
  const auto sample_row_off       = new_pool.offsets[graph_id] + local_row_i * sample_degree;
  const auto old_sample_row_off   = old_pool.offsets[graph_id] + local_row_i * sample_degree;
  const int num_segments = max(1, (degree + kRangeCagraSegmentSize - 1) / kRangeCagraSegmentSize);

  if (threadIdx.x == 0) {
    int new_count = 0;
    int old_count = 0;
    for (int i = 0; i < sample_degree; ++i) {
      new_pool.edges[sample_row_off + i]     = kRangeCagraBuildInvalidId;
      old_pool.edges[old_sample_row_off + i] = kRangeCagraBuildInvalidId;
    }

    for (int j = 0;
         j < kRangeCagraSegmentSize && (new_count < sample_count || old_count < sample_count);
         ++j) {
      for (int seg = 0;
           seg < num_segments && (new_count < sample_count || old_count < sample_count);
           ++seg) {
        const int slot = seg * kRangeCagraSegmentSize + j;
        if (slot >= degree) { continue; }
        const auto raw = graph_pool.edges[edge_row_off + slot];
        const auto id  = build_local_id(raw);
        if (id >= static_cast<std::uint32_t>(rows) || id == self) { continue; }
        if (build_is_new(raw)) {
          if (new_count < sample_count) {
            new_pool.edges[sample_row_off + new_count] = id;
            ++new_count;
            graph_pool.edges[edge_row_off + slot] = build_mark_old(id);
          }
        } else if (old_count < sample_count) {
          old_pool.edges[old_sample_row_off + old_count] = id;
          ++old_count;
        }
      }
    }
    new_counts[flat_row] = new_count;
    old_counts[flat_row] = old_count;
  }
}

template <int MaxBiSamples, int LocalJoinThreads, bool SegmentedInsert>
RAFT_KERNEL range_graph_local_join_kernel(GlobalDatasetView dataset,
                                          DeviceRangeGraphPoolView graph_pool,
                                          float* __restrict__ graph_dists,
                                          int* __restrict__ row_locks,
                                          DeviceRangeGraphPoolView new_pool,
                                          DeviceRangeGraphPoolView rev_new_pool,
                                          const int* __restrict__ new_counts,
                                          const int* __restrict__ rev_new_counts,
                                          DeviceRangeGraphPoolView old_pool,
                                          DeviceRangeGraphPoolView rev_old_pool,
                                          const int* __restrict__ old_counts,
                                          const int* __restrict__ rev_old_counts,
                                          const std::int64_t* __restrict__ row_offsets,
                                          std::int64_t total_rows)
{
  constexpr int skewed_max_bi_samples = range_cagra_skewed_bi_samples<MaxBiSamples>();
  const std::int64_t flat_row = static_cast<std::int64_t>(blockIdx.x);
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i = flat_row - row_offsets[graph_id];
  const int rows                 = graph_pool.rows[graph_id];
  const std::int64_t rank_l      = graph_pool.rank_l[graph_id];
  const int sample_degree        = new_pool.degrees[graph_id];
  const auto sample_row_off      = new_pool.offsets[graph_id] + local_row_i * sample_degree;

  __shared__ std::uint32_t s_new[MaxBiSamples];
  __shared__ std::uint32_t s_old[MaxBiSamples];
  __shared__ int s_counts[2];
  __shared__ float s_new_vec[MaxBiSamples][kRangeCagraGnndTileDim + kRangeCagraGnndTilePad];
  __shared__ float s_old_vec[MaxBiSamples][kRangeCagraGnndTileDim + kRangeCagraGnndTilePad];
  __shared__ float s_dist[MaxBiSamples * skewed_max_bi_samples];

  if (threadIdx.x == 0) {
    int new_size          = 0;
    int old_size          = 0;
    const int forward_new = min(new_counts[flat_row], sample_degree);
    const int reverse_new = min(rev_new_counts[flat_row], sample_degree);
    const int forward_old = min(old_counts[flat_row], sample_degree);
    const int reverse_old = min(rev_old_counts[flat_row], sample_degree);

    for (int i = 0; i < forward_new; ++i) {
      append_unique_build_sample<MaxBiSamples>(
        s_new, new_size, new_pool.edges[sample_row_off + i], rows);
    }
    for (int i = 0; i < reverse_new; ++i) {
      append_unique_build_sample<MaxBiSamples>(
        s_new, new_size, rev_new_pool.edges[sample_row_off + i], rows);
    }
    for (int i = 0; i < forward_old; ++i) {
      append_unique_build_sample<MaxBiSamples>(
        s_old, old_size, old_pool.edges[sample_row_off + i], rows);
    }
    for (int i = 0; i < reverse_old; ++i) {
      append_unique_build_sample<MaxBiSamples>(
        s_old, old_size, rev_old_pool.edges[sample_row_off + i], rows);
    }

    s_counts[0] = new_size;
    s_counts[1] = old_size;
  }
  __syncthreads();

  const int new_size = s_counts[0];
  const int old_size = s_counts[1];
  if (new_size <= 0) { return; }

  for (int i = threadIdx.x; i < MaxBiSamples * skewed_max_bi_samples; i += blockDim.x) {
    s_dist[i] = 0.0f;
  }
  __syncthreads();

  for (int dim_base = 0; dim_base < dataset.dim; dim_base += kRangeCagraGnndTileDim) {
    const int tile_dims = min(kRangeCagraGnndTileDim, dataset.dim - dim_base);
    for (int i = threadIdx.x; i < new_size * kRangeCagraGnndTileDim; i += blockDim.x) {
      const int row = i / kRangeCagraGnndTileDim;
      const int d   = i - row * kRangeCagraGnndTileDim;
      s_new_vec[row][d] =
        d < tile_dims ? dataset.row(rank_l + static_cast<std::int64_t>(s_new[row]))[dim_base + d]
                      : 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < new_size * new_size; i += blockDim.x) {
      const int row = i / new_size;
      const int col = i - row * new_size;
      float acc     = 0.0f;
      for (int d = 0; d < tile_dims; ++d) {
        const float diff = s_new_vec[row][d] - s_new_vec[col][d];
        acc += diff * diff;
      }
      s_dist[row * skewed_max_bi_samples + col] += acc;
    }
    __syncthreads();
  }

  constexpr int num_warps = LocalJoinThreads / kRangeCagraWarpSize;
  const int warp_id       = threadIdx.x / kRangeCagraWarpSize;
  const int lane          = threadIdx.x & (kRangeCagraWarpSize - 1);
  for (int row = warp_id; row < new_size; row += num_warps) {
    float best_dist       = INFINITY;
    std::uint32_t best_id = kRangeCagraBuildInvalidId;
    for (int col = lane; col < new_size; col += kRangeCagraWarpSize) {
      if (col == row) { continue; }
      const float dist = s_dist[row * skewed_max_bi_samples + col];
      const auto id    = s_new[col];
      if (better_build_pair(dist, id, best_dist, best_id)) {
        best_dist = dist;
        best_id   = id;
      }
    }
    warp_reduce_best_pair(best_dist, best_id);
    if (lane == 0 && best_id < static_cast<std::uint32_t>(rows)) {
      if constexpr (SegmentedInsert) {
        insert_segmented_build_candidate(graph_pool,
                                         graph_dists,
                                         row_locks,
                                         row_offsets,
                                         graph_id,
                                         s_new[row],
                                         best_id,
                                         best_dist);
      } else {
        insert_global_build_candidate(graph_pool,
                                      graph_dists,
                                      row_locks,
                                      row_offsets,
                                      graph_id,
                                      s_new[row],
                                      best_id,
                                      best_dist);
      }
    }
  }

  if (old_size <= 0) { return; }
  __syncthreads();

  for (int i = threadIdx.x; i < MaxBiSamples * skewed_max_bi_samples; i += blockDim.x) {
    s_dist[i] = 0.0f;
  }
  __syncthreads();

  for (int dim_base = 0; dim_base < dataset.dim; dim_base += kRangeCagraGnndTileDim) {
    const int tile_dims = min(kRangeCagraGnndTileDim, dataset.dim - dim_base);
    for (int i = threadIdx.x; i < new_size * kRangeCagraGnndTileDim; i += blockDim.x) {
      const int row = i / kRangeCagraGnndTileDim;
      const int d   = i - row * kRangeCagraGnndTileDim;
      s_new_vec[row][d] =
        d < tile_dims ? dataset.row(rank_l + static_cast<std::int64_t>(s_new[row]))[dim_base + d]
                      : 0.0f;
    }
    for (int i = threadIdx.x; i < old_size * kRangeCagraGnndTileDim; i += blockDim.x) {
      const int row = i / kRangeCagraGnndTileDim;
      const int d   = i - row * kRangeCagraGnndTileDim;
      s_old_vec[row][d] =
        d < tile_dims ? dataset.row(rank_l + static_cast<std::int64_t>(s_old[row]))[dim_base + d]
                      : 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < new_size * old_size; i += blockDim.x) {
      const int row = i / old_size;
      const int col = i - row * old_size;
      float acc     = 0.0f;
      for (int d = 0; d < tile_dims; ++d) {
        const float diff = s_new_vec[row][d] - s_old_vec[col][d];
        acc += diff * diff;
      }
      s_dist[row * skewed_max_bi_samples + col] += acc;
    }
    __syncthreads();
  }

  for (int row = warp_id; row < new_size; row += num_warps) {
    float best_dist       = INFINITY;
    std::uint32_t best_id = kRangeCagraBuildInvalidId;
    for (int col = lane; col < old_size; col += kRangeCagraWarpSize) {
      const float dist = s_dist[row * skewed_max_bi_samples + col];
      const auto id    = s_old[col];
      if (better_build_pair(dist, id, best_dist, best_id)) {
        best_dist = dist;
        best_id   = id;
      }
    }
    warp_reduce_best_pair(best_dist, best_id);
    if (lane == 0 && best_id < static_cast<std::uint32_t>(rows)) {
      if constexpr (SegmentedInsert) {
        insert_segmented_build_candidate(graph_pool,
                                         graph_dists,
                                         row_locks,
                                         row_offsets,
                                         graph_id,
                                         s_new[row],
                                         best_id,
                                         best_dist);
      } else {
        insert_global_build_candidate(graph_pool,
                                      graph_dists,
                                      row_locks,
                                      row_offsets,
                                      graph_id,
                                      s_new[row],
                                      best_id,
                                      best_dist);
      }
    }
  }

  for (int col = warp_id; col < old_size; col += num_warps) {
    float best_dist       = INFINITY;
    std::uint32_t best_id = kRangeCagraBuildInvalidId;
    for (int row = lane; row < new_size; row += kRangeCagraWarpSize) {
      const float dist = s_dist[row * skewed_max_bi_samples + col];
      const auto id    = s_new[row];
      if (better_build_pair(dist, id, best_dist, best_id)) {
        best_dist = dist;
        best_id   = id;
      }
    }
    warp_reduce_best_pair(best_dist, best_id);
    if (lane == 0 && best_id < static_cast<std::uint32_t>(rows)) {
      if constexpr (SegmentedInsert) {
        insert_segmented_build_candidate(graph_pool,
                                         graph_dists,
                                         row_locks,
                                         row_offsets,
                                         graph_id,
                                         s_old[col],
                                         best_id,
                                         best_dist);
      } else {
        insert_global_build_candidate(graph_pool,
                                      graph_dists,
                                      row_locks,
                                      row_offsets,
                                      graph_id,
                                      s_old[col],
                                      best_id,
                                      best_dist);
      }
    }
  }
}

RAFT_KERNEL range_graph_reverse_edges_kernel(DeviceRangeGraphPoolView graph_pool,
                                             const std::int64_t* __restrict__ row_offsets,
                                             std::int64_t edge_count,
                                             std::uint32_t* __restrict__ reverse_edges,
                                             int* __restrict__ reverse_counts)
{
  const std::int64_t flat_edge = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (flat_edge >= edge_count) { return; }

  const int graph_id         = graph_id_from_flat_edge(graph_pool, flat_edge);
  const std::int64_t local_e = flat_edge - graph_pool.offsets[graph_id];
  const int degree           = graph_pool.degrees[graph_id];
  const std::uint32_t src    = static_cast<std::uint32_t>(local_e / degree);
  const std::uint32_t dst    = build_local_id(graph_pool.edges[flat_edge]);
  const int rows             = graph_pool.rows[graph_id];
  if (dst >= static_cast<std::uint32_t>(rows) || dst == src) { return; }

  const std::int64_t dst_flat_row = row_offsets[graph_id] + static_cast<std::int64_t>(dst);
  const int slot                  = atomicAdd(reverse_counts + dst_flat_row, 1);
  if (slot < degree) {
    reverse_edges[graph_pool.offsets[graph_id] + static_cast<std::int64_t>(dst) * degree + slot] =
      src;
  }
}

RAFT_KERNEL range_graph_reverse_sample_kernel(DeviceRangeGraphPoolView source_pool,
                                              DeviceRangeGraphPoolView reverse_pool,
                                              const std::int64_t* __restrict__ row_offsets,
                                              std::int64_t source_edge_count,
                                              int* __restrict__ reverse_counts)
{
  const std::int64_t flat_edge = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (flat_edge >= source_edge_count) { return; }

  const int graph_id         = graph_id_from_flat_edge(source_pool, flat_edge);
  const std::int64_t local_e = flat_edge - source_pool.offsets[graph_id];
  const int source_degree    = source_pool.degrees[graph_id];
  const int reverse_degree   = reverse_pool.degrees[graph_id];
  const std::uint32_t src    = static_cast<std::uint32_t>(local_e / source_degree);
  const std::uint32_t dst    = build_local_id(source_pool.edges[flat_edge]);
  const int rows             = source_pool.rows[graph_id];
  if (reverse_degree <= 0 || dst >= static_cast<std::uint32_t>(rows) || dst == src) { return; }

  const std::int64_t dst_flat_row = row_offsets[graph_id] + static_cast<std::int64_t>(dst);
  const int slot                  = atomicAdd(reverse_counts + dst_flat_row, 1);
  if (slot < reverse_degree) {
    reverse_pool.edges[reverse_pool.offsets[graph_id] +
                       static_cast<std::int64_t>(dst) * reverse_degree + slot] = src;
  }
}

__device__ inline void warp_reduce_best_slot(float& dist, std::uint32_t& id, std::uint32_t& slot)
{
#pragma unroll
  for (int offset = kRangeCagraWarpSize >> 1; offset > 0; offset >>= 1) {
    const float other_dist = __shfl_down_sync(0xffffffffu, dist, offset);
    const auto other_id    = __shfl_down_sync(0xffffffffu, id, offset);
    const auto other_slot  = __shfl_down_sync(0xffffffffu, slot, offset);
    if (better_build_pair(other_dist, other_id, dist, id)) {
      dist = other_dist;
      id   = other_id;
      slot = other_slot;
    }
  }
}

RAFT_KERNEL range_graph_sort_by_distance_kernel(DeviceRangeGraphPoolView graph_pool,
                                                float* __restrict__ graph_dists,
                                                const std::int64_t* __restrict__ row_offsets,
                                                std::int64_t total_rows)
{
  const int wid  = threadIdx.x / kRangeCagraWarpSize;
  const int lane = threadIdx.x & (kRangeCagraWarpSize - 1);
  const std::int64_t flat_row =
    static_cast<std::int64_t>(blockIdx.x) * kRangeCagraOptimizeWarps + wid;
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i = flat_row - row_offsets[graph_id];
  const auto self                = static_cast<std::uint32_t>(local_row_i);
  const int rows                 = graph_pool.rows[graph_id];
  const int degree               = graph_pool.degrees[graph_id];
  const auto row_off             = graph_pool.offsets[graph_id] + local_row_i * degree;

  extern __shared__ unsigned char raw_shared[];
  auto* smem_ids = reinterpret_cast<std::uint32_t*>(raw_shared) + wid * kRangeCagraBuildMaxDegree;
  auto* smem_dists =
    reinterpret_cast<float*>(raw_shared + kRangeCagraOptimizeWarps * kRangeCagraBuildMaxDegree *
                                            sizeof(std::uint32_t)) +
    wid * kRangeCagraBuildMaxDegree;

  for (int k = lane; k < degree; k += kRangeCagraWarpSize) {
    const auto id = build_local_id(graph_pool.edges[row_off + k]);
    smem_ids[k] =
      (id < static_cast<std::uint32_t>(rows) && id != self) ? id : kRangeCagraBuildInvalidId;
    smem_dists[k] =
      smem_ids[k] < static_cast<std::uint32_t>(rows) ? graph_dists[row_off + k] : INFINITY;
  }
  __syncwarp();

  for (int out_k = 0; out_k < degree; ++out_k) {
    float local_dist       = INFINITY;
    std::uint32_t local_id = kRangeCagraBuildInvalidId;
    std::uint32_t local_k  = kRangeCagraBuildInvalidId;
    for (int k = lane; k < degree; k += kRangeCagraWarpSize) {
      if (better_build_pair(smem_dists[k], smem_ids[k], local_dist, local_id)) {
        local_dist = smem_dists[k];
        local_id   = smem_ids[k];
        local_k    = static_cast<std::uint32_t>(k);
      }
    }
    warp_reduce_best_slot(local_dist, local_id, local_k);
    const auto selected_k = __shfl_sync(0xffffffffu, local_k, 0);
    if (selected_k >= static_cast<std::uint32_t>(degree)) { break; }
    if (lane == 0) {
      graph_pool.edges[row_off + out_k] = local_id;
      graph_dists[row_off + out_k]      = local_dist;
    }
    __syncwarp();
    if (static_cast<int>(selected_k) >= lane &&
        (static_cast<int>(selected_k) - lane) % kRangeCagraWarpSize == 0) {
      smem_ids[selected_k]   = kRangeCagraBuildInvalidId;
      smem_dists[selected_k] = INFINITY;
    }
    __syncwarp();
  }
}

RAFT_KERNEL range_graph_prune_kernel(DeviceRangeGraphPoolView intermediate_pool,
                                     DeviceRangeGraphPoolView final_pool,
                                     const std::int64_t* __restrict__ row_offsets,
                                     std::int64_t total_rows,
                                     std::uint32_t* __restrict__ invalid_rows)
{
  const int wid  = threadIdx.x / kRangeCagraWarpSize;
  const int lane = threadIdx.x & (kRangeCagraWarpSize - 1);
  const std::int64_t flat_row =
    static_cast<std::int64_t>(blockIdx.x) * kRangeCagraOptimizeWarps + wid;
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, final_pool.graph_count, flat_row);
  const std::int64_t local_row_i = flat_row - row_offsets[graph_id];
  const auto self                = static_cast<std::uint32_t>(local_row_i);
  const int rows                 = final_pool.rows[graph_id];
  const int intermediate_degree  = intermediate_pool.degrees[graph_id];
  const int final_degree         = final_pool.degrees[graph_id];
  const auto intermediate_row_off =
    intermediate_pool.offsets[graph_id] + local_row_i * intermediate_degree;
  const auto final_row_off = final_pool.offsets[graph_id] + local_row_i * final_degree;

  extern __shared__ unsigned char raw_shared[];
  auto* smem_indices =
    reinterpret_cast<std::uint32_t*>(raw_shared) + wid * kRangeCagraBuildMaxDegree;
  auto* smem_detours =
    reinterpret_cast<std::uint32_t*>(
      raw_shared + kRangeCagraOptimizeWarps * kRangeCagraBuildMaxDegree * sizeof(std::uint32_t)) +
    wid * kRangeCagraBuildMaxDegree;

  constexpr std::uint32_t maxval16 = 0x0000ffffu;
  for (int k = lane; k < intermediate_degree; k += kRangeCagraWarpSize) {
    auto id = build_local_id(intermediate_pool.edges[intermediate_row_off + k]);
    if (id >= static_cast<std::uint32_t>(rows) || id == self) {
      id              = kRangeCagraBuildInvalidId;
      smem_detours[k] = maxval16;
    } else {
      smem_detours[k] = 0;
    }
    smem_indices[k] = id;
  }
  __syncwarp();

  for (int k_ad = 0; k_ad < intermediate_degree - 1; ++k_ad) {
    const auto detour = smem_indices[k_ad];
    if (detour >= static_cast<std::uint32_t>(rows)) { continue; }
    const auto detour_row_off =
      intermediate_pool.offsets[graph_id] + static_cast<std::int64_t>(detour) * intermediate_degree;
    for (int k_db = lane; k_db < intermediate_degree; k_db += kRangeCagraWarpSize) {
      const auto candidate = build_local_id(intermediate_pool.edges[detour_row_off + k_db]);
      if (candidate >= static_cast<std::uint32_t>(rows)) { continue; }
      for (int k_ab = k_ad + 1; k_ab < intermediate_degree; ++k_ab) {
        if (smem_indices[k_ab] == candidate) {
          atomicAdd(smem_detours + k_ab, 1u);
          break;
        }
      }
    }
    __syncwarp();
  }

  for (int k = lane; k < intermediate_degree; k += kRangeCagraWarpSize) {
    smem_detours[k] = min(smem_detours[k], maxval16);
    if (smem_indices[k] >= static_cast<std::uint32_t>(rows)) { smem_detours[k] = maxval16; }
  }
  __syncwarp();

  for (int out_k = 0; out_k < final_degree; ++out_k) {
    std::uint32_t local_min = maxval16;
    std::uint32_t local_idx = maxval16;
    for (int k = lane; k < intermediate_degree; k += kRangeCagraWarpSize) {
      if (smem_detours[k] < local_min) {
        local_min = smem_detours[k];
        local_idx = static_cast<std::uint32_t>(k);
      }
    }

    const auto min_with_idx = warp_reduce_min_u32((local_min << 16) | local_idx);
    const auto min_count    = min_with_idx >> 16;
    const auto selected_idx = min_with_idx & 0x0000ffffu;
    if (min_count == maxval16 || selected_idx == maxval16 ||
        selected_idx >= static_cast<std::uint32_t>(intermediate_degree)) {
      if (lane == 0) {
        atomicAdd(invalid_rows, 1u);
        for (int k = out_k; k < final_degree; ++k) {
          auto fallback = static_cast<std::uint32_t>((static_cast<std::int64_t>(self) + k + 1) %
                                                     static_cast<std::int64_t>(rows));
          if (fallback == self) { fallback = (fallback + 1) % static_cast<std::uint32_t>(rows); }
          final_pool.edges[final_row_off + k] = fallback;
        }
      }
      break;
    }

    const auto selected_node = smem_indices[selected_idx];
    for (int k = lane; k < intermediate_degree; k += kRangeCagraWarpSize) {
      if (smem_indices[k] == selected_node) { smem_detours[k] = maxval16; }
    }
    __syncwarp();

    if (lane == 0) { final_pool.edges[final_row_off + out_k] = selected_node; }
  }
}

RAFT_KERNEL range_graph_merge_reverse_kernel(DeviceRangeGraphPoolView graph_pool,
                                             const std::uint32_t* __restrict__ reverse_edges,
                                             const int* __restrict__ reverse_counts,
                                             const std::int64_t* __restrict__ row_offsets,
                                             std::int64_t total_rows)
{
  const int wid  = threadIdx.x / kRangeCagraWarpSize;
  const int lane = threadIdx.x & (kRangeCagraWarpSize - 1);
  const std::int64_t flat_row =
    static_cast<std::int64_t>(blockIdx.x) * kRangeCagraOptimizeWarps + wid;
  if (flat_row >= total_rows) { return; }

  const int graph_id = graph_id_from_flat_row(row_offsets, graph_pool.graph_count, flat_row);
  const std::int64_t local_row_i = flat_row - row_offsets[graph_id];
  const int rows                 = graph_pool.rows[graph_id];
  const int degree               = graph_pool.degrees[graph_id];
  const auto row_off             = graph_pool.offsets[graph_id] + local_row_i * degree;

  extern __shared__ unsigned char raw_shared[];
  auto* smem_graph = reinterpret_cast<std::uint32_t*>(raw_shared) + wid * kRangeCagraBuildMaxDegree;
  for (int k = lane; k < degree; k += kRangeCagraWarpSize) {
    smem_graph[k] = build_local_id(graph_pool.edges[row_off + k]);
  }
  __syncwarp();

  const int protected_edges = degree / 2;
  int kr                    = min(reverse_counts[flat_row], degree);
  while (kr > 0) {
    --kr;
    const auto reverse_value = build_local_id(reverse_edges[row_off + kr]);
    if (reverse_value < static_cast<std::uint32_t>(rows)) {
      if (lane == 0) {
        int pos = degree;
        for (int k = 0; k < degree; ++k) {
          if (smem_graph[k] == reverse_value) {
            pos = k;
            break;
          }
        }
        if (pos >= protected_edges && protected_edges < degree) {
          const int shift_end = pos < degree ? pos : degree - 1;
          for (int k = shift_end; k > protected_edges; --k) {
            smem_graph[k] = smem_graph[k - 1];
          }
          smem_graph[protected_edges] = reverse_value;
        }
      }
      __syncwarp();
    }
  }

  for (int k = lane; k < degree; k += kRangeCagraWarpSize) {
    graph_pool.edges[row_off + k] = smem_graph[k];
  }
}

}  // namespace build_kernels

[[nodiscard]] inline std::size_t range_graph_build_shared_bytes()
{
  const int teams_per_cta = kRangeCagraBuildThreads / kRangeCagraBuildTeamSize;
  return kRangeCagraBuildMaxDegree * (sizeof(float) + sizeof(std::uint32_t)) +
         teams_per_cta * (sizeof(float) + sizeof(std::uint32_t));
}

[[nodiscard]] inline std::size_t range_graph_optimize_shared_bytes(bool with_detours)
{
  auto bytes = static_cast<std::size_t>(kRangeCagraOptimizeWarps) * kRangeCagraBuildMaxDegree *
               sizeof(std::uint32_t);
  if (with_detours) { bytes *= 2; }
  return bytes;
}

[[nodiscard]] inline std::size_t range_graph_sort_shared_bytes()
{
  return static_cast<std::size_t>(kRangeCagraOptimizeWarps) * kRangeCagraBuildMaxDegree *
         (sizeof(std::uint32_t) + sizeof(float));
}

[[nodiscard]] inline int range_cagra_round_up_32(int value)
{
  return ((value + kRangeCagraSegmentSize - 1) / kRangeCagraSegmentSize) * kRangeCagraSegmentSize;
}

[[nodiscard]] inline int range_cagra_cagra_like_internal_degree(int degree)
{
  if (degree <= kRangeCagraSegmentSize) { return range_cagra_round_up_32(degree); }
  return range_cagra_round_up_32((degree * 13 + 9) / 10);
}

inline void build_range_graph_pool_edges_on_gpu(raft::resources const& res,
                                                GlobalDatasetView const& dataset,
                                                DeviceRangeGraphPoolView graph_pool,
                                                std::vector<std::int64_t> const& row_offsets,
                                                std::vector<int> const& final_degrees,
                                                std::vector<int> const& requested_intermediate_degrees,
                                                RangeGraphBuildParams const& params)
{
  RAFT_EXPECTS(dataset.base != nullptr, "GlobalDatasetView::base must not be null");
  RAFT_EXPECTS(dataset.rows > 0, "GlobalDatasetView::rows must be positive");
  RAFT_EXPECTS(dataset.dim > 0, "GlobalDatasetView::dim must be positive");
  RAFT_EXPECTS(dataset.stride > 0, "GlobalDatasetView::stride must be positive");
  RAFT_EXPECTS(dataset.stride == dataset.dim,
               "range_cagra build uses one global compact base table; do not gather per-range "
               "vectors as a fallback");
  RAFT_EXPECTS(graph_pool.graph_count > 0, "graph_pool must contain at least one graph");
  RAFT_EXPECTS(row_offsets.size() == static_cast<std::size_t>(graph_pool.graph_count) + 1,
               "row_offsets must have graph_count + 1 elements");
  RAFT_EXPECTS(final_degrees.size() == static_cast<std::size_t>(graph_pool.graph_count),
               "final_degrees must have graph_count elements");
  RAFT_EXPECTS(requested_intermediate_degrees.size() ==
                 static_cast<std::size_t>(graph_pool.graph_count),
               "requested_intermediate_degrees must have graph_count elements");
  RAFT_EXPECTS(params.graph_degree > 0, "graph_degree must be positive");
  RAFT_EXPECTS(params.graph_degree <= kRangeCagraBuildMaxDegree,
               "graph_degree exceeds range-CAGRA GPU build maximum");
  RAFT_EXPECTS(params.intermediate_graph_degree > 0, "intermediate_graph_degree must be positive");
  RAFT_EXPECTS(params.intermediate_graph_degree <= kRangeCagraBuildMaxDegree,
               "intermediate_graph_degree exceeds range-CAGRA GPU build maximum");
  RAFT_EXPECTS(params.nn_descent_iterations >= 0, "nn_descent_iterations must be non-negative");

  const auto stream     = raft::resource::get_cuda_stream(res);
  const auto total_rows = row_offsets.back();
  if (total_rows <= 0) { return; }

  const bool segmented_build = range_cagra_uses_segmented_build(params);
  const int requested_samples =
    segmented_build ? kRangeCagraSegmentedBuildSamples : kRangeCagraFlatBuildSamples;
  const int max_requested_intermediate =
    *std::max_element(requested_intermediate_degrees.begin(), requested_intermediate_degrees.end());
  const int refine_samples =
    std::min<int>({max_requested_intermediate, requested_samples, kRangeCagraBuildMaxDegree});
  const auto shared_bytes       = range_graph_build_shared_bytes();
  constexpr std::uint64_t seed  = 0x9e3779b97f4a7c15ull;
  constexpr int reverse_threads = 256;

  std::vector<std::int64_t> global_final_offsets(static_cast<std::size_t>(graph_pool.graph_count) +
                                                 1);
  std::vector<int> intermediate_degrees(static_cast<std::size_t>(graph_pool.graph_count));
  std::vector<int> reverse_degrees(static_cast<std::size_t>(graph_pool.graph_count));
  std::vector<int> init_candidate_counts(static_cast<std::size_t>(graph_pool.graph_count));
  std::vector<std::size_t> graph_temp_bytes(static_cast<std::size_t>(graph_pool.graph_count));
  std::int64_t expected_final_edge_count = 0;
  for (int graph_id = 0; graph_id < graph_pool.graph_count; ++graph_id) {
    const auto rows_i64 = row_offsets[static_cast<std::size_t>(graph_id) + 1] -
                          row_offsets[static_cast<std::size_t>(graph_id)];
    const auto rows                   = static_cast<int>(rows_i64);
    const auto final_degree = final_degrees[static_cast<std::size_t>(graph_id)];
    const auto requested_intermediate =
      std::max(requested_intermediate_degrees[static_cast<std::size_t>(graph_id)], final_degree);
    const auto internal_intermediate =
      segmented_build ? range_cagra_cagra_like_internal_degree(requested_intermediate)
                      : requested_intermediate;
    const auto intermediate_degree =
      std::min({internal_intermediate, rows - 1, kRangeCagraBuildMaxDegree});
    const auto reverse_degree = std::min(refine_samples, intermediate_degree);
    const auto init_candidate_count =
      std::min(std::max(intermediate_degree * 4, 256), kRangeCagraBuildMaxInitCandidates);
    RAFT_EXPECTS(final_degree > 0, "final graph degree must be positive");
    RAFT_EXPECTS(final_degree <= rows - 1, "final graph degree must fit the graph row count");
    RAFT_EXPECTS(intermediate_degree >= final_degree,
                 "intermediate graph degree must be at least final graph degree");
    intermediate_degrees[static_cast<std::size_t>(graph_id)] = intermediate_degree;
    reverse_degrees[static_cast<std::size_t>(graph_id)]      = reverse_degree;
    init_candidate_counts[static_cast<std::size_t>(graph_id)] = init_candidate_count;
    global_final_offsets[static_cast<std::size_t>(graph_id)] = expected_final_edge_count;
    expected_final_edge_count +=
      static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(final_degree);
    graph_temp_bytes[static_cast<std::size_t>(graph_id)] =
      static_cast<std::size_t>(rows) *
      (static_cast<std::size_t>(intermediate_degree) * (sizeof(std::uint32_t) + sizeof(float)) +
       static_cast<std::size_t>(reverse_degree) * 4 * sizeof(std::uint32_t) +
       (segmented_build ? kRangeCagraMaxBuildSegments + 4 : 5) * sizeof(int));
  }
  global_final_offsets[static_cast<std::size_t>(graph_pool.graph_count)] =
    expected_final_edge_count;
  RAFT_EXPECTS(expected_final_edge_count == graph_pool.edge_count,
               "computed final edge count must match graph_pool.edge_count");

  std::size_t free_bytes  = 0;
  std::size_t total_bytes = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  constexpr std::size_t min_temp_budget = std::size_t{512} << 20;
  constexpr std::size_t max_temp_budget = std::size_t{3} << 30;
  auto temp_budget                      = std::min(max_temp_budget, free_bytes / 2);
  temp_budget                           = std::max(temp_budget, min_temp_budget);

  int chunk_begin = 0;
  while (chunk_begin < graph_pool.graph_count) {
    int chunk_end              = chunk_begin;
    std::size_t chunk_estimate = 0;
    do {
      const auto next_estimate =
        chunk_estimate + graph_temp_bytes[static_cast<std::size_t>(chunk_end)];
      if (chunk_end > chunk_begin && next_estimate > temp_budget) { break; }
      chunk_estimate = next_estimate;
      ++chunk_end;
    } while (chunk_end < graph_pool.graph_count);

    const int chunk_count = chunk_end - chunk_begin;
    std::vector<std::int64_t> h_row_offsets(static_cast<std::size_t>(chunk_count) + 1);
    std::vector<std::int64_t> h_intermediate_offsets(static_cast<std::size_t>(chunk_count));
    std::vector<std::int64_t> h_reverse_offsets(static_cast<std::size_t>(chunk_count));
    std::vector<std::int64_t> h_final_offsets(static_cast<std::size_t>(chunk_count));
    std::vector<int> h_intermediate_degrees(static_cast<std::size_t>(chunk_count));
    std::vector<int> h_reverse_degrees(static_cast<std::size_t>(chunk_count));
    std::vector<int> h_init_candidate_counts(static_cast<std::size_t>(chunk_count));

    std::int64_t chunk_rows              = 0;
    std::int64_t intermediate_edge_count = 0;
    std::int64_t reverse_edge_count      = 0;
    const auto first_final_edge_offset =
      global_final_offsets[static_cast<std::size_t>(chunk_begin)];
    for (int local_graph_id = 0; local_graph_id < chunk_count; ++local_graph_id) {
      const int global_graph_id = chunk_begin + local_graph_id;
      const auto rows_i64       = row_offsets[static_cast<std::size_t>(global_graph_id) + 1] -
                            row_offsets[static_cast<std::size_t>(global_graph_id)];
      const auto rows = static_cast<int>(rows_i64);
      const auto intermediate_degree =
        intermediate_degrees[static_cast<std::size_t>(global_graph_id)];
      const auto reverse_degree = reverse_degrees[static_cast<std::size_t>(global_graph_id)];
      h_row_offsets[static_cast<std::size_t>(local_graph_id)]          = chunk_rows;
      h_intermediate_offsets[static_cast<std::size_t>(local_graph_id)] = intermediate_edge_count;
      h_reverse_offsets[static_cast<std::size_t>(local_graph_id)]      = reverse_edge_count;
      h_final_offsets[static_cast<std::size_t>(local_graph_id)] =
        global_final_offsets[static_cast<std::size_t>(global_graph_id)] - first_final_edge_offset;
      h_intermediate_degrees[static_cast<std::size_t>(local_graph_id)] = intermediate_degree;
      h_reverse_degrees[static_cast<std::size_t>(local_graph_id)]      = reverse_degree;
      h_init_candidate_counts[static_cast<std::size_t>(local_graph_id)] =
        init_candidate_counts[static_cast<std::size_t>(global_graph_id)];
      chunk_rows += rows;
      intermediate_edge_count +=
        static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(intermediate_degree);
      reverse_edge_count +=
        static_cast<std::int64_t>(rows) * static_cast<std::int64_t>(reverse_degree);
    }
    h_row_offsets[static_cast<std::size_t>(chunk_count)] = chunk_rows;
    const auto final_edge_count =
      global_final_offsets[static_cast<std::size_t>(chunk_end)] - first_final_edge_offset;

    std::int64_t* d_row_offsets          = nullptr;
    std::int64_t* d_intermediate_offsets = nullptr;
    std::int64_t* d_reverse_offsets      = nullptr;
    std::int64_t* d_final_offsets        = nullptr;
    int* d_intermediate_degrees          = nullptr;
    int* d_reverse_degrees               = nullptr;
    int* d_init_candidate_counts         = nullptr;
    std::uint32_t* d_intermediate_edges  = nullptr;
    float* d_intermediate_dists          = nullptr;
    std::uint32_t* d_new_edges           = nullptr;
    std::uint32_t* d_old_edges           = nullptr;
    std::uint32_t* d_rev_new_edges       = nullptr;
    std::uint32_t* d_rev_old_edges       = nullptr;
    int* d_new_counts                    = nullptr;
    int* d_old_counts                    = nullptr;
    int* d_rev_new_counts                = nullptr;
    int* d_rev_old_counts                = nullptr;
    int* d_row_locks                     = nullptr;
    std::uint32_t* d_invalid_prune_rows  = nullptr;
    const auto row_lock_count =
      static_cast<std::size_t>(chunk_rows) *
      static_cast<std::size_t>(segmented_build ? kRangeCagraMaxBuildSegments : 1);

    RAFT_CUDA_TRY(cudaMalloc(&d_row_offsets, sizeof(std::int64_t) * h_row_offsets.size()));
    RAFT_CUDA_TRY(
      cudaMalloc(&d_intermediate_offsets, sizeof(std::int64_t) * h_intermediate_offsets.size()));
    RAFT_CUDA_TRY(cudaMalloc(&d_reverse_offsets, sizeof(std::int64_t) * h_reverse_offsets.size()));
    RAFT_CUDA_TRY(cudaMalloc(&d_final_offsets, sizeof(std::int64_t) * h_final_offsets.size()));
    RAFT_CUDA_TRY(cudaMalloc(&d_intermediate_degrees, sizeof(int) * h_intermediate_degrees.size()));
    RAFT_CUDA_TRY(cudaMalloc(&d_reverse_degrees, sizeof(int) * h_reverse_degrees.size()));
    RAFT_CUDA_TRY(
      cudaMalloc(&d_init_candidate_counts, sizeof(int) * h_init_candidate_counts.size()));
    RAFT_CUDA_TRY(
      cudaMalloc(&d_intermediate_edges,
                 sizeof(std::uint32_t) * static_cast<std::size_t>(intermediate_edge_count)));
    RAFT_CUDA_TRY(cudaMalloc(&d_intermediate_dists,
                             sizeof(float) * static_cast<std::size_t>(intermediate_edge_count)));
    RAFT_CUDA_TRY(cudaMalloc(&d_new_edges,
                             sizeof(std::uint32_t) * static_cast<std::size_t>(reverse_edge_count)));
    RAFT_CUDA_TRY(cudaMalloc(&d_old_edges,
                             sizeof(std::uint32_t) * static_cast<std::size_t>(reverse_edge_count)));
    RAFT_CUDA_TRY(cudaMalloc(&d_rev_new_edges,
                             sizeof(std::uint32_t) * static_cast<std::size_t>(reverse_edge_count)));
    RAFT_CUDA_TRY(cudaMalloc(&d_rev_old_edges,
                             sizeof(std::uint32_t) * static_cast<std::size_t>(reverse_edge_count)));
    RAFT_CUDA_TRY(cudaMalloc(&d_new_counts, sizeof(int) * static_cast<std::size_t>(chunk_rows)));
    RAFT_CUDA_TRY(cudaMalloc(&d_old_counts, sizeof(int) * static_cast<std::size_t>(chunk_rows)));
    RAFT_CUDA_TRY(
      cudaMalloc(&d_rev_new_counts, sizeof(int) * static_cast<std::size_t>(chunk_rows)));
    RAFT_CUDA_TRY(
      cudaMalloc(&d_rev_old_counts, sizeof(int) * static_cast<std::size_t>(chunk_rows)));
    RAFT_CUDA_TRY(cudaMalloc(&d_row_locks, sizeof(int) * row_lock_count));
    RAFT_CUDA_TRY(cudaMalloc(&d_invalid_prune_rows, sizeof(std::uint32_t)));

    RAFT_CUDA_TRY(cudaMemcpyAsync(d_row_offsets,
                                  h_row_offsets.data(),
                                  sizeof(std::int64_t) * h_row_offsets.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_intermediate_offsets,
                                  h_intermediate_offsets.data(),
                                  sizeof(std::int64_t) * h_intermediate_offsets.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_reverse_offsets,
                                  h_reverse_offsets.data(),
                                  sizeof(std::int64_t) * h_reverse_offsets.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_final_offsets,
                                  h_final_offsets.data(),
                                  sizeof(std::int64_t) * h_final_offsets.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_intermediate_degrees,
                                  h_intermediate_degrees.data(),
                                  sizeof(int) * h_intermediate_degrees.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_reverse_degrees,
                                  h_reverse_degrees.data(),
                                  sizeof(int) * h_reverse_degrees.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_init_candidate_counts,
                                  h_init_candidate_counts.data(),
                                  sizeof(int) * h_init_candidate_counts.size(),
                                  cudaMemcpyHostToDevice,
                                  stream));

    DeviceRangeGraphPoolView intermediate_pool{d_intermediate_edges,
                                               d_intermediate_offsets,
                                               graph_pool.rank_l + chunk_begin,
                                               graph_pool.rows + chunk_begin,
                                               d_intermediate_degrees,
                                               chunk_count,
                                               intermediate_edge_count};
    DeviceRangeGraphPoolView new_pool{d_new_edges,
                                      d_reverse_offsets,
                                      graph_pool.rank_l + chunk_begin,
                                      graph_pool.rows + chunk_begin,
                                      d_reverse_degrees,
                                      chunk_count,
                                      reverse_edge_count};
    DeviceRangeGraphPoolView old_pool{d_old_edges,
                                      d_reverse_offsets,
                                      graph_pool.rank_l + chunk_begin,
                                      graph_pool.rows + chunk_begin,
                                      d_reverse_degrees,
                                      chunk_count,
                                      reverse_edge_count};
    DeviceRangeGraphPoolView rev_new_pool{d_rev_new_edges,
                                          d_reverse_offsets,
                                          graph_pool.rank_l + chunk_begin,
                                          graph_pool.rows + chunk_begin,
                                          d_reverse_degrees,
                                          chunk_count,
                                          reverse_edge_count};
    DeviceRangeGraphPoolView rev_old_pool{d_rev_old_edges,
                                          d_reverse_offsets,
                                          graph_pool.rank_l + chunk_begin,
                                          graph_pool.rows + chunk_begin,
                                          d_reverse_degrees,
                                          chunk_count,
                                          reverse_edge_count};
    DeviceRangeGraphPoolView final_pool{graph_pool.edges + first_final_edge_offset,
                                        d_final_offsets,
                                        graph_pool.rank_l + chunk_begin,
                                        graph_pool.rows + chunk_begin,
                                        graph_pool.degrees + chunk_begin,
                                        chunk_count,
                                        final_edge_count};

    if (segmented_build) {
      build_kernels::range_graph_segmented_init_kernel<<<static_cast<std::uint32_t>(chunk_rows),
                                                         kRangeCagraBuildThreads,
                                                         0,
                                                         stream>>>(
        intermediate_pool, d_intermediate_dists, d_row_offsets, chunk_rows, seed);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {
      build_kernels::range_graph_init_kernel<<<static_cast<std::uint32_t>(chunk_rows),
                                               kRangeCagraBuildThreads,
                                               shared_bytes,
                                               stream>>>(dataset,
                                                         intermediate_pool,
                                                         d_intermediate_dists,
                                                         d_row_offsets,
                                                         d_init_candidate_counts,
                                                         chunk_rows,
                                                         seed);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    RAFT_CUDA_TRY(cudaMemsetAsync(d_row_locks, 0, sizeof(int) * row_lock_count, stream));
    const int sample_reverse_blocks =
      static_cast<int>((reverse_edge_count + reverse_threads - 1) / reverse_threads);
    for (int iteration = 0; iteration < params.nn_descent_iterations; ++iteration) {
      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_rev_new_edges,
                        0xff,
                        sizeof(std::uint32_t) * static_cast<std::size_t>(reverse_edge_count),
                        stream));
      RAFT_CUDA_TRY(
        cudaMemsetAsync(d_rev_old_edges,
                        0xff,
                        sizeof(std::uint32_t) * static_cast<std::size_t>(reverse_edge_count),
                        stream));
      RAFT_CUDA_TRY(cudaMemsetAsync(
        d_rev_new_counts, 0, sizeof(int) * static_cast<std::size_t>(chunk_rows), stream));
      RAFT_CUDA_TRY(cudaMemsetAsync(
        d_rev_old_counts, 0, sizeof(int) * static_cast<std::size_t>(chunk_rows), stream));
      if (segmented_build) {
        build_kernels::
          range_graph_segmented_sample_old_new_kernel<<<static_cast<std::uint32_t>(chunk_rows),
                                                        kRangeCagraBuildThreads,
                                                        0,
                                                        stream>>>(intermediate_pool,
                                                                  new_pool,
                                                                  old_pool,
                                                                  d_new_counts,
                                                                  d_old_counts,
                                                                  d_row_offsets,
                                                                  chunk_rows,
                                                                  refine_samples);
      } else {
        build_kernels::range_graph_sample_old_new_kernel<<<static_cast<std::uint32_t>(chunk_rows),
                                                           kRangeCagraBuildThreads,
                                                           0,
                                                           stream>>>(intermediate_pool,
                                                                     new_pool,
                                                                     old_pool,
                                                                     d_new_counts,
                                                                     d_old_counts,
                                                                     d_row_offsets,
                                                                     chunk_rows,
                                                                     refine_samples,
                                                                     iteration);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      build_kernels::
        range_graph_reverse_sample_kernel<<<sample_reverse_blocks, reverse_threads, 0, stream>>>(
          new_pool, rev_new_pool, d_row_offsets, reverse_edge_count, d_rev_new_counts);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      build_kernels::
        range_graph_reverse_sample_kernel<<<sample_reverse_blocks, reverse_threads, 0, stream>>>(
          old_pool, rev_old_pool, d_row_offsets, reverse_edge_count, d_rev_old_counts);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      if (segmented_build) {
        build_kernels::range_graph_local_join_kernel<kRangeCagraSegmentedGnndMaxBiSamples,
                                                     kRangeCagraSegmentedLocalJoinThreads,
                                                     true>
          <<<static_cast<std::uint32_t>(chunk_rows),
             kRangeCagraSegmentedLocalJoinThreads,
             0,
             stream>>>(dataset,
                       intermediate_pool,
                       d_intermediate_dists,
                       d_row_locks,
                       new_pool,
                       rev_new_pool,
                       d_new_counts,
                       d_rev_new_counts,
                       old_pool,
                       rev_old_pool,
                       d_old_counts,
                       d_rev_old_counts,
                       d_row_offsets,
                       chunk_rows);
      } else {
        build_kernels::range_graph_local_join_kernel<kRangeCagraFlatGnndMaxBiSamples,
                                                     kRangeCagraFlatLocalJoinThreads,
                                                     false>
          <<<static_cast<std::uint32_t>(chunk_rows),
             kRangeCagraFlatLocalJoinThreads,
             0,
             stream>>>(dataset,
                       intermediate_pool,
                       d_intermediate_dists,
                       d_row_locks,
                       new_pool,
                       rev_new_pool,
                       d_new_counts,
                       d_rev_new_counts,
                       old_pool,
                       rev_old_pool,
                       d_old_counts,
                       d_rev_old_counts,
                       d_row_offsets,
                       chunk_rows);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    RAFT_CUDA_TRY(cudaMemsetAsync(d_invalid_prune_rows, 0, sizeof(std::uint32_t), stream));
    const int optimize_blocks =
      static_cast<int>((chunk_rows + kRangeCagraOptimizeWarps - 1) / kRangeCagraOptimizeWarps);
    if (segmented_build) {
      build_kernels::
        range_graph_sort_by_distance_kernel<<<optimize_blocks,
                                              kRangeCagraOptimizeWarps * kRangeCagraWarpSize,
                                              range_graph_sort_shared_bytes(),
                                              stream>>>(
          intermediate_pool, d_intermediate_dists, d_row_offsets, chunk_rows);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    build_kernels::range_graph_prune_kernel<<<optimize_blocks,
                                              kRangeCagraOptimizeWarps * kRangeCagraWarpSize,
                                              range_graph_optimize_shared_bytes(true),
                                              stream>>>(
      intermediate_pool, final_pool, d_row_offsets, chunk_rows, d_invalid_prune_rows);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);

    std::uint32_t h_invalid_prune_rows = 0;
    RAFT_CUDA_TRY(cudaMemcpy(
      &h_invalid_prune_rows, d_invalid_prune_rows, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    RAFT_EXPECTS(h_invalid_prune_rows == 0,
                 "range-CAGRA graph prune could not fill all final neighbor rows");

    RAFT_CUDA_TRY(cudaFree(d_rev_old_edges));
    d_rev_old_edges = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_rev_new_edges));
    d_rev_new_edges = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_old_edges));
    d_old_edges = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_new_edges));
    d_new_edges = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_intermediate_dists));
    d_intermediate_dists = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_intermediate_edges));
    d_intermediate_edges = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_intermediate_degrees));
    d_intermediate_degrees = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_intermediate_offsets));
    d_intermediate_offsets = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_reverse_degrees));
    d_reverse_degrees = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_init_candidate_counts));
    d_init_candidate_counts = nullptr;
    RAFT_CUDA_TRY(cudaFree(d_reverse_offsets));
    d_reverse_offsets = nullptr;

    std::uint32_t* d_reverse_edges = nullptr;
    int* d_reverse_counts          = d_rev_new_counts;
    d_rev_new_counts               = nullptr;
    RAFT_CUDA_TRY(cudaMalloc(&d_reverse_edges,
                             sizeof(std::uint32_t) * static_cast<std::size_t>(final_edge_count)));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(d_reverse_edges,
                      0xff,
                      sizeof(std::uint32_t) * static_cast<std::size_t>(final_edge_count),
                      stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_reverse_counts, 0, sizeof(int) * static_cast<std::size_t>(chunk_rows), stream));
    const int final_reverse_blocks =
      static_cast<int>((final_edge_count + reverse_threads - 1) / reverse_threads);
    build_kernels::
      range_graph_reverse_edges_kernel<<<final_reverse_blocks, reverse_threads, 0, stream>>>(
        final_pool, d_row_offsets, final_edge_count, d_reverse_edges, d_reverse_counts);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    build_kernels::
      range_graph_merge_reverse_kernel<<<optimize_blocks,
                                         kRangeCagraOptimizeWarps * kRangeCagraWarpSize,
                                         range_graph_optimize_shared_bytes(false),
                                         stream>>>(
        final_pool, d_reverse_edges, d_reverse_counts, d_row_offsets, chunk_rows);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);

    RAFT_CUDA_TRY(cudaFree(d_invalid_prune_rows));
    RAFT_CUDA_TRY(cudaFree(d_row_locks));
    RAFT_CUDA_TRY(cudaFree(d_rev_old_counts));
    RAFT_CUDA_TRY(cudaFree(d_old_counts));
    RAFT_CUDA_TRY(cudaFree(d_new_counts));
    RAFT_CUDA_TRY(cudaFree(d_reverse_counts));
    RAFT_CUDA_TRY(cudaFree(d_reverse_edges));
    RAFT_CUDA_TRY(cudaFree(d_final_offsets));
    RAFT_CUDA_TRY(cudaFree(d_row_offsets));

    chunk_begin = chunk_end;
  }
}

inline DeviceRangeGraph build_one_range_graph(raft::resources const& res,
                                              GlobalDatasetView const& dataset,
                                              std::int64_t rank_l,
                                              std::int64_t rank_r,
                                              RangeGraphBuildParams const& params)
{
  RAFT_EXPECTS(rank_l >= 0 && rank_l <= rank_r && rank_r < dataset.rows,
               "invalid [rank_l, rank_r] for global dataset");
  const auto range_rows_i64 = rank_r - rank_l + 1;
  RAFT_EXPECTS(range_rows_i64 > 1, "range graph build requires at least two vectors");
  const auto range_rows   = static_cast<int>(range_rows_i64);
  const auto graph_degree = std::min(params.graph_degree, range_rows - 1);

  DeviceRangeGraph out;
  out.rank_l = rank_l;
  out.rows   = range_rows;
  out.degree = graph_degree;

  const auto edge_count = static_cast<std::size_t>(out.rows) * static_cast<std::size_t>(out.degree);
  RAFT_CUDA_TRY(cudaMalloc(&out.edges, sizeof(std::uint32_t) * edge_count));

  std::int64_t* d_offsets = nullptr;
  std::int64_t* d_rank_l  = nullptr;
  int* d_rows             = nullptr;
  int* d_degrees          = nullptr;
  const auto stream       = raft::resource::get_cuda_stream(res);
  const std::int64_t h_offset{0};
  RAFT_CUDA_TRY(cudaMalloc(&d_offsets, sizeof(std::int64_t)));
  RAFT_CUDA_TRY(cudaMalloc(&d_rank_l, sizeof(std::int64_t)));
  RAFT_CUDA_TRY(cudaMalloc(&d_rows, sizeof(int)));
  RAFT_CUDA_TRY(cudaMalloc(&d_degrees, sizeof(int)));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_offsets, &h_offset, sizeof(h_offset), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_rank_l, &rank_l, sizeof(rank_l), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(d_rows, &range_rows, sizeof(range_rows), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    d_degrees, &graph_degree, sizeof(graph_degree), cudaMemcpyHostToDevice, stream));

  DeviceRangeGraphPoolView pool{
    out.edges, d_offsets, d_rank_l, d_rows, d_degrees, 1, static_cast<std::int64_t>(edge_count)};
  build_range_graph_pool_edges_on_gpu(res,
                                      dataset,
                                      pool,
                                      std::vector<std::int64_t>{0, range_rows},
                                      std::vector<int>{graph_degree},
                                      std::vector<int>{params.intermediate_graph_degree},
                                      params);

  RAFT_CUDA_TRY(cudaFree(d_degrees));
  RAFT_CUDA_TRY(cudaFree(d_rows));
  RAFT_CUDA_TRY(cudaFree(d_rank_l));
  RAFT_CUDA_TRY(cudaFree(d_offsets));
  return out;
}

}  // namespace cuvs::neighbors::range_cagra::detail
