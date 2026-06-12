/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/range_cagra/range_cagra_segment_tree_workspace_search.cuh"

#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {
namespace {

namespace fs = std::filesystem;

struct FloatMatrix {
  std::vector<float> data;
  std::int64_t rows = 0;
  int dim           = 0;
};

struct Int64Matrix {
  std::vector<std::int64_t> data;
  std::int64_t rows = 0;
  int cols          = 0;
};

struct NpyHeader {
  std::string descr;
  bool fortran_order = false;
  std::vector<std::size_t> shape;
};

const char* getenv_or_null(const char* name) { return std::getenv(name); }

std::string getenv_or(const char* name, std::string fallback = {})
{
  if (auto value = getenv_or_null(name)) { return value; }
  return fallback;
}

int getenv_int_or(const char* name, int fallback)
{
  if (auto value = getenv_or_null(name)) { return std::stoi(value); }
  return fallback;
}

SegmentTreeWorkspaceLaunchMode parse_workspace_launch_mode(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (value.empty() || value == "device" || value == "device_no_presync" ||
      value == "no_presync") {
    return SegmentTreeWorkspaceLaunchMode::kDeviceCountNoPreSync;
  }
  if (value == "host" || value == "host_count" || value == "host_count_sync") {
    return SegmentTreeWorkspaceLaunchMode::kHostCountSync;
  }
  throw std::runtime_error(
    "RANGE_CAGRA_SEGMENT_WORKSPACE_MODE must be device_no_presync or host_count_sync");
}

const char* workspace_launch_mode_name(SegmentTreeWorkspaceLaunchMode mode)
{
  switch (mode) {
    case SegmentTreeWorkspaceLaunchMode::kDeviceCountNoPreSync: return "device_no_presync";
    case SegmentTreeWorkspaceLaunchMode::kHostCountSync: return "host_count_sync";
  }
  return "unknown";
}

FloatMatrix make_host_dataset(int rows, int dim)
{
  FloatMatrix out;
  out.rows = rows;
  out.dim  = dim;
  out.data.resize(static_cast<std::size_t>(rows) * dim);
  for (int i = 0; i < rows; ++i) {
    for (int d = 0; d < dim; ++d) {
      out.data[static_cast<std::size_t>(i) * dim + d] =
        std::sin(0.013f * static_cast<float>(i * 17 + d * 3)) +
        0.01f * static_cast<float>((i + d) % 7);
    }
  }
  return out;
}

FloatMatrix make_host_queries(FloatMatrix const& base, std::vector<std::int64_t> const& ids)
{
  FloatMatrix out;
  out.rows = static_cast<std::int64_t>(ids.size());
  out.dim  = base.dim;
  out.data.resize(static_cast<std::size_t>(out.rows) * out.dim);
  for (std::int64_t q = 0; q < out.rows; ++q) {
    const auto src = ids[static_cast<std::size_t>(q)];
    for (int d = 0; d < out.dim; ++d) {
      out.data[static_cast<std::size_t>(q) * out.dim + d] =
        base.data[static_cast<std::size_t>(src) * base.dim + d] +
        0.001f * static_cast<float>((q + d) % 3);
    }
  }
  return out;
}

auto copy_to_device(raft::resources const& res, FloatMatrix const& matrix)
{
  auto out = raft::make_device_matrix<float, int64_t>(res, matrix.rows, matrix.dim);
  RAFT_CUDA_TRY(cudaMemcpyAsync(out.data_handle(),
                                matrix.data.data(),
                                sizeof(float) * matrix.data.size(),
                                cudaMemcpyHostToDevice,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);
  return out;
}

void insert_topk_ref(float dist,
                     std::uint32_t id,
                     std::vector<float>& dists,
                     std::vector<std::uint32_t>& ids,
                     int topk)
{
  if (dist > dists[topk - 1] || (dist == dists[topk - 1] && id >= ids[topk - 1])) { return; }
  int pos = topk - 1;
  while (pos > 0 && (dist < dists[pos - 1] || (dist == dists[pos - 1] && id < ids[pos - 1]))) {
    dists[pos] = dists[pos - 1];
    ids[pos]   = ids[pos - 1];
    --pos;
  }
  dists[pos] = dist;
  ids[pos]   = id;
}

std::vector<std::uint32_t> exact_range_ids(FloatMatrix const& base,
                                           FloatMatrix const& queries,
                                           std::vector<std::int64_t> const& ranges,
                                           int query_id,
                                           int topk)
{
  std::vector<float> dists(static_cast<std::size_t>(topk), INFINITY);
  std::vector<std::uint32_t> ids(static_cast<std::size_t>(topk),
                                 std::numeric_limits<std::uint32_t>::max());
  const auto left  = ranges[static_cast<std::size_t>(query_id) * 2];
  const auto right = ranges[static_cast<std::size_t>(query_id) * 2 + 1];
  for (std::int64_t id = left; id <= right; ++id) {
    float dist = 0.0f;
    for (int d = 0; d < base.dim; ++d) {
      const float diff = base.data[static_cast<std::size_t>(id) * base.dim + d] -
                         queries.data[static_cast<std::size_t>(query_id) * queries.dim + d];
      dist += diff * diff;
    }
    insert_topk_ref(dist, static_cast<std::uint32_t>(id), dists, ids, topk);
  }
  return ids;
}

std::uint16_t read_u16_le(std::ifstream& in)
{
  unsigned char bytes[2];
  in.read(reinterpret_cast<char*>(bytes), 2);
  return static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8));
}

std::uint32_t read_u32_le(std::ifstream& in)
{
  unsigned char bytes[4];
  in.read(reinterpret_cast<char*>(bytes), 4);
  return static_cast<std::uint32_t>(bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) |
                                    (bytes[3] << 24));
}

std::string parse_descr(std::string const& header)
{
  const auto key          = header.find("descr");
  const auto colon        = header.find(':', key);
  const auto first_quote  = header.find_first_of("'\"", colon);
  const auto second_quote = header.find_first_of("'\"", first_quote + 1);
  if (key == std::string::npos || colon == std::string::npos || first_quote == std::string::npos ||
      second_quote == std::string::npos) {
    throw std::runtime_error("failed to parse npy descr");
  }
  return header.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::vector<std::size_t> parse_shape(std::string const& header)
{
  const auto key   = header.find("shape");
  const auto open  = header.find('(', key);
  const auto close = header.find(')', open);
  if (key == std::string::npos || open == std::string::npos || close == std::string::npos) {
    throw std::runtime_error("failed to parse npy shape");
  }
  std::vector<std::size_t> shape;
  std::stringstream ss(header.substr(open + 1, close - open - 1));
  std::string token;
  while (std::getline(ss, token, ',')) {
    const auto first = token.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) { continue; }
    const auto last = token.find_last_not_of(" \t\r\n");
    shape.push_back(static_cast<std::size_t>(std::stoull(token.substr(first, last - first + 1))));
  }
  return shape;
}

NpyHeader read_npy_header(std::ifstream& in, fs::path const& path)
{
  char magic[6];
  in.read(magic, 6);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not an npy file: " + path.string());
  }
  unsigned char version[2];
  in.read(reinterpret_cast<char*>(version), 2);
  std::uint32_t header_len = 0;
  if (version[0] == 1) {
    header_len = read_u16_le(in);
  } else if (version[0] == 2 || version[0] == 3) {
    header_len = read_u32_le(in);
  } else {
    throw std::runtime_error("unsupported npy version: " + path.string());
  }
  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) { throw std::runtime_error("truncated npy header: " + path.string()); }
  NpyHeader out;
  out.descr         = parse_descr(header);
  out.fortran_order = header.find("fortran_order") != std::string::npos &&
                      header.find("True", header.find("fortran_order")) != std::string::npos;
  out.shape = parse_shape(header);
  return out;
}

template <typename T>
std::vector<T> read_npy_payload(std::ifstream& in, fs::path const& path, NpyHeader const& header)
{
  std::size_t count = 1;
  for (auto dim : header.shape) {
    count *= dim;
  }
  std::vector<T> data(count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) { throw std::runtime_error("truncated npy payload: " + path.string()); }
  return data;
}

FloatMatrix read_npy_float_matrix(fs::path const& path, int max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open npy: " + path.string()); }
  auto header = read_npy_header(in, path);
  if ((header.descr != "<f4" && header.descr != "|f4") || header.fortran_order ||
      header.shape.size() != 2) {
    throw std::runtime_error("expected C-order float32 matrix: " + path.string());
  }
  FloatMatrix out;
  out.rows = static_cast<std::int64_t>(header.shape[0]);
  out.dim  = static_cast<int>(header.shape[1]);
  out.data = read_npy_payload<float>(in, path, header);
  if (max_rows > 0 && out.rows > max_rows) {
    out.rows = max_rows;
    out.data.resize(static_cast<std::size_t>(out.rows) * out.dim);
  }
  return out;
}

Int64Matrix read_npy_int64_matrix(fs::path const& path, int max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open npy: " + path.string()); }
  auto header = read_npy_header(in, path);
  if ((header.descr != "<i8" && header.descr != "|i8") || header.fortran_order ||
      header.shape.size() != 2) {
    throw std::runtime_error("expected C-order int64 matrix: " + path.string());
  }
  Int64Matrix out;
  out.rows = static_cast<std::int64_t>(header.shape[0]);
  out.cols = static_cast<int>(header.shape[1]);
  out.data = read_npy_payload<std::int64_t>(in, path, header);
  if (max_rows > 0 && out.rows > max_rows) {
    out.rows = max_rows;
    out.data.resize(static_cast<std::size_t>(out.rows) * out.cols);
  }
  return out;
}

std::vector<std::int64_t> read_npy_int64_vector(fs::path const& path, int max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open npy: " + path.string()); }
  auto header = read_npy_header(in, path);
  if ((header.descr != "<i8" && header.descr != "|i8") || header.fortran_order ||
      header.shape.size() != 1) {
    throw std::runtime_error("expected C-order int64 vector: " + path.string());
  }
  auto data = read_npy_payload<std::int64_t>(in, path, header);
  if (max_rows > 0 && data.size() > static_cast<std::size_t>(max_rows)) {
    data.resize(static_cast<std::size_t>(max_rows));
  }
  return data;
}

FloatMatrix read_fvecs(fs::path const& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("failed to open fvecs: " + path.string()); }
  in.seekg(0, std::ios::end);
  const auto bytes = static_cast<std::uintmax_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::int32_t dim32 = 0;
  in.read(reinterpret_cast<char*>(&dim32), sizeof(dim32));
  if (!in || dim32 <= 0) { throw std::runtime_error("invalid fvecs dim: " + path.string()); }
  const auto record_bytes =
    static_cast<std::uintmax_t>(sizeof(std::int32_t) + sizeof(float) * dim32);
  if (bytes % record_bytes != 0) {
    throw std::runtime_error("fvecs size is not divisible by row size: " + path.string());
  }
  FloatMatrix out;
  out.rows = static_cast<std::int64_t>(bytes / record_bytes);
  out.dim  = dim32;
  out.data.resize(static_cast<std::size_t>(out.rows) * out.dim);
  in.seekg(0, std::ios::beg);
  for (std::int64_t row = 0; row < out.rows; ++row) {
    std::int32_t row_dim = 0;
    in.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
    if (!in || row_dim != dim32) { throw std::runtime_error("inconsistent fvecs dim"); }
    in.read(reinterpret_cast<char*>(out.data.data() + static_cast<std::size_t>(row) * out.dim),
            sizeof(float) * out.dim);
    if (!in) { throw std::runtime_error("truncated fvecs: " + path.string()); }
  }
  return out;
}

double recall_at_k(std::vector<std::uint32_t> const& ids, Int64Matrix const& gt, int topk, int rows)
{
  std::int64_t hits  = 0;
  std::int64_t total = 0;
  for (int q = 0; q < rows; ++q) {
    std::unordered_set<std::int64_t> gt_set;
    for (int k = 0; k < topk; ++k) {
      const auto id = gt.data[static_cast<std::size_t>(q) * gt.cols + k];
      if (id >= 0) { gt_set.insert(id); }
    }
    total += static_cast<std::int64_t>(gt_set.size());
    for (int k = 0; k < topk; ++k) {
      const auto id = ids[static_cast<std::size_t>(q) * topk + k];
      if (gt_set.find(id) != gt_set.end()) { ++hits; }
    }
  }
  return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

void print_recall_breakdown(std::vector<std::uint32_t> const& ids,
                            Int64Matrix const& gt,
                            std::vector<std::int64_t> const& anchors,
                            int topk,
                            int rows,
                            std::size_t config_id)
{
  std::vector<std::int64_t> gt_rank_hits(static_cast<std::size_t>(topk), 0);
  std::int64_t queries_with_any_hit = 0;
  std::int64_t full_recall_queries  = 0;
  std::int64_t anchor_hit_any       = 0;
  std::int64_t anchor_at_result0    = 0;
  std::int64_t invalid_outputs      = 0;

  for (int q = 0; q < rows; ++q) {
    int query_hits = 0;
    for (int out_k = 0; out_k < topk; ++out_k) {
      const auto out_id = ids[static_cast<std::size_t>(q) * topk + out_k];
      if (out_id == std::numeric_limits<std::uint32_t>::max()) { ++invalid_outputs; }
    }
    for (int gt_k = 0; gt_k < topk; ++gt_k) {
      const auto gt_id = gt.data[static_cast<std::size_t>(q) * gt.cols + gt_k];
      bool found       = false;
      for (int out_k = 0; out_k < topk; ++out_k) {
        if (static_cast<std::int64_t>(ids[static_cast<std::size_t>(q) * topk + out_k]) == gt_id) {
          found = true;
          break;
        }
      }
      if (found) {
        ++gt_rank_hits[static_cast<std::size_t>(gt_k)];
        ++query_hits;
      }
    }
    if (query_hits > 0) { ++queries_with_any_hit; }
    if (query_hits == topk) { ++full_recall_queries; }
    if (!anchors.empty()) {
      const auto anchor = anchors[static_cast<std::size_t>(q)];
      bool found_anchor = false;
      for (int out_k = 0; out_k < topk; ++out_k) {
        if (static_cast<std::int64_t>(ids[static_cast<std::size_t>(q) * topk + out_k]) == anchor) {
          found_anchor = true;
          break;
        }
      }
      if (found_anchor) { ++anchor_hit_any; }
      if (static_cast<std::int64_t>(ids[static_cast<std::size_t>(q) * topk]) == anchor) {
        ++anchor_at_result0;
      }
    }
  }

  std::cout << "range_cagra_recall_breakdown,"
            << "search_config_id=" << config_id << ","
            << "queries_with_any_hit="
            << static_cast<double>(queries_with_any_hit) / static_cast<double>(rows) << ","
            << "full_recall_query_rate="
            << static_cast<double>(full_recall_queries) / static_cast<double>(rows) << ","
            << "invalid_outputs=" << invalid_outputs;
  if (!anchors.empty()) {
    std::cout << ",anchor_hit_any="
              << static_cast<double>(anchor_hit_any) / static_cast<double>(rows)
              << ",anchor_at_result0="
              << static_cast<double>(anchor_at_result0) / static_cast<double>(rows);
  }
  for (int k = 0; k < topk; ++k) {
    std::cout << ",gt_rank" << k << "_hit="
              << static_cast<double>(gt_rank_hits[static_cast<std::size_t>(k)]) /
                   static_cast<double>(rows);
  }
  std::cout << std::endl;
}

std::int64_t epoch_millis()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::system_clock::now().time_since_epoch())
    .count();
}

struct SearchSweepConfig {
  int ef                       = 0;
  int graph_iterations         = 0;
  int graph_search_concurrency = 0;
  int entry_count              = 0;
};

std::vector<SearchSweepConfig> parse_search_sweep_configs(int default_ef,
                                                          int default_graph_iterations,
                                                          int default_graph_search_concurrency,
                                                          int default_entry_count)
{
  std::vector<SearchSweepConfig> configs;
  const auto spec = getenv_or("RANGE_CAGRA_SEGMENT_SEARCH_SWEEP", "");
  if (spec.empty()) {
    configs.push_back({default_ef,
                       default_graph_iterations,
                       default_graph_search_concurrency,
                       default_entry_count});
    return configs;
  }

  std::stringstream outer(spec);
  std::string item;
  while (std::getline(outer, item, ';')) {
    if (item.empty()) { continue; }
    std::replace(item.begin(), item.end(), ',', ':');
    std::stringstream inner(item);
    std::string field;
    std::vector<int> values;
    while (std::getline(inner, field, ':')) {
      if (!field.empty()) { values.push_back(std::stoi(field)); }
    }
    if (values.size() != 4) {
      throw std::runtime_error(
        "RANGE_CAGRA_SEGMENT_SEARCH_SWEEP items must be ef:graph_iterations:search_width:entry");
    }
    configs.push_back({values[0], values[1], values[2], values[3]});
  }
  if (configs.empty()) {
    throw std::runtime_error("RANGE_CAGRA_SEGMENT_SEARCH_SWEEP did not contain any configs");
  }
  return configs;
}

SegmentTreeSearchSchedule parse_search_schedule(std::string const& value)
{
  if (value == "overlap") { return SegmentTreeSearchSchedule::kOverlap; }
  if (value == "graph_then_exact") { return SegmentTreeSearchSchedule::kGraphThenExact; }
  if (value == "exact_then_graph") { return SegmentTreeSearchSchedule::kExactThenGraph; }
  throw std::runtime_error(
    "RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE must be overlap, graph_then_exact, or exact_then_graph");
}

std::vector<std::pair<std::string, SegmentTreeSearchSchedule>> parse_search_schedule_sweep(
  std::string const& default_schedule)
{
  std::vector<std::pair<std::string, SegmentTreeSearchSchedule>> schedules;
  auto spec = getenv_or("RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE_SWEEP", "");
  if (spec.empty()) { spec = default_schedule; }
  std::replace(spec.begin(), spec.end(), ',', ';');
  std::stringstream outer(spec);
  std::string item;
  while (std::getline(outer, item, ';')) {
    if (item.empty()) { continue; }
    schedules.emplace_back(item, parse_search_schedule(item));
  }
  if (schedules.empty()) {
    throw std::runtime_error("RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE_SWEEP did not contain any schedules");
  }
  return schedules;
}

SegmentTreeSearchIterationPolicy parse_search_iteration_policy(std::string const& value)
{
  if (value == "uniform" || value.empty()) { return SegmentTreeSearchIterationPolicy::kUniform; }
  if (value == "lower_layers") { return SegmentTreeSearchIterationPolicy::kLowerLayers; }
  if (value == "upper_layers") { return SegmentTreeSearchIterationPolicy::kUpperLayers; }
  if (value == "layer_adaptive" || value == "adaptive") {
    return SegmentTreeSearchIterationPolicy::kLayerAdaptive;
  }
  throw std::runtime_error(
    "RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY must be uniform, lower_layers, "
    "upper_layers, or layer_adaptive");
}

std::vector<std::pair<std::string, SegmentTreeSearchIterationParams>>
parse_search_iteration_policy_sweep_spec(std::string spec,
                                         SegmentTreeSearchIterationParams const& base_params)
{
  std::vector<std::pair<std::string, SegmentTreeSearchIterationParams>> policies;
  std::replace(spec.begin(), spec.end(), ',', ';');
  std::stringstream outer(spec);
  std::string item;
  while (std::getline(outer, item, ';')) {
    if (item.empty()) { continue; }
    std::stringstream inner(item);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(inner, field, ':')) {
      if (!field.empty()) { fields.push_back(field); }
    }
    if (fields.empty()) { continue; }

    auto params   = base_params;
    params.policy = parse_search_iteration_policy(fields[0]);
    std::string label = fields[0];
    if (fields.size() > 1) {
      if (params.policy != SegmentTreeSearchIterationPolicy::kLayerAdaptive) {
        throw std::runtime_error(
          "parameterized search-iteration sweep items are only supported for layer_adaptive");
      }
      if (fields.size() != 3 && fields.size() != 4) {
        throw std::runtime_error(
          "layer_adaptive sweep items must be layer_adaptive:min:max[:granularity]");
      }
      params.adaptive_min_iterations = std::stoi(fields[1]);
      params.adaptive_max_iterations = std::stoi(fields[2]);
      if (fields.size() == 4) { params.adaptive_granularity = std::stoi(fields[3]); }
      label = "layer_adaptive_" + fields[1] + "_" + fields[2] + "_g" +
              std::to_string(params.adaptive_granularity);
    } else if (params.policy == SegmentTreeSearchIterationPolicy::kLayerAdaptive) {
      label = "layer_adaptive_" + std::to_string(params.adaptive_min_iterations) + "_" +
              std::to_string(params.adaptive_max_iterations) + "_g" +
              std::to_string(params.adaptive_granularity);
    }
    policies.emplace_back(label, params);
  }
  if (policies.empty()) {
    throw std::runtime_error(
      "RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY_SWEEP did not contain any policies");
  }
  return policies;
}

std::vector<std::pair<std::string, SegmentTreeSearchIterationParams>>
parse_search_iteration_policy_sweep(std::string const& default_policy_name,
                                    SegmentTreeSearchIterationParams const& base_params)
{
  auto spec = getenv_or("RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY_SWEEP", "");
  if (spec.empty()) { spec = default_policy_name; }
  return parse_search_iteration_policy_sweep_spec(spec, base_params);
}

std::vector<std::string> parse_workload_sweep(std::string spec, std::string const& default_workload)
{
  if (spec.empty()) { spec = default_workload; }
  std::replace(spec.begin(), spec.end(), ',', ';');
  std::vector<std::string> workloads;
  std::stringstream outer(spec);
  std::string item;
  while (std::getline(outer, item, ';')) {
    if (!item.empty()) { workloads.push_back(item); }
  }
  if (workloads.empty()) { workloads.push_back(default_workload); }
  return workloads;
}

struct DegreeStats {
  int count                       = 0;
  int graph_min                   = std::numeric_limits<int>::max();
  int graph_max                   = 0;
  std::int64_t graph_sum          = 0;
  int intermediate_min            = std::numeric_limits<int>::max();
  int intermediate_max            = 0;
  std::int64_t intermediate_sum   = 0;
};

void add_degree_stats(DegreeStats& stats, HostRangeGraphMeta const& meta)
{
  ++stats.count;
  stats.graph_min = std::min(stats.graph_min, meta.degree);
  stats.graph_max = std::max(stats.graph_max, meta.degree);
  stats.graph_sum += meta.degree;
  stats.intermediate_min = std::min(stats.intermediate_min, meta.intermediate_degree);
  stats.intermediate_max = std::max(stats.intermediate_max, meta.intermediate_degree);
  stats.intermediate_sum += meta.intermediate_degree;
}

DegreeStats summarize_degree_stats(SegmentTreeRangeCagraIndex const& index)
{
  DegreeStats stats;
  for (auto const& meta : index.graph_metas) {
    add_degree_stats(stats, meta);
  }
  if (stats.count == 0) {
    stats.graph_min        = 0;
    stats.intermediate_min = 0;
  }
  return stats;
}

void print_degree_layer_summary(SegmentTreeRangeCagraIndex const& index)
{
  const int max_layer = segment_tree_max_internal_layer(index.layout);
  std::vector<DegreeStats> layer_stats(static_cast<std::size_t>(max_layer + 1));
  for (auto const& meta : index.graph_metas) {
    const int layer = std::clamp(segment_node_layer_from_bottom(meta.range), 0, max_layer);
    add_degree_stats(layer_stats[static_cast<std::size_t>(layer)], meta);
  }
  for (int layer = 0; layer <= max_layer; ++layer) {
    auto const& stats = layer_stats[static_cast<std::size_t>(layer)];
    if (stats.count == 0) { continue; }
    std::cout << "range_cagra_degree_layer,"
              << "layer_from_bottom=" << layer << ","
              << "graph_count=" << stats.count << ","
              << "graph_degree_min=" << stats.graph_min << ","
              << "graph_degree_max=" << stats.graph_max << ","
              << "graph_degree_avg="
              << static_cast<double>(stats.graph_sum) / static_cast<double>(stats.count) << ","
              << "intermediate_degree_min=" << stats.intermediate_min << ","
              << "intermediate_degree_max=" << stats.intermediate_max << ","
              << "intermediate_degree_avg="
              << static_cast<double>(stats.intermediate_sum) / static_cast<double>(stats.count)
              << std::endl;
  }
}

TEST(RangeCagraSegmentTree, DecompositionUsesAtMostFourExactLeafTasks)
{
  auto layout = make_segment_tree_layout(257, 16, 17);
  for (std::int64_t node = 1; node < 2 * layout.leaf_base; ++node) {
    auto range = range_from_node_id(node, layout);
    if (range.valid() && range.size() > layout.leaf_size) {
      layout.graph_slot[static_cast<std::size_t>(node)] = static_cast<int>(node);
    }
  }

  for (std::int64_t left = 0; left < layout.rows; left += 7) {
    for (std::int64_t right = left; right < layout.rows; right += 11) {
      std::vector<ExactSearchTask> exact_tasks;
      std::vector<GraphSearchTask> graph_tasks;
      std::int64_t scanned = 0;
      decompose_query_range(0, left, right, layout, exact_tasks, graph_tasks, scanned);
      ASSERT_LE(exact_tasks.size(), 4);
      for (auto const& task : exact_tasks) {
        EXPECT_LE(task.left, task.right);
        EXPECT_GE(task.left, left);
        EXPECT_LE(task.right, right);
      }
    }
  }
}

TEST(RangeCagraSegmentTree, LayerAdaptiveDegreeDecreasesTowardLeaves)
{
  auto layout = make_segment_tree_layout(128, 16, 16);

  RangeGraphBuildParams params;
  params.graph_degree                  = 32;
  params.intermediate_graph_degree     = 96;
  params.layer_adaptive_degree         = true;
  params.min_graph_degree              = 8;
  params.min_intermediate_graph_degree = 24;
  params.degree_granularity            = 1;

  const auto root   = range_graph_degrees_for_node(layout, range_from_node_id(1, layout), params);
  const auto middle = range_graph_degrees_for_node(layout, range_from_node_id(2, layout), params);
  const auto bottom = range_graph_degrees_for_node(layout, range_from_node_id(4, layout), params);

  EXPECT_EQ(root.graph_degree, 32);
  EXPECT_EQ(root.intermediate_graph_degree, 96);
  EXPECT_GT(root.graph_degree, middle.graph_degree);
  EXPECT_GT(middle.graph_degree, bottom.graph_degree);
  EXPECT_EQ(bottom.graph_degree, 8);
  EXPECT_EQ(bottom.intermediate_graph_degree, 24);
}

TEST(RangeCagraSegmentTree, LayerAdaptiveSearchIterationsOnlyReduceBelowBase)
{
  SegmentTreeSearchIterationParams params;
  params.policy                  = SegmentTreeSearchIterationPolicy::kLayerAdaptive;
  params.adaptive_min_iterations = 6;
  params.adaptive_max_iterations = 24;
  params.adaptive_granularity    = 2;

  EXPECT_TRUE(graph_search_iteration_policy_needs_per_graph(params, 8));
  EXPECT_EQ(graph_search_iterations_for_layer(0, 4, 8, params), 2);
  EXPECT_EQ(graph_search_iterations_for_layer(4, 4, 8, params), 8);

  int previous_iterations = 0;
  for (int layer = 0; layer <= 4; ++layer) {
    const int iterations = graph_search_iterations_for_layer(layer, 4, 8, params);
    EXPECT_LE(iterations, 8);
    EXPECT_GE(iterations, previous_iterations);
    previous_iterations = iterations;
  }

  EXPECT_EQ(graph_search_iterations_for_layer(0, 4, 24, params), 6);
  EXPECT_EQ(graph_search_iterations_for_layer(4, 4, 24, params), 24);
  EXPECT_EQ(graph_search_iterations_for_layer(0, 4, 48, params), 12);
  EXPECT_EQ(graph_search_iterations_for_layer(4, 4, 48, params), 48);

  params.adaptive_min_iterations = 0;
  params.adaptive_max_iterations = 0;
  EXPECT_TRUE(graph_search_iteration_policy_needs_per_graph(params, 16));
  EXPECT_EQ(graph_search_iterations_for_layer(0, 4, 16, params), 4);
  EXPECT_EQ(graph_search_iterations_for_layer(4, 4, 16, params), 16);

  params.adaptive_min_iterations = 12;
  params.adaptive_max_iterations = 0;
  EXPECT_FALSE(graph_search_iteration_policy_needs_per_graph(params, 8));
  for (int layer = 0; layer <= 4; ++layer) {
    EXPECT_EQ(graph_search_iterations_for_layer(layer, 4, 8, params), 8);
  }
}

TEST(RangeCagraSegmentTree, SearchIterationPolicySweepParsesLayerAdaptiveRatios)
{
  SegmentTreeSearchIterationParams params;
  params.policy                  = SegmentTreeSearchIterationPolicy::kLayerAdaptive;
  params.adaptive_min_iterations = 12;
  params.adaptive_max_iterations = 32;
  params.adaptive_granularity    = 4;

  const auto policies =
    parse_search_iteration_policy_sweep_spec("uniform;layer_adaptive:10:32:4;adaptive:8:32:2", params);
  ASSERT_EQ(policies.size(), 3);

  EXPECT_EQ(policies[0].first, "uniform");
  EXPECT_EQ(policies[0].second.policy, SegmentTreeSearchIterationPolicy::kUniform);

  EXPECT_EQ(policies[1].first, "layer_adaptive_10_32_g4");
  EXPECT_EQ(policies[1].second.policy, SegmentTreeSearchIterationPolicy::kLayerAdaptive);
  EXPECT_EQ(policies[1].second.adaptive_min_iterations, 10);
  EXPECT_EQ(policies[1].second.adaptive_max_iterations, 32);
  EXPECT_EQ(policies[1].second.adaptive_granularity, 4);

  EXPECT_EQ(policies[2].first, "layer_adaptive_8_32_g2");
  EXPECT_EQ(policies[2].second.policy, SegmentTreeSearchIterationPolicy::kLayerAdaptive);
  EXPECT_EQ(policies[2].second.adaptive_min_iterations, 8);
  EXPECT_EQ(policies[2].second.adaptive_max_iterations, 32);
  EXPECT_EQ(policies[2].second.adaptive_granularity, 2);
}

TEST(RangeCagraSegmentTree, BuildsAndSearchesSyntheticSegmentTree)
{
  raft::resources res;

  constexpr int rows      = 128;
  constexpr int dim       = 16;
  constexpr int leaf_size = 16;
  constexpr int topk      = 5;

  auto base    = make_host_dataset(rows, dim);
  auto queries = make_host_queries(base, {7, 32, 55, 90, 110});
  std::vector<std::int64_t> ranges{
    3,
    12,
    5,
    40,
    16,
    63,
    0,
    127,
    31,
    96,
  };

  auto d_base    = copy_to_device(res, base);
  auto d_queries = copy_to_device(res, queries);
  GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree              = 8;
  build_params.intermediate_graph_degree = 16;
  build_params.nn_descent_iterations     = 5;

  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);
  EXPECT_GT(index.graph_pool.graph_count, 0);
  EXPECT_EQ(index.layout.leaf_size, leaf_size);

  std::vector<std::uint32_t> ids;
  std::vector<float> distances;
  SegmentTreeSearchStats stats;
  run_segment_tree_range_cagra_search(
    res,
    dataset,
    index,
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      d_queries.data_handle(), queries.rows, queries.dim),
    ranges,
    topk,
    32,
    8,
    32,
    64,
    ids,
    distances,
    &stats);

  EXPECT_GT(stats.exact_task_count, 0);
  EXPECT_GT(stats.graph_node_task_count, 0);
  ASSERT_EQ(ids.size(), static_cast<std::size_t>(queries.rows) * topk);
  for (int q = 0; q < queries.rows; ++q) {
    const auto left  = ranges[static_cast<std::size_t>(q) * 2];
    const auto right = ranges[static_cast<std::size_t>(q) * 2 + 1];
    bool any_result  = false;
    for (int k = 0; k < topk; ++k) {
      const auto id = ids[static_cast<std::size_t>(q) * topk + k];
      if (id == std::numeric_limits<std::uint32_t>::max()) { continue; }
      any_result = true;
      EXPECT_GE(static_cast<std::int64_t>(id), left);
      EXPECT_LE(static_cast<std::int64_t>(id), right);
    }
    EXPECT_TRUE(any_result);
  }

  auto exact_ids = exact_range_ids(base, queries, ranges, 0, topk);
  for (int k = 0; k < topk; ++k) {
    EXPECT_EQ(ids[static_cast<std::size_t>(k)], exact_ids[static_cast<std::size_t>(k)]);
  }
}

TEST(RangeCagraSegmentTree, BuildsLayerAdaptiveDegreePool)
{
  raft::resources res;

  constexpr int rows      = 128;
  constexpr int dim       = 16;
  constexpr int leaf_size = 16;

  auto base = make_host_dataset(rows, dim);
  auto d_base = copy_to_device(res, base);
  GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree                  = 16;
  build_params.intermediate_graph_degree     = 32;
  build_params.nn_descent_iterations         = 2;
  build_params.layer_adaptive_degree         = true;
  build_params.min_graph_degree              = 4;
  build_params.min_intermediate_graph_degree = 8;
  build_params.degree_granularity            = 1;

  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);
  ASSERT_GT(index.graph_pool.graph_count, 0);

  std::int64_t expected_edge_count = 0;
  bool saw_root_degree             = false;
  bool saw_bottom_degree           = false;
  for (auto const& meta : index.graph_metas) {
    expected_edge_count += meta.range.size() * static_cast<std::int64_t>(meta.degree);
    if (meta.node_id == 1) {
      EXPECT_EQ(meta.degree, 16);
      EXPECT_EQ(meta.intermediate_degree, 32);
      saw_root_degree = true;
    }
    if (segment_node_layer_from_bottom(meta.range) == 0) {
      EXPECT_EQ(meta.degree, 4);
      EXPECT_EQ(meta.intermediate_degree, 8);
      saw_bottom_degree = true;
    }
  }
  EXPECT_TRUE(saw_root_degree);
  EXPECT_TRUE(saw_bottom_degree);
  EXPECT_EQ(index.graph_pool.edge_count, expected_edge_count);
}

TEST(RangeCagraSegmentTree, FullRangeGraphKernelMatchesExactWhenEntriesCoverNode)
{
  raft::resources res;

  constexpr int rows      = 128;
  constexpr int dim       = 16;
  constexpr int leaf_size = 16;
  constexpr int topk      = 5;

  auto base    = make_host_dataset(rows, dim);
  auto queries = make_host_queries(base, {7, 55, 110});
  std::vector<std::int64_t> ranges{
    0,
    rows - 1,
    0,
    rows - 1,
    0,
    rows - 1,
  };

  auto d_base    = copy_to_device(res, base);
  auto d_queries = copy_to_device(res, queries);
  GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree              = 8;
  build_params.intermediate_graph_degree = 16;
  build_params.nn_descent_iterations     = 5;

  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);

  std::vector<std::uint32_t> ids;
  std::vector<float> distances;
  SegmentTreeSearchStats stats;
  run_segment_tree_range_cagra_search(
    res,
    dataset,
    index,
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      d_queries.data_handle(), queries.rows, queries.dim),
    ranges,
    topk,
    rows,
    rows,
    0,
    64,
    ids,
    distances,
    &stats);

  EXPECT_EQ(stats.exact_task_count, 0);
  EXPECT_EQ(stats.graph_node_task_count, queries.rows);
  for (int q = 0; q < queries.rows; ++q) {
    auto exact_ids = exact_range_ids(base, queries, ranges, q, topk);
    for (int k = 0; k < topk; ++k) {
      EXPECT_EQ(ids[static_cast<std::size_t>(q) * topk + k],
                exact_ids[static_cast<std::size_t>(k)]);
    }
  }
}

TEST(RangeCagraSegmentTree, GraphSearchRadixTopkPathRuns)
{
  raft::resources res;

  constexpr int rows      = 256;
  constexpr int dim       = 16;
  constexpr int leaf_size = 32;
  constexpr int topk      = 5;

  auto base    = make_host_dataset(rows, dim);
  auto queries = make_host_queries(base, {7, 55, 110, 180});
  std::vector<std::int64_t> ranges{
    0,
    rows - 1,
    0,
    rows - 1,
    0,
    rows - 1,
    0,
    rows - 1,
  };

  auto d_base    = copy_to_device(res, base);
  auto d_queries = copy_to_device(res, queries);
  GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree              = 16;
  build_params.intermediate_graph_degree = 32;
  build_params.nn_descent_iterations     = 5;

  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);

  std::vector<std::uint32_t> ids;
  std::vector<float> distances;
  SegmentTreeSearchStats stats;
  run_segment_tree_range_cagra_search(
    res,
    dataset,
    index,
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      d_queries.data_handle(), queries.rows, queries.dim),
    ranges,
    topk,
    64,
    32,
    4,
    64,
    ids,
    distances,
    &stats,
    17);

  EXPECT_EQ(stats.exact_task_count, 0);
  EXPECT_EQ(stats.graph_node_task_count, queries.rows);
  for (int q = 0; q < queries.rows; ++q) {
    bool any_result = false;
    for (int k = 0; k < topk; ++k) {
      const auto offset = static_cast<std::size_t>(q) * topk + k;
      const auto id     = ids[offset];
      if (id == std::numeric_limits<std::uint32_t>::max()) { continue; }
      any_result = true;
      EXPECT_LT(id, static_cast<std::uint32_t>(rows));
      EXPECT_TRUE(std::isfinite(distances[offset]));
    }
    EXPECT_TRUE(any_result);
  }
}

TEST(RangeCagraSegmentTree, GeneralBuildIsIndependentFromExactOnlyQueryRanges)
{
  raft::resources res;

  constexpr int rows      = 128;
  constexpr int dim       = 16;
  constexpr int leaf_size = 16;
  constexpr int topk      = 5;

  auto base    = make_host_dataset(rows, dim);
  auto queries = make_host_queries(base, {7, 18, 35});
  std::vector<std::int64_t> ranges{
    3,
    12,
    17,
    20,
    33,
    39,
  };

  auto d_base    = copy_to_device(res, base);
  auto d_queries = copy_to_device(res, queries);
  GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree              = 8;
  build_params.intermediate_graph_degree = 16;
  build_params.nn_descent_iterations     = 5;

  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);
  EXPECT_EQ(index.graph_pool.graph_count, 7);
  EXPECT_GT(index.graph_pool.edge_count, 0);

  std::vector<std::uint32_t> ids;
  std::vector<float> distances;
  SegmentTreeSearchStats stats;
  run_segment_tree_range_cagra_search(
    res,
    dataset,
    index,
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      d_queries.data_handle(), queries.rows, queries.dim),
    ranges,
    topk,
    32,
    8,
    32,
    64,
    ids,
    distances,
    &stats);

  EXPECT_GT(stats.exact_task_count, 0);
  EXPECT_EQ(stats.graph_node_task_count, 0);
  for (int q = 0; q < queries.rows; ++q) {
    auto exact_ids = exact_range_ids(base, queries, ranges, q, topk);
    for (int k = 0; k < topk; ++k) {
      EXPECT_EQ(ids[static_cast<std::size_t>(q) * topk + k],
                exact_ids[static_cast<std::size_t>(k)]);
    }
  }
}

TEST(RangeCagraSegmentTree, GeneralBuildIncludesAllReusableInternalNodes)
{
  raft::resources res;

  constexpr int rows      = 128;
  constexpr int dim       = 16;
  constexpr int leaf_size = 16;

  auto base   = make_host_dataset(rows, dim);
  auto d_base = copy_to_device(res, base);
  GlobalDatasetView dataset{d_base.data_handle(), rows, dim, dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree              = 8;
  build_params.intermediate_graph_degree = 16;
  build_params.nn_descent_iterations     = 5;

  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);
  ASSERT_EQ(index.graph_metas.size(), 7);
  EXPECT_EQ(index.graph_pool.graph_count, 7);

  auto has_range = [&](std::int64_t vec_l, std::int64_t vec_r) {
    return std::any_of(index.graph_metas.begin(), index.graph_metas.end(), [&](auto const& meta) {
      return meta.range.vec_l == vec_l && meta.range.vec_r == vec_r;
    });
  };
  EXPECT_TRUE(has_range(0, 127));
  EXPECT_TRUE(has_range(32, 63));
  EXPECT_TRUE(has_range(64, 95));
}

TEST(RangeCagraSegmentTree, OptionalRfannWorkloadSmoke)
{
  const auto base_path     = getenv_or("RANGE_CAGRA_SEGMENT_BASE", "data/msong/msong_base.fvecs");
  const auto workload_path = getenv_or("RANGE_CAGRA_SEGMENT_WORKLOAD",
                                       "generated_queries/order_range_raw_attr/msong/pos_w50");
  const auto workload_paths =
    parse_workload_sweep(getenv_or("RANGE_CAGRA_SEGMENT_WORKLOAD_SWEEP", ""), workload_path);
  if (!fs::exists(base_path)) {
    GTEST_SKIP() << "symlink rfann data/generated_queries or set RANGE_CAGRA_SEGMENT_BASE and "
                    "RANGE_CAGRA_SEGMENT_WORKLOAD";
  }
  for (auto const& path : workload_paths) {
    if (!fs::is_directory(path)) {
      GTEST_SKIP() << "missing workload directory in RANGE_CAGRA_SEGMENT_WORKLOAD_SWEEP: " << path;
    }
  }

  const int max_queries = getenv_int_or("RANGE_CAGRA_SEGMENT_MAX_QUERIES", 10000);
  const int leaf_size   = getenv_int_or("RANGE_CAGRA_SEGMENT_LEAF_SIZE", 1000);
  const int topk        = getenv_int_or("RANGE_CAGRA_SEGMENT_TOPK", 10);
  const int ef          = getenv_int_or("RANGE_CAGRA_SEGMENT_EF", 64);
  ASSERT_GT(max_queries, 0);
  ASSERT_GT(leaf_size, 0);
  ASSERT_GT(topk, 0);
  ASSERT_LE(topk, kRangeCagraMaxTopK);
  ASSERT_GE(ef, topk);

  auto base    = read_fvecs(base_path);
  raft::resources res;
  auto d_base    = copy_to_device(res, base);
  GlobalDatasetView dataset{d_base.data_handle(), base.rows, base.dim, base.dim};

  RangeGraphBuildParams build_params;
  build_params.graph_degree = getenv_int_or("RANGE_CAGRA_SEGMENT_GRAPH_DEGREE", 32);
  build_params.intermediate_graph_degree =
    getenv_int_or("RANGE_CAGRA_SEGMENT_INTERMEDIATE_GRAPH_DEGREE", 96);
  build_params.nn_descent_iterations         = getenv_int_or("RANGE_CAGRA_SEGMENT_NN_DESCENT_ITERS", 20);
  build_params.layer_adaptive_degree =
    getenv_int_or("RANGE_CAGRA_SEGMENT_LAYER_ADAPTIVE_DEGREE", 0) != 0;
  build_params.min_graph_degree =
    getenv_int_or("RANGE_CAGRA_SEGMENT_MIN_GRAPH_DEGREE", 0);
  build_params.min_intermediate_graph_degree =
    getenv_int_or("RANGE_CAGRA_SEGMENT_MIN_INTERMEDIATE_GRAPH_DEGREE", 0);
  build_params.degree_granularity =
    getenv_int_or("RANGE_CAGRA_SEGMENT_DEGREE_GRANULARITY", 8);
  const auto build_algo = getenv_or("RANGE_CAGRA_SEGMENT_BUILD_ALGO", "flat_gnnd");
  if (build_algo == "segmented_gnnd" || build_algo == "segmented") {
    build_params.build_algorithm = RangeGraphBuildAlgorithm::kSegmentedGnnd;
  } else if (build_algo != "flat_gnnd" && build_algo != "flat") {
    throw std::runtime_error("RANGE_CAGRA_SEGMENT_BUILD_ALGO must be flat_gnnd or segmented_gnnd");
  }

  const int graph_iterations         = getenv_int_or("RANGE_CAGRA_SEGMENT_GRAPH_ITERATIONS", ef);
  const int graph_search_concurrency = getenv_int_or("RANGE_CAGRA_SEGMENT_SEARCH_CONCURRENCY", 1);
  const int search_repeats           = getenv_int_or("RANGE_CAGRA_SEGMENT_SEARCH_REPEATS", 1);
  const int entry_count              = getenv_int_or("RANGE_CAGRA_SEGMENT_ENTRY_COUNT", 32);
  const int exact_threads            = getenv_int_or("RANGE_CAGRA_SEGMENT_EXACT_THREADS", 128);
  const int graph_threads            = getenv_int_or("RANGE_CAGRA_SEGMENT_GRAPH_THREADS", 128);
  const bool graph_profile_enabled   = getenv_int_or("RANGE_CAGRA_SEGMENT_GRAPH_PROFILE", 0) != 0;
  const int low_layer_search_layers =
    getenv_int_or("RANGE_CAGRA_SEGMENT_LOW_LAYER_SEARCH_LAYERS", 0);
  const int low_layer_graph_iterations =
    getenv_int_or("RANGE_CAGRA_SEGMENT_LOW_LAYER_GRAPH_ITERATIONS", 0);
  const int upper_layer_search_layers =
    getenv_int_or("RANGE_CAGRA_SEGMENT_UPPER_LAYER_SEARCH_LAYERS", 0);
  const int upper_layer_graph_iterations =
    getenv_int_or("RANGE_CAGRA_SEGMENT_UPPER_LAYER_GRAPH_ITERATIONS", 0);
  const int adaptive_min_graph_iterations =
    getenv_int_or("RANGE_CAGRA_SEGMENT_ADAPTIVE_MIN_GRAPH_ITERATIONS", 0);
  const int adaptive_max_graph_iterations =
    getenv_int_or("RANGE_CAGRA_SEGMENT_ADAPTIVE_MAX_GRAPH_ITERATIONS", 0);
  const int adaptive_iteration_granularity =
    getenv_int_or("RANGE_CAGRA_SEGMENT_ADAPTIVE_ITERATION_GRANULARITY", 1);
  auto search_iteration_policy_value =
    getenv_or("RANGE_CAGRA_SEGMENT_SEARCH_ITERATION_POLICY", "");
  if (search_iteration_policy_value.empty()) {
    if (upper_layer_search_layers > 0 && upper_layer_graph_iterations > 0) {
      search_iteration_policy_value = "upper_layers";
    } else if (adaptive_max_graph_iterations > 0) {
      search_iteration_policy_value = "layer_adaptive";
    } else if (low_layer_search_layers > 0 && low_layer_graph_iterations > 0) {
      search_iteration_policy_value = "lower_layers";
    } else {
      search_iteration_policy_value = "uniform";
    }
  }
  SegmentTreeSearchIterationParams search_iteration_params;
  search_iteration_params.policy = parse_search_iteration_policy(search_iteration_policy_value);
  search_iteration_params.lower_layer_count      = low_layer_search_layers;
  search_iteration_params.lower_layer_iterations = low_layer_graph_iterations;
  search_iteration_params.upper_layer_count      = upper_layer_search_layers;
  search_iteration_params.upper_layer_iterations = upper_layer_graph_iterations;
  search_iteration_params.adaptive_min_iterations = adaptive_min_graph_iterations;
  search_iteration_params.adaptive_max_iterations = adaptive_max_graph_iterations;
  search_iteration_params.adaptive_granularity    = adaptive_iteration_granularity;
  const auto search_schedule_name = getenv_or("RANGE_CAGRA_SEGMENT_SEARCH_SCHEDULE", "overlap");
  const auto search_schedules     = parse_search_schedule_sweep(search_schedule_name);
  const auto search_iteration_policies =
    parse_search_iteration_policy_sweep(search_iteration_policy_value, search_iteration_params);
  const auto workspace_mode =
    parse_workspace_launch_mode(getenv_or("RANGE_CAGRA_SEGMENT_WORKSPACE_MODE", "device_no_presync"));
  ASSERT_GT(search_repeats, 0);
  ASSERT_GT(entry_count, 0);
  ASSERT_GT(exact_threads, 0);
  ASSERT_GT(graph_threads, 0);
  ASSERT_GE(low_layer_search_layers, 0);
  ASSERT_GE(low_layer_graph_iterations, 0);
  ASSERT_GE(upper_layer_search_layers, 0);
  ASSERT_GE(upper_layer_graph_iterations, 0);
  ASSERT_GE(adaptive_min_graph_iterations, 0);
  ASSERT_GE(adaptive_max_graph_iterations, 0);
  ASSERT_GE(adaptive_iteration_granularity, 0);
  const auto search_configs =
    parse_search_sweep_configs(ef, graph_iterations, graph_search_concurrency, entry_count);

  std::cout << "range_cagra_phase,phase=build_start,epoch_ms=" << epoch_millis()
            << ",build_algo=" << build_algo
            << ",layer_adaptive_degree=" << (build_params.layer_adaptive_degree ? 1 : 0)
            << ",min_graph_degree=" << build_params.min_graph_degree
            << ",min_intermediate_graph_degree=" << build_params.min_intermediate_graph_degree
            << ",degree_granularity=" << build_params.degree_granularity << std::endl;
  auto index = build_segment_tree_range_cagra(res, dataset, leaf_size, build_params);
  print_degree_layer_summary(index);
  const auto degree_stats = summarize_degree_stats(index);
  SegmentTreeRangeCagraSearchWorkspace search_workspace;
  std::cout << "range_cagra_phase,phase=build_end,epoch_ms=" << epoch_millis()
            << ",build_seconds=" << index.build_seconds << std::endl;

  std::cout << "range_cagra_phase,phase=search_start,epoch_ms=" << epoch_millis()
            << ",search_repeats=" << search_repeats << ",search_configs=" << search_configs.size()
            << ",search_schedules=" << search_schedules.size()
            << ",search_iteration_policies=" << search_iteration_policies.size()
            << ",search_iteration_policy=" << search_iteration_policy_value
            << ",low_layer_search_layers=" << low_layer_search_layers
            << ",low_layer_graph_iterations=" << low_layer_graph_iterations
            << ",upper_layer_search_layers=" << upper_layer_search_layers
            << ",upper_layer_graph_iterations=" << upper_layer_graph_iterations
            << ",adaptive_min_graph_iterations=" << adaptive_min_graph_iterations
            << ",adaptive_max_graph_iterations=" << adaptive_max_graph_iterations
            << ",adaptive_iteration_granularity=" << adaptive_iteration_granularity
            << ",search_workspace_mode=" << workspace_launch_mode_name(workspace_mode)
            << std::endl;
  for (auto const& active_workload_path : workload_paths) {
    auto queries =
      read_npy_float_matrix(fs::path(active_workload_path) / "queries.npy", max_queries);
    auto ranges =
      read_npy_int64_matrix(fs::path(active_workload_path) / "range_ids.npy", max_queries);
    auto gt = read_npy_int64_matrix(fs::path(active_workload_path) / "gt.npy", max_queries);
    std::vector<std::int64_t> anchors;
    const auto anchor_path = fs::path(active_workload_path) / "anchor_ids.npy";
    if (fs::exists(anchor_path)) { anchors = read_npy_int64_vector(anchor_path, max_queries); }
    ASSERT_EQ(queries.dim, base.dim);
    ASSERT_EQ(ranges.cols, 2);
    ASSERT_EQ(ranges.rows, queries.rows);
    ASSERT_EQ(gt.rows, queries.rows);
    ASSERT_GE(gt.cols, topk);
    auto d_queries = copy_to_device(res, queries);
    std::cout << "range_cagra_phase,phase=workload_start,epoch_ms=" << epoch_millis()
              << ",workload=" << active_workload_path << ",nq=" << queries.rows << std::endl;
    for (auto const& schedule_config : search_schedules) {
      const auto& search_schedule_name = schedule_config.first;
      const auto search_schedule       = schedule_config.second;
      for (std::size_t policy_id = 0; policy_id < search_iteration_policies.size(); ++policy_id) {
        const auto& policy_config                  = search_iteration_policies[policy_id];
        const auto& active_search_iteration_name   = policy_config.first;
        const auto& active_search_iteration_params = policy_config.second;
        for (std::size_t config_id = 0; config_id < search_configs.size(); ++config_id) {
          const auto config = search_configs[config_id];
          ASSERT_GE(config.ef, topk);
          ASSERT_GE(config.graph_iterations, 0);
          ASSERT_GT(config.graph_search_concurrency, 0);
          ASSERT_GT(config.entry_count, 0);

          std::vector<std::uint32_t> ids;
          std::vector<std::uint32_t> best_ids;
          std::vector<float> distances;
          SegmentTreeSearchStats best_stats;
          double total_search_seconds = 0.0;
          double best_search_seconds  = std::numeric_limits<double>::infinity();

          std::cout << "range_cagra_phase,phase=search_config_start,epoch_ms=" << epoch_millis()
                    << ",search_config_id=" << config_id
                    << ",search_iteration_policy_id=" << policy_id << ",ef=" << config.ef
                    << ",graph_iterations=" << config.graph_iterations
                    << ",graph_search_concurrency=" << config.graph_search_concurrency
                    << ",entry_count=" << config.entry_count
                    << ",search_schedule=" << search_schedule_name
                    << ",search_iteration_policy=" << active_search_iteration_name
                    << ",low_layer_search_layers=" << low_layer_search_layers
                    << ",low_layer_graph_iterations=" << low_layer_graph_iterations
                    << ",upper_layer_search_layers=" << upper_layer_search_layers
                    << ",upper_layer_graph_iterations=" << upper_layer_graph_iterations
                    << ",adaptive_min_graph_iterations=" << adaptive_min_graph_iterations
                    << ",adaptive_max_graph_iterations=" << adaptive_max_graph_iterations
                    << ",adaptive_iteration_granularity=" << adaptive_iteration_granularity
                    << std::endl;
          for (int repeat = 0; repeat < search_repeats; ++repeat) {
            SegmentTreeSearchStats repeat_stats;
            run_segment_tree_range_cagra_search_with_workspace(
              res,
              dataset,
              index,
              raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
                d_queries.data_handle(), queries.rows, queries.dim),
              ranges.data,
              topk,
              config.ef,
              config.entry_count,
              config.graph_iterations,
              exact_threads,
              ids,
              distances,
              search_workspace,
              &repeat_stats,
              config.graph_search_concurrency,
              graph_profile_enabled,
              graph_threads,
              search_schedule,
              workspace_mode,
              active_search_iteration_params);
            total_search_seconds += repeat_stats.search_seconds;
            if (repeat_stats.search_seconds < best_search_seconds) {
              best_search_seconds = repeat_stats.search_seconds;
              best_stats          = repeat_stats;
              best_ids            = ids;
            }
          }
          const auto stats = best_stats;
          const double avg_search_seconds =
            total_search_seconds / static_cast<double>(search_repeats);
          std::cout << "range_cagra_phase,phase=search_config_end,epoch_ms=" << epoch_millis()
                    << ",search_config_id=" << config_id
                    << ",search_iteration_policy_id=" << policy_id
                    << ",search_repeats=" << search_repeats
                    << ",total_search_seconds=" << total_search_seconds
                    << ",avg_search_seconds=" << avg_search_seconds
                    << ",best_search_seconds=" << best_search_seconds << std::endl;

          int violations = 0;
          for (int q = 0; q < queries.rows; ++q) {
            const auto left  = ranges.data[static_cast<std::size_t>(q) * 2];
            const auto right = ranges.data[static_cast<std::size_t>(q) * 2 + 1];
            for (int k = 0; k < topk; ++k) {
              const auto id = best_ids[static_cast<std::size_t>(q) * topk + k];
              if (id != std::numeric_limits<std::uint32_t>::max() &&
                  (static_cast<std::int64_t>(id) < left || static_cast<std::int64_t>(id) > right)) {
                ++violations;
              }
            }
          }
          const double recall = recall_at_k(best_ids, gt, topk, static_cast<int>(queries.rows));
          print_recall_breakdown(
            best_ids, gt, anchors, topk, static_cast<int>(queries.rows), config_id);
          std::cout
            << "range_cagra_segment_tree,"
            << "search_config_id=" << config_id << ","
            << "search_iteration_policy_id=" << policy_id << ","
            << "base=" << base_path << ","
            << "workload=" << active_workload_path << ","
            << "rows=" << base.rows << ","
            << "dim=" << base.dim << ","
            << "nq=" << queries.rows << ","
            << "topk=" << topk << ","
            << "leaf_size=" << leaf_size << ","
            << "index_mode=general,"
            << "build_algo=" << build_algo << ","
            << "layer_adaptive_degree=" << (build_params.layer_adaptive_degree ? 1 : 0) << ","
            << "requested_graph_degree=" << build_params.graph_degree << ","
            << "requested_intermediate_graph_degree=" << build_params.intermediate_graph_degree
            << ","
            << "min_graph_degree=" << build_params.min_graph_degree << ","
            << "min_intermediate_graph_degree=" << build_params.min_intermediate_graph_degree << ","
            << "degree_granularity=" << build_params.degree_granularity << ","
            << "graph_degree_min=" << degree_stats.graph_min << ","
            << "graph_degree_max=" << degree_stats.graph_max << ","
            << "graph_degree_avg="
            << (degree_stats.count > 0 ? static_cast<double>(degree_stats.graph_sum) /
                                           static_cast<double>(degree_stats.count)
                                       : 0.0)
            << ","
            << "intermediate_graph_degree_min=" << degree_stats.intermediate_min << ","
            << "intermediate_graph_degree_max=" << degree_stats.intermediate_max << ","
            << "intermediate_graph_degree_avg="
            << (degree_stats.count > 0 ? static_cast<double>(degree_stats.intermediate_sum) /
                                           static_cast<double>(degree_stats.count)
                                       : 0.0)
            << ","
            << "nn_descent_iterations=" << build_params.nn_descent_iterations << ","
            << "graph_search_concurrency=" << config.graph_search_concurrency << ","
            << "search_width=" << config.graph_search_concurrency << ","
            << "exact_threads=" << exact_threads << ","
            << "graph_threads=" << graph_threads << ","
            << "search_schedule=" << search_schedule_name << ","
            << "search_workspace_mode=" << workspace_launch_mode_name(workspace_mode) << ","
            << "search_iteration_policy="
            << search_iteration_policy_name(stats.search_iteration_policy) << ","
            << "search_iteration_policy_label=" << active_search_iteration_name << ","
            << "search_iteration_base_graph_iterations="
            << stats.search_iteration_base_graph_iterations << ","
            << "search_iteration_min_graph_iterations="
            << stats.search_iteration_min_graph_iterations << ","
            << "search_iteration_max_graph_iterations="
            << stats.search_iteration_max_graph_iterations << ","
            << "search_iteration_avg_graph_iterations="
            << stats.search_iteration_avg_graph_iterations << ","
            << "search_iteration_max_graph_layer=" << stats.search_iteration_max_graph_layer << ","
            << "search_iteration_override_graph_count="
            << stats.search_iteration_override_graph_count << ","
            << "low_layer_search_layers=" << stats.low_layer_search_layers << ","
            << "low_layer_graph_iterations=" << stats.low_layer_graph_iterations << ","
            << "low_layer_graph_count=" << stats.low_layer_graph_count << ","
            << "upper_layer_search_layers=" << stats.upper_layer_search_layers << ","
            << "upper_layer_graph_iterations=" << stats.upper_layer_graph_iterations << ","
            << "upper_layer_graph_count=" << stats.upper_layer_graph_count << ","
            << "search_repeats=" << search_repeats << ","
            << "ef=" << config.ef << ","
            << "graph_iterations=" << config.graph_iterations << ","
            << "entry_count=" << config.entry_count << ","
            << "graph_count=" << index.graph_pool.graph_count << ","
            << "edge_count=" << index.graph_pool.edge_count << ","
            << "build_seconds=" << index.build_seconds << ","
            << "exact_tasks=" << stats.exact_task_count << ","
            << "graph_node_tasks=" << stats.graph_node_task_count << ","
            << "exact_vectors_scanned=" << stats.exact_vectors_scanned << ","
            << "exact_seconds=" << stats.exact_seconds << ","
            << "graph_seconds=" << stats.graph_seconds << ","
            << "merge_seconds=" << stats.merge_seconds << ","
            << "search_seconds=" << stats.search_seconds << ","
            << "avg_search_seconds=" << avg_search_seconds << ","
            << "best_search_seconds=" << best_search_seconds << ","
            << "avg_qps=" << static_cast<double>(queries.rows) / avg_search_seconds << ","
            << "best_qps=" << static_cast<double>(queries.rows) / best_search_seconds << ","
            << "recall_at_k=" << recall << ","
            << "filter_violations=" << violations << std::endl;
          if (graph_profile_enabled) {
            const auto& p = stats.graph_profile;
            std::cout << "range_cagra_graph_profile,"
                      << "search_config_id=" << config_id << ","
                      << "search_iteration_policy_id=" << policy_id << ","
                      << "base=" << base_path << ","
                      << "workload=" << active_workload_path << ","
                      << "search_schedule=" << search_schedule_name << ","
                      << "search_iteration_policy="
                      << search_iteration_policy_name(stats.search_iteration_policy) << ","
                      << "search_iteration_policy_label=" << active_search_iteration_name << ","
                      << "search_iteration_base_graph_iterations="
                      << stats.search_iteration_base_graph_iterations << ","
                      << "search_iteration_min_graph_iterations="
                      << stats.search_iteration_min_graph_iterations << ","
                      << "search_iteration_max_graph_iterations="
                      << stats.search_iteration_max_graph_iterations << ","
                      << "search_iteration_avg_graph_iterations="
                      << stats.search_iteration_avg_graph_iterations << ","
                      << "search_iteration_max_graph_layer="
                      << stats.search_iteration_max_graph_layer << ","
                      << "search_iteration_override_graph_count="
                      << stats.search_iteration_override_graph_count << ","
                      << "low_layer_search_layers=" << stats.low_layer_search_layers << ","
                      << "low_layer_graph_iterations=" << stats.low_layer_graph_iterations << ","
                      << "low_layer_graph_count=" << stats.low_layer_graph_count << ","
                      << "upper_layer_search_layers=" << stats.upper_layer_search_layers << ","
                      << "upper_layer_graph_iterations=" << stats.upper_layer_graph_iterations
                      << ","
                      << "upper_layer_graph_count=" << stats.upper_layer_graph_count << ","
                      << "ef=" << config.ef << ","
                      << "graph_iterations=" << config.graph_iterations << ","
                      << "graph_search_concurrency=" << config.graph_search_concurrency << ","
                      << "entry_count=" << config.entry_count << ","
                      << "graph_tasks=" << p.graph_tasks << ","
                      << "iterations_started=" << p.iterations_started << ","
                      << "iterations_completed=" << p.iterations_completed << ","
                      << "terminated_iterations=" << p.terminated_iterations << ","
                      << "hash_reset_count=" << p.hash_reset_count << ","
                      << "initial_candidates=" << p.initial_candidates << ","
                      << "candidate_slots=" << p.candidate_slots << ","
                      << "valid_child_candidates=" << p.valid_child_candidates << ","
                      << "inserted_child_candidates=" << p.inserted_child_candidates << ","
                      << "duplicate_child_candidates=" << p.duplicate_child_candidates << ","
                      << "distance_evaluations=" << p.distance_evaluations << ","
                      << "cycles_total=" << p.cycles_total << ","
                      << "cycles_query_init=" << p.cycles_query_init << ","
                      << "cycles_initial_prepare=" << p.cycles_initial_prepare << ","
                      << "cycles_initial_distance=" << p.cycles_initial_distance << ","
                      << "cycles_initial_merge=" << p.cycles_initial_merge << ","
                      << "cycles_hash_reset=" << p.cycles_hash_reset << ","
                      << "cycles_pickup=" << p.cycles_pickup << ","
                      << "cycles_clear_candidates=" << p.cycles_clear_candidates << ","
                      << "cycles_expand_prepare=" << p.cycles_expand_prepare << ","
                      << "cycles_expand_distance=" << p.cycles_expand_distance << ","
                      << "cycles_iter_merge=" << p.cycles_iter_merge << ","
                      << "cycles_output=" << p.cycles_output << std::endl;
          }
          EXPECT_EQ(violations, 0);
        }
      }
    }
    std::cout << "range_cagra_phase,phase=workload_end,epoch_ms=" << epoch_millis()
              << ",workload=" << active_workload_path << std::endl;
  }
  std::cout << "range_cagra_phase,phase=search_end,epoch_ms=" << epoch_millis()
            << ",search_repeats=" << search_repeats << ",search_configs=" << search_configs.size()
            << ",search_schedules=" << search_schedules.size()
            << ",search_iteration_policies=" << search_iteration_policies.size()
            << std::endl;
}

}  // namespace
}  // namespace cuvs::neighbors::range_cagra::detail
