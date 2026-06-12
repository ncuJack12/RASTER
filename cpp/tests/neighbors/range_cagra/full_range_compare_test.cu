/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/range_cagra/range_cagra_build.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace cuvs::neighbors::range_cagra::detail {
namespace {

struct FloatMatrix {
  std::vector<float> data;
  std::int64_t rows = 0;
  int dim           = 0;
};

struct IdMatrix {
  std::vector<std::uint32_t> data;
  std::int64_t rows = 0;
  int dim           = 0;
};

struct NpyHeader {
  std::string descr;
  bool fortran_order = false;
  std::vector<std::size_t> shape;
};

struct RecallStats {
  double recall           = 0.0;
  std::int64_t hits       = 0;
  std::int64_t total      = 0;
  std::int64_t mismatches = 0;
};

struct SearchStats {
  double seconds = 0.0;
  double qps     = 0.0;
  RecallStats recall;
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

std::int64_t getenv_i64_or(const char* name, std::int64_t fallback)
{
  if (auto value = getenv_or_null(name)) { return std::stoll(value); }
  return fallback;
}

std::uintmax_t file_size_or_throw(std::string const& path)
{
  std::error_code ec;
  auto size = std::filesystem::file_size(path, ec);
  if (ec) { throw std::runtime_error("cannot stat file: " + path + ": " + ec.message()); }
  return size;
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

std::string parse_npy_descr(std::string const& header)
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

std::vector<std::size_t> parse_npy_shape(std::string const& header)
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

NpyHeader read_npy_header(std::ifstream& in, std::string const& path)
{
  char magic[6];
  in.read(magic, 6);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("not an npy file: " + path);
  }

  unsigned char version[2];
  in.read(reinterpret_cast<char*>(version), 2);
  std::uint32_t header_len = 0;
  if (version[0] == 1) {
    header_len = read_u16_le(in);
  } else if (version[0] == 2 || version[0] == 3) {
    header_len = read_u32_le(in);
  } else {
    throw std::runtime_error("unsupported npy version: " + path);
  }

  std::string header(header_len, '\0');
  in.read(header.data(), header_len);
  if (!in) { throw std::runtime_error("truncated npy header: " + path); }

  NpyHeader out;
  out.descr         = parse_npy_descr(header);
  out.fortran_order = header.find("fortran_order") != std::string::npos &&
                      header.find("True", header.find("fortran_order")) != std::string::npos;
  out.shape = parse_npy_shape(header);
  return out;
}

template <typename T>
std::vector<T> read_npy_payload(std::ifstream& in, std::string const& path, NpyHeader const& header)
{
  std::size_t count = 1;
  for (auto dim : header.shape) {
    count *= dim;
  }
  std::vector<T> data(count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) { throw std::runtime_error("truncated npy payload: " + path); }
  return data;
}

FloatMatrix read_fvecs(std::string const& path, std::int64_t max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open fvecs file: " + path); }

  int dim = 0;
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!in || dim <= 0) { throw std::runtime_error("invalid fvecs header: " + path); }

  const auto record_bytes = static_cast<std::uintmax_t>(sizeof(int) + sizeof(float) * dim);
  const auto bytes        = file_size_or_throw(path);
  if (bytes % record_bytes != 0) {
    throw std::runtime_error("fvecs file size is not divisible by record size: " + path);
  }

  auto rows = static_cast<std::int64_t>(bytes / record_bytes);
  if (max_rows > 0) { rows = std::min(rows, max_rows); }

  FloatMatrix out;
  out.rows = rows;
  out.dim  = dim;
  out.data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim));

  in.clear();
  in.seekg(0, std::ios::beg);
  for (std::int64_t row = 0; row < rows; ++row) {
    int row_dim = 0;
    in.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
    if (!in || row_dim != dim) { throw std::runtime_error("inconsistent fvecs row dim: " + path); }
    in.read(reinterpret_cast<char*>(out.data.data() + row * dim), sizeof(float) * dim);
    if (!in) { throw std::runtime_error("truncated fvecs file: " + path); }
  }
  return out;
}

FloatMatrix read_npy_float_matrix(std::string const& path, std::int64_t max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open npy file: " + path); }
  const auto header = read_npy_header(in, path);
  if (header.fortran_order || header.descr != "<f4" || header.shape.size() != 2) {
    throw std::runtime_error("expected C-order float32 [rows, dim] npy file: " + path);
  }

  auto rows      = static_cast<std::int64_t>(header.shape[0]);
  const auto dim = static_cast<int>(header.shape[1]);
  auto data      = read_npy_payload<float>(in, path, header);
  if (max_rows > 0 && rows > max_rows) {
    rows = max_rows;
    data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim));
  }

  FloatMatrix out;
  out.rows = rows;
  out.dim  = dim;
  out.data = std::move(data);
  return out;
}

FloatMatrix read_float_matrix(std::string const& path, std::int64_t max_rows = 0)
{
  if (std::filesystem::path(path).extension() == ".npy") {
    return read_npy_float_matrix(path, max_rows);
  }
  return read_fvecs(path, max_rows);
}

IdMatrix read_ivecs_topk(std::string const& path, int topk, std::int64_t max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open ivecs file: " + path); }

  int width = 0;
  in.read(reinterpret_cast<char*>(&width), sizeof(width));
  if (!in || width <= 0) { throw std::runtime_error("invalid ivecs header: " + path); }
  if (topk <= 0 || topk > width) { throw std::runtime_error("topk exceeds ivecs width: " + path); }

  const auto record_bytes = static_cast<std::uintmax_t>(sizeof(int) + sizeof(std::int32_t) * width);
  const auto bytes        = file_size_or_throw(path);
  if (bytes % record_bytes != 0) {
    throw std::runtime_error("ivecs file size is not divisible by record size: " + path);
  }

  auto rows = static_cast<std::int64_t>(bytes / record_bytes);
  if (max_rows > 0) { rows = std::min(rows, max_rows); }

  IdMatrix out;
  out.rows = rows;
  out.dim  = topk;
  out.data.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(topk));
  std::vector<std::int32_t> row_values(static_cast<std::size_t>(width));

  in.clear();
  in.seekg(0, std::ios::beg);
  for (std::int64_t row = 0; row < rows; ++row) {
    int row_width = 0;
    in.read(reinterpret_cast<char*>(&row_width), sizeof(row_width));
    if (!in || row_width != width) {
      throw std::runtime_error("inconsistent ivecs row width: " + path);
    }
    in.read(reinterpret_cast<char*>(row_values.data()), sizeof(std::int32_t) * width);
    if (!in) { throw std::runtime_error("truncated ivecs file: " + path); }
    for (int k = 0; k < topk; ++k) {
      if (row_values[static_cast<std::size_t>(k)] < 0) {
        throw std::runtime_error("negative unfiltered GT id in ivecs file: " + path);
      }
      out.data[static_cast<std::size_t>(row) * topk + k] =
        static_cast<std::uint32_t>(row_values[static_cast<std::size_t>(k)]);
    }
  }
  return out;
}

IdMatrix read_npy_ids_topk(std::string const& path, int topk, std::int64_t max_rows = 0)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot open npy file: " + path); }
  const auto header = read_npy_header(in, path);
  if (header.fortran_order || header.shape.size() != 2) {
    throw std::runtime_error("expected C-order [rows, width] npy id file: " + path);
  }
  if (header.descr != "<i8" && header.descr != "<i4" && header.descr != "<u4") {
    throw std::runtime_error("expected int64/int32/uint32 npy id file: " + path);
  }

  auto rows        = static_cast<std::int64_t>(header.shape[0]);
  const int width  = static_cast<int>(header.shape[1]);
  const auto limit = max_rows > 0 ? std::min(rows, max_rows) : rows;
  if (topk <= 0 || topk > width) { throw std::runtime_error("topk exceeds npy width: " + path); }

  IdMatrix out;
  out.rows = limit;
  out.dim  = topk;
  out.data.resize(static_cast<std::size_t>(limit) * static_cast<std::size_t>(topk));

  if (header.descr == "<i8") {
    auto ids = read_npy_payload<std::int64_t>(in, path, header);
    for (std::int64_t row = 0; row < limit; ++row) {
      for (int k = 0; k < topk; ++k) {
        const auto id = ids[static_cast<std::size_t>(row) * width + k];
        if (id < 0 || id > std::numeric_limits<std::uint32_t>::max()) {
          throw std::runtime_error("npy id is outside uint32 range: " + path);
        }
        out.data[static_cast<std::size_t>(row) * topk + k] = static_cast<std::uint32_t>(id);
      }
    }
  } else if (header.descr == "<i4") {
    auto ids = read_npy_payload<std::int32_t>(in, path, header);
    for (std::int64_t row = 0; row < limit; ++row) {
      for (int k = 0; k < topk; ++k) {
        const auto id = ids[static_cast<std::size_t>(row) * width + k];
        if (id < 0) { throw std::runtime_error("negative npy id: " + path); }
        out.data[static_cast<std::size_t>(row) * topk + k] = static_cast<std::uint32_t>(id);
      }
    }
  } else {
    auto ids = read_npy_payload<std::uint32_t>(in, path, header);
    for (std::int64_t row = 0; row < limit; ++row) {
      for (int k = 0; k < topk; ++k) {
        out.data[static_cast<std::size_t>(row) * topk + k] =
          ids[static_cast<std::size_t>(row) * width + k];
      }
    }
  }
  return out;
}

IdMatrix read_ids_topk(std::string const& path, int topk, std::int64_t max_rows = 0)
{
  if (std::filesystem::path(path).extension() == ".npy") {
    return read_npy_ids_topk(path, topk, max_rows);
  }
  return read_ivecs_topk(path, topk, max_rows);
}

template <typename Func>
double time_seconds(raft::resources const& res, Func&& func)
{
  raft::resource::sync_stream(res);
  const auto start = std::chrono::steady_clock::now();
  func();
  raft::resource::sync_stream(res);
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

RecallStats calc_recall(std::vector<std::uint32_t> const& expected,
                        std::vector<std::uint32_t> const& actual,
                        std::int64_t rows,
                        int topk)
{
  RecallStats stats;
  stats.total = rows * topk;
  for (std::int64_t row = 0; row < rows; ++row) {
    for (int k = 0; k < topk; ++k) {
      const auto actual_id = actual[static_cast<std::size_t>(row) * topk + k];
      bool matched         = false;
      for (int gt_k = 0; gt_k < topk; ++gt_k) {
        const auto expected_id = expected[static_cast<std::size_t>(row) * topk + gt_k];
        if (actual_id == expected_id) {
          matched = true;
          break;
        }
      }
      stats.hits += matched ? 1 : 0;
    }
  }
  stats.mismatches = stats.total - stats.hits;
  stats.recall =
    stats.total == 0 ? 0.0 : static_cast<double>(stats.hits) / static_cast<double>(stats.total);
  return stats;
}

template <typename IndexT>
SearchStats run_search(raft::resources const& res,
                       cuvs::neighbors::cagra::search_params const& params,
                       IndexT const& index,
                       raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                       std::vector<std::uint32_t> const& expected,
                       int topk,
                       int warmup,
                       int repeat)
{
  auto neighbors = raft::make_device_matrix<std::uint32_t, int64_t>(res, queries.extent(0), topk);
  auto distances = raft::make_device_matrix<float, int64_t>(res, queries.extent(0), topk);

  for (int i = 0; i < warmup; ++i) {
    cuvs::neighbors::cagra::search(res, params, index, queries, neighbors.view(), distances.view());
    raft::resource::sync_stream(res);
  }

  double best_seconds = std::numeric_limits<double>::max();
  for (int i = 0; i < repeat; ++i) {
    const auto seconds = time_seconds(res, [&] {
      cuvs::neighbors::cagra::search(
        res, params, index, queries, neighbors.view(), distances.view());
    });
    best_seconds       = std::min(best_seconds, seconds);
  }

  std::vector<std::uint32_t> actual(static_cast<std::size_t>(queries.extent(0)) *
                                    static_cast<std::size_t>(topk));
  RAFT_CUDA_TRY(cudaMemcpyAsync(actual.data(),
                                neighbors.data_handle(),
                                sizeof(std::uint32_t) * actual.size(),
                                cudaMemcpyDeviceToHost,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);

  SearchStats stats;
  stats.seconds = best_seconds;
  stats.qps     = static_cast<double>(queries.extent(0)) / best_seconds;
  stats.recall  = calc_recall(expected, actual, queries.extent(0), topk);
  return stats;
}

cuvs::neighbors::cagra::index_params make_index_params(int graph_degree,
                                                       int intermediate_graph_degree,
                                                       int nn_descent_iterations,
                                                       bool attach_dataset_on_build)
{
  cuvs::neighbors::cagra::index_params index_params;
  index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
  index_params.graph_degree              = static_cast<std::size_t>(graph_degree);
  index_params.intermediate_graph_degree = static_cast<std::size_t>(intermediate_graph_degree);
  index_params.attach_dataset_on_build   = attach_dataset_on_build;

  cuvs::neighbors::nn_descent::index_params nn_params(
    static_cast<std::size_t>(intermediate_graph_degree), cuvs::distance::DistanceType::L2Expanded);
  nn_params.max_iterations        = static_cast<std::size_t>(nn_descent_iterations);
  nn_params.return_distances      = false;
  index_params.graph_build_params = nn_params;
  return index_params;
}

TEST(RangeCagraFullRangeCompare, ExternalFvecsFullRangeMatchesNativeCagra)
{
  const auto base_path  = getenv_or("RANGE_CAGRA_BASE");
  const auto query_path = getenv_or("RANGE_CAGRA_QUERY");
  const auto gt_path    = getenv_or("RANGE_CAGRA_GT");
  if (base_path.empty() || query_path.empty() || gt_path.empty()) {
    GTEST_SKIP() << "set RANGE_CAGRA_BASE, RANGE_CAGRA_QUERY, and RANGE_CAGRA_GT to run";
  }

  const auto max_queries              = getenv_i64_or("RANGE_CAGRA_MAX_QUERIES", 0);
  const int topk                      = getenv_int_or("RANGE_CAGRA_TOPK", 10);
  const int graph_degree              = getenv_int_or("RANGE_CAGRA_GRAPH_DEGREE", 64);
  const int intermediate_graph_degree = getenv_int_or("RANGE_CAGRA_INTERMEDIATE_GRAPH_DEGREE", 128);
  const int nn_descent_iterations     = getenv_int_or("RANGE_CAGRA_NN_DESCENT_ITERS", 50);
  const int itopk_size                = getenv_int_or("RANGE_CAGRA_ITOPK_SIZE", 64);
  const int search_width              = getenv_int_or("RANGE_CAGRA_SEARCH_WIDTH", 1);
  const int warmup                    = getenv_int_or("RANGE_CAGRA_WARMUP", 1);
  const int repeat                    = getenv_int_or("RANGE_CAGRA_REPEAT", 3);

  auto base    = read_float_matrix(base_path);
  auto queries = read_float_matrix(query_path, max_queries);
  auto gt      = read_ids_topk(gt_path, topk, max_queries);

  ASSERT_EQ(base.dim, queries.dim);
  const auto nq = std::min(queries.rows, gt.rows);
  ASSERT_GT(base.rows, 1);
  ASSERT_GT(nq, 0);
  queries.rows = nq;
  queries.data.resize(static_cast<std::size_t>(nq) * static_cast<std::size_t>(queries.dim));
  gt.rows = nq;
  gt.data.resize(static_cast<std::size_t>(nq) * static_cast<std::size_t>(topk));

  raft::resources res;
  auto d_base    = raft::make_device_matrix<float, int64_t>(res, base.rows, base.dim);
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, queries.rows, queries.dim);
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_base.data_handle(),
                                base.data.data(),
                                sizeof(float) * base.data.size(),
                                cudaMemcpyHostToDevice,
                                raft::resource::get_cuda_stream(res)));
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_queries.data_handle(),
                                queries.data.data(),
                                sizeof(float) * queries.data.size(),
                                cudaMemcpyHostToDevice,
                                raft::resource::get_cuda_stream(res)));
  raft::resource::sync_stream(res);

  auto base_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    d_base.data_handle(), base.rows, base.dim);
  auto query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    d_queries.data_handle(), queries.rows, queries.dim);

  auto original_params =
    make_index_params(graph_degree, intermediate_graph_degree, nn_descent_iterations, true);
  cuvs::neighbors::cagra::index<float, std::uint32_t> original_index(res);
  const auto original_build_seconds = time_seconds(
    res, [&] { original_index = cuvs::neighbors::cagra::build(res, original_params, base_view); });

  RangeGraphBuildParams range_params;
  range_params.graph_degree              = graph_degree;
  range_params.intermediate_graph_degree = intermediate_graph_degree;
  range_params.nn_descent_iterations     = nn_descent_iterations;

  DeviceRangeGraph range_graph;
  GlobalDatasetView dataset{d_base.data_handle(), base.rows, base.dim, base.dim};
  const auto range_build_seconds = time_seconds(res, [&] {
    range_graph = build_one_range_graph(res, dataset, 0, base.rows - 1, range_params);
  });

  auto range_graph_view =
    raft::make_device_matrix_view<const std::uint32_t, int64_t, raft::row_major>(
      range_graph.edges, range_graph.rows, range_graph.degree);
  cuvs::neighbors::cagra::index<float, std::uint32_t> range_index(
    res, cuvs::distance::DistanceType::L2Expanded, base_view, range_graph_view);

  cuvs::neighbors::cagra::search_params search_params;
  search_params.itopk_size   = static_cast<std::size_t>(itopk_size);
  search_params.search_width = static_cast<std::size_t>(search_width);

  const auto original_search =
    run_search(res, search_params, original_index, query_view, gt.data, topk, warmup, repeat);
  const auto range_search =
    run_search(res, search_params, range_index, query_view, gt.data, topk, warmup, repeat);

  EXPECT_EQ(range_graph.rank_l, 0);
  EXPECT_EQ(range_graph.rows, base.rows);
  EXPECT_EQ(range_graph.degree, graph_degree);

  const auto graph_bytes =
    static_cast<std::int64_t>(range_graph.rows) * range_graph.degree * sizeof(std::uint32_t);
  const auto dataset_bytes = static_cast<std::int64_t>(base.rows) * base.dim * sizeof(float);

  std::cout << std::fixed << std::setprecision(6) << "range_cagra_full_range_compare,"
            << "base=" << base_path << ","
            << "query=" << query_path << ","
            << "gt=" << gt_path << ","
            << "rows=" << base.rows << ","
            << "dim=" << base.dim << ","
            << "nq=" << nq << ","
            << "topk=" << topk << ","
            << "graph_degree=" << graph_degree << ","
            << "intermediate_graph_degree=" << intermediate_graph_degree << ","
            << "nn_descent_iterations=" << nn_descent_iterations << ","
            << "itopk_size=" << itopk_size << ","
            << "search_width=" << search_width << ","
            << "original_build_seconds=" << original_build_seconds << ","
            << "range_build_seconds=" << range_build_seconds << ","
            << "original_search_seconds=" << original_search.seconds << ","
            << "range_search_seconds=" << range_search.seconds << ","
            << "original_qps=" << original_search.qps << ","
            << "range_qps=" << range_search.qps << ","
            << "original_recall_at_k=" << original_search.recall.recall << ","
            << "range_recall_at_k=" << range_search.recall.recall << ","
            << "original_hits=" << original_search.recall.hits << ","
            << "range_hits=" << range_search.recall.hits << ","
            << "recall_total=" << range_search.recall.total << ","
            << "graph_bytes=" << graph_bytes << ","
            << "dataset_bytes=" << dataset_bytes << std::endl;
}

}  // namespace
}  // namespace cuvs::neighbors::range_cagra::detail
