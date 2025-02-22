/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../common/ann_types.hpp"
#include "song_wrapper.cuh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuvs::bench {

template <typename T>
void parse_build_param(const nlohmann::json& conf, typename cuvs::bench::song::build_param& param)
{
  param.num_threads = conf.at("num_threads");
  param.degree = conf.at("degree");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf, typename cuvs::bench::song::search_param& param)
{
  param.finish_cnt = conf.at("finish_cnt");
  param.pq_size = conf.at("pq_size");
}

template <typename T, class Algo>
auto make_algo(cuvs::bench::Metric metric, int dim, const nlohmann::json& conf, VisitedTableType visited_table_type)
  -> std::unique_ptr<cuvs::bench::algo<T>>
{
  typename Algo::build_param param;
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo>(metric, dim, param, visited_table_type);
}

VisitedTableType algo_name_to_visited_table_type(const std::string& algo_name)
{
  if (algo_name == "song") { return VisitedTableType::kBloomFilter; }
  if (algo_name == "song_cuckoofilter") { return VisitedTableType::kCuckooFilter; }
  if (algo_name == "song_hashtable") { return VisitedTableType::kHashTable; }
  if (algo_name == "song_hashtable_sel") { return VisitedTableType::kHashTableSel; }
  if (algo_name == "song_hashtable_sel_del") { return VisitedTableType::kHashTableSelDel; }

  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  auto visited_table_type = algo_name_to_visited_table_type(algo_name);
  if constexpr (std::is_same_v<T, float>) {
    a = make_algo<T, cuvs::bench::song>(metric, dim, conf, visited_table_type);
  }
  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if constexpr (std::is_same_v<T, float>) {
    auto param = std::make_unique<typename cuvs::bench::song::search_param>();
    parse_search_param<float>(conf, *param);
    return param;
  }
  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

// ganns only support float
REGISTER_ALGO_INSTANCE(float);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv)
{
  // You may need to increase this parameter for some new GPUs
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 30ULL * 1024 * 1024 * 1024);
  return cuvs::bench::run_main(argc, argv);
}
#endif
