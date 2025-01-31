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

#include "cuhnsw_wrapper.cuh"
#include "types.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench {

template <typename T>
void parse_build_param(const nlohmann::json& conf, typename cuvs::bench::cuhnsw::build_param& param)
{
  param.max_m              = conf.at("max_m");
  param.max_m0             = conf.at("max_m0");
  param.ef_construction    = conf.at("ef_construction");
  param.heuristic_coef     = conf.at("heuristic_coef");
  param.save_remains       = conf.at("save_remains");
  param.block_dim          = conf.at("block_dim");
  param.hyper_threads      = conf.at("hyper_threads");
  param.visited_table_size = conf.at("visited_table_size");
  param.visited_list_size  = conf.at("visited_list_size");
  param.reverse_cand       = conf.at("reverse_cand");
  param.log_level          = conf.at("log_level");
}

template <typename T>
void parse_search_param(const nlohmann::json& conf,
                        typename cuvs::bench::cuhnsw::search_param& param)
{
  param.ef_search = conf.at("ef_search");
  param.block_dim_search = conf.at("block_dim_search");
}

template <typename T, class Algo>
auto make_algo(cuvs::bench::Metric metric, int dim, const nlohmann::json& conf)
  -> std::unique_ptr<cuvs::bench::algo<T>>
{
  typename Algo::build_param param{};
  parse_build_param<T>(conf, param);
  return std::make_unique<Algo>(metric, dim, param);
}

template <typename T>
auto create_algo(const std::string& algo_name,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  cuvs::bench::Metric metric = parse_metric(distance);
  std::unique_ptr<cuvs::bench::algo<T>> a;

  if constexpr (std::is_same_v<T, float>) {
    if (algo_name == "cuhnsw") { a = make_algo<T, cuvs::bench::cuhnsw>(metric, dim, conf); }
  }
  if (!a) { throw std::runtime_error("invalid algo: '" + algo_name + "'"); }

  return a;
}

template <typename T>
auto create_search_param(const std::string& algo_name, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
  if constexpr (std::is_same_v<T, float>) {
    if (algo_name == "cuhnsw") {
      auto param = std::make_unique<typename cuvs::bench::cuhnsw::search_param>();
      parse_search_param<float>(conf, *param);
      return param;
    }
  }
  throw std::runtime_error("invalid algo: '" + algo_name + "'");
}

}  // namespace cuvs::bench

// cuhnsw only support float
REGISTER_ALGO_INSTANCE(float);

#ifdef ANN_BENCH_BUILD_MAIN
#include "../common/benchmark.hpp"
int main(int argc, char** argv) { return cuvs::bench::run_main(argc, argv); }
#endif
