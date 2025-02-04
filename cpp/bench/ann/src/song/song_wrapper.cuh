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
#pragma once

#include "../common/ann_types.hpp"

#include "song/config.h"
#include "song/warp_astar_accelerator.h"
#include <memory>

namespace cuvs::bench {
template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
class song_impl;

class song : public algo<data_value_t>, public algo_gpu {
 public:
  struct build_param {
    int num_threads{32};
    int degree{31};
  };

  using search_param_base = typename algo<data_value_t>::search_param;

  struct search_param : public search_param_base {
    size_t finish_cnt;
    int pq_size;
    [[nodiscard]] virtual auto needs_dataset() const -> bool override { return true; };
  };

  song(Metric metric, int dim, const build_param& param, VisitedTableType visited_table_type);
  void build(const float* dataset, size_t nrow) override { impl_->build(dataset, nrow); }

  void set_search_param(const search_param_base& param) override { impl_->set_search_param(param); }
  void set_search_dataset(const data_value_t* dataset, size_t nrow) override
  {
    impl_->set_search_dataset(dataset, nrow);
  };
  void search(const float* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    impl_->search(queries, batch_size, k, neighbors, distances);
  }
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return dynamic_cast<algo_gpu*>(impl_.get())->get_sync_stream();
  }

  void save(const std::string& file) const override { impl_->save(file); }

  void load(const std::string& file) override { impl_->load(file); }
  std::unique_ptr<algo<data_value_t>> copy() override { return std::make_unique<song>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return impl_->get_preference();
  }

  benchmark::UserCounters get_custom_counters() const override
  {
    benchmark::UserCounters counters;
    if constexpr (cuvs::bench::collect_metrics) {
      auto& measure                                    = Measure::get_instance();
      counters["metrics_stage_init"]                   = measure.stage_init;
      counters["metrics_stage1"]                       = measure.stage1;
      counters["metrics_stage_distance_computation"]   = measure.stage_distance_computation;
      counters["metrics_stage3"]                       = measure.stage3;
      counters["metrics_stage_final"]                  = measure.stage_final;
      counters["metrics_distance_computation_counter"] = measure.distance_computation_counter;
      counters["metrics_queries"]                      = measure.metric_queries;
    }
    return counters;
  }

  void print_metrics() const override
  {
    if constexpr (cuvs::bench::collect_metrics) {
      auto& measure = Measure::get_instance();
      measure.print_metrics();
    }
  }

  void reset_metrics() override
  {
    if constexpr (cuvs::bench::collect_metrics) {
      auto& measure = Measure::get_instance();
      measure.reset();
    }
  }

 private:
  template <Metric metric, VisitedTableType visited_table_type>
  void make_impl(int dim, const build_param& param);
  template <Metric metric>
  void make_impl(int dim, const build_param& param, VisitedTableType visited_table_type);
  std::shared_ptr<algo<data_value_t>> impl_;
};


template <Metric metric>
struct ConvertMetricTrait;

template <>
struct ConvertMetricTrait<Metric::kEuclidean> {
  static constexpr SongMetricType metric = SongMetricType::L2;
};

template <>
struct ConvertMetricTrait<Metric::kInnerProduct> {
  static constexpr SongMetricType metric = SongMetricType::IP;
};


}  // namespace cuvs::bench
