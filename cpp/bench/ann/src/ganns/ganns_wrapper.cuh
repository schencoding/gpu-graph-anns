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
#include "../common/util.hpp"

#include "ganns/data.h"
#include "ganns/graph_index/metrics.h"
#include "ganns/ganns-inl.cuh"
#include "ganns/metric_type.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuvs::bench {

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
class ganns_impl;

class ganns : public algo<float>, public algo_gpu {
 public:
  struct build_param {
    bool hierarchical;             // whether to use hierarchical graph
    int num_of_candidates;         // the number of explored vertices;
    int num_of_initial_neighbors;  // minimum degree in the proximity graph (by default, d_max = 2 *
                                   // d_min)
  };

  using search_param_base = typename algo<float>::search_param;
  struct search_param : public search_param_base {
    int num_of_candidates;  // the number of explored vertices;
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
  };

  ganns(Metric metric, int dim, const build_param& param);

  void build(const float* dataset, size_t nrow) override { impl_->build(dataset, nrow); }

  void set_search_param(const search_param_base& param) override { impl_->set_search_param(param); }
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
  std::unique_ptr<algo<float>> copy() override { return std::make_unique<ganns>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return impl_->get_preference();
  }

  void set_search_dataset(const float* dataset, size_t nrow) override
  {
    impl_->set_search_dataset(dataset, nrow);
  };

  benchmark::UserCounters get_custom_counters() const override
  {
    benchmark::UserCounters counters;
    if constexpr (cuvs::bench::collect_metrics) {
      auto& metrics = ::ganns::Metrics::get_instance();

      counters["metrics_counter_thread_idx_x0"]           = metrics.counter_thread_idx_x0;
      counters["metrics_counter_iterations"]              = metrics.counter_iterations;
      counters["metrics_stage_init"]                      = metrics.stage_init;
      counters["metrics_stage_init_distance_computation"] = metrics.stage_init_distance_computation;
      counters["metrics_stage_1"]                         = metrics.stage_1;
      counters["metrics_stage_2"]                         = metrics.stage_2;
      counters["metrics_stage_3_distance_computation"]    = metrics.stage_3_distance_computation;
      counters["metrics_stage_4"]                         = metrics.stage_4;
      counters["metrics_stage_5"]                         = metrics.stage_5;
      counters["metrics_stage_6"]                         = metrics.stage_6;
      counters["metrics_counter_stage1"]                  = metrics.counter_stage1;
      counters["metrics_counter_stage2"]                  = metrics.counter_stage2;
      counters["metrics_counter_stage4"]                  = metrics.counter_stage4;
      counters["metrics_counter_stage5"]                  = metrics.counter_stage5;
      counters["metrics_counter_stage6"]                  = metrics.counter_stage6;
      counters["metrics_stage_final"]                     = metrics.stage_final;
      counters["metrics_distance_computation_counter"]    = metrics.distance_computation_counter;
      counters["metrics_queries"]                         = metrics.metric_queries;
    }
    return counters;
  }

  void print_metrics() const override
  {
    if constexpr (cuvs::bench::collect_metrics) {
      auto& metrics = ::ganns::Metrics::get_instance();
      metrics.print_metrics();
    }
  }

  void reset_metrics() override
  {
    if constexpr (cuvs::bench::collect_metrics) {
      auto& metrics = ::ganns::Metrics::get_instance();
      metrics.reset();
    }
  }

 private:
  template <Metric metric, int dim>
  void build_impl(const build_param& param);
  std::shared_ptr<algo<float>> impl_;
};

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
class ganns_impl : public algo<float>, public algo_gpu {
 public:
  using search_param_base = typename algo<float>::search_param;

  ganns_impl(Metric metric, int dim, const typename ganns::build_param& param)
    : algo<float>(metric, dim),
      build_param_(param),
      stream_(cuvs::bench::get_stream_from_global_pool())
  {
  }

  void build(const float* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param) override;
  void search(const float* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override { return stream_; }

  void save(const std::string& file) const override;
  void load(const std::string& file) override;
  std::unique_ptr<algo<float>> copy() override
  {
    auto r = std::make_unique<ganns_impl<D, metric_type, HIERARCHICAL>>(*this);
    // set the thread-local stream to the copied handle.
    r->stream_ = cuvs::bench::get_stream_from_global_pool();
    return r;
  };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHost;
    property.query_memory_type   = MemoryType::kHost;
    return property;
  }

  void set_search_dataset(const float* dataset, size_t nrow) override;

 private:
  using algo<float>::metric_;
  using algo<float>::dim_;

  using gannsgpu_instance = ::ganns::GANNS;
  std::shared_ptr<gannsgpu_instance> ganns_;
  typename ganns::build_param build_param_;
  typename ganns::search_param search_param_;
  cudaStream_t stream_;
  const float* base_dataset_             = nullptr;
  size_t base_n_rows_                    = 0;
  std::optional<std::string> graph_file_ = std::nullopt;

  void load_impl()
  {
    if (base_dataset_ == nullptr) { return; }
    if (base_n_rows_ == 0) { return; }

    if (dim_ > 960) { throw std::runtime_error("GANNS instance only supports dim <= 960"); }

    Data* data = new Data(const_cast<float*>(base_dataset_), base_n_rows_, dim_);
    ganns_     = std::make_shared<gannsgpu_instance>();
    ganns_->AddGraph<metric_type, D, collect_metrics>(HIERARCHICAL ? "hnsw" : "nsw", data);
    if (graph_file_.has_value()) { ganns_->Load(graph_file_.value()); }
  }
};

template <Metric metric>
struct ConvertMetricTrait;

template <>
struct ConvertMetricTrait<Metric::kEuclidean> {
  static constexpr ::ganns::MetricType metric_type = ::ganns::MetricType::L2;
};

template <>
struct ConvertMetricTrait<Metric::kInnerProduct> {
  static constexpr ::ganns::MetricType metric_type = ::ganns::MetricType::IP;
};

template <Metric metric, int dim>
void ganns::build_impl(const build_param& param)
{
  if (param.hierarchical) {
    impl_ = std::make_shared<ganns_impl<dim, ConvertMetricTrait<metric>::metric_type, true>>(
      metric, dim, param);
  } else {
    impl_ = std::make_shared<ganns_impl<dim, ConvertMetricTrait<metric>::metric_type, false>>(
      metric, dim, param);
  }
}

ganns::ganns(Metric metric, int dim, const build_param& param) : algo<float>(metric, dim)
{
  // ggnn/src/deep1b_multi_gpu.cu, and adapt it deep1B
  if (metric == Metric::kEuclidean && dim == 96) {
    build_impl<Metric::kEuclidean, 96>(param);
    return;
  }
  if (metric == Metric::kInnerProduct && dim == 96) {
    build_impl<Metric::kInnerProduct, 96>(param);
    return;
  }
  if (metric == Metric::kInnerProduct && dim == 100) {
    build_impl<Metric::kInnerProduct, 100>(param);
    return;
  }
  if (metric == Metric::kEuclidean && dim == 100) {
    build_impl<Metric::kEuclidean, 100>(param);
    return;
  }
  if (metric == Metric::kEuclidean && dim == 128) {
    build_impl<Metric::kEuclidean, 128>(param);
    return;
  }
  if (metric == Metric::kInnerProduct && dim == 128) {
    build_impl<Metric::kInnerProduct, 128>(param);
    return;
  }
  if (metric == Metric::kEuclidean && dim == 200) {
    build_impl<Metric::kEuclidean, 200>(param);
    return;
  }
  if (metric == Metric::kInnerProduct && dim == 200) {
    build_impl<Metric::kInnerProduct, 200>(param);
    return;
  }
  if (metric == Metric::kEuclidean && dim == 256) {
    build_impl<Metric::kEuclidean, 256>(param);
    return;
  }
  if (metric == Metric::kEuclidean && dim == 784) {
    build_impl<Metric::kEuclidean, 784>(param);
    return;
  }
  if (metric == Metric::kEuclidean && dim == 960) {
    build_impl<Metric::kEuclidean, 960>(param);
    return;
  }

  throw std::runtime_error(
    "ganns: not supported combination of metric, dim and build param: metric=" +
    std::to_string(static_cast<int>(metric)) + ", dim=" + std::to_string(dim) + " . " +
    "see GANNS's constructor in ggnn_wrapper.cuh for available combinations");
}

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
void ganns_impl<D, metric_type, HIERARCHICAL>::build(const float* dataset, size_t nrow)
{
  base_dataset_ = dataset;
  base_n_rows_  = nrow;
  load_impl();
  ganns_->Establishment(build_param_.num_of_initial_neighbors, build_param_.num_of_candidates);
}

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
void ganns_impl<D, metric_type, HIERARCHICAL>::set_search_dataset(const float* dataset, size_t nrow)
{
  if (base_dataset_ != dataset || base_n_rows_ != nrow) {
    base_dataset_ = dataset;
    base_n_rows_  = nrow;
    load_impl();
    ganns_->PrepareDeviceData();
  }
}

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
void ganns_impl<D, metric_type, HIERARCHICAL>::set_search_param(const search_param_base& param)
{
  search_param_ = dynamic_cast<const typename ganns::search_param&>(param);
}

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
void ganns_impl<D, metric_type, HIERARCHICAL>::search(const float* queries,
                                                      int batch_size,
                                                      int k,
                                                      algo_base::index_type* neighbors,
                                                      float* distances) const
{
  static_assert(sizeof(size_t) == sizeof(int64_t),
                "sizes of size_t and GANNS's KeyT are different");

  int* results = nullptr;
  ganns_->SearchTopKonDevice(
    const_cast<float*>(queries), k, results, batch_size, search_param_.num_of_candidates);
  // std::cout << "batch: " << batch_size << " k: " << k << std::endl;
  int internal_topk = pow(2.0, ceil(log(k) / log(2)));
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < k; ++j) {
      neighbors[i * k + j] = results[i * internal_topk + j];
      // if (i < 10) { std::cout << results[i * internal_topk + j] << " "; }
    }
    // if (i < 10) { std::cout << std::endl; }
  }
  ganns_->FreeResults(results);
}

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
void ganns_impl<D, metric_type, HIERARCHICAL>::save(const std::string& file) const
{
  ganns_->Dump(file);
}

template <int D, ::ganns::MetricType metric_type, bool HIERARCHICAL>
void ganns_impl<D, metric_type, HIERARCHICAL>::load(const std::string& file)
{
  if (!graph_file_.has_value() || graph_file_.value() != file) {
    graph_file_ = file;
    load_impl();
  }
}

}  // namespace cuvs::bench
