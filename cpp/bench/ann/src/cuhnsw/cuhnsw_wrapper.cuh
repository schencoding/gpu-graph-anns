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
#include "cuhnsw_statistics.hpp"
#include "cuhnsw.hpp"
#include <memory>
#include <random>

namespace cuvs::bench {

class cuhnsw_impl;

class cuhnsw : public algo<float>, public algo_gpu {
 public:
  struct build_param {
    int max_m;            // max number of neighbors
    int max_m0;           // max number of neighbors in the bottom layer
    int ef_construction;  // construction parameter
    float heuristic_coef;
    bool save_remains;

    int block_dim;
    int hyper_threads;
    int visited_table_size;
    int visited_list_size;
    bool reverse_cand;
    int log_level;
  };

  using search_param_base = typename algo<float>::search_param;
  struct search_param : public search_param_base {
    int ef_search;
    int block_dim_search;
    int hyper_threads_search;
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
  };

  cuhnsw(Metric metric, int dim, const build_param& param);

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
  std::unique_ptr<algo<float>> copy() override { return std::make_unique<cuhnsw>(*this); };

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
      auto& statistics = ::cuhnsw::Statistics::GetInstance();

      // unsigned long long counter_threadIdx_x0 = 0;
      // unsigned long long clk_distance_computation = 0;
      // unsigned long long distance_computation_counter = 0;
      // unsigned long long distance_computation_counter_upper_layers = 0;
      // unsigned long long num_queries = 0;
      //
      // // _visited_table
      // unsigned long long clk_check_visited_table = 0;
      // unsigned long long counter_check_visited_table = 0;
      // unsigned long long clk_set_visited_table = 0;
      // unsigned long long counter_set_visited_table = 0;
      //
      // // ef_search_pq
      // unsigned long long clk_get_candidates = 0;
      // unsigned long long counter_get_candidates = 0;
      // unsigned long long clk_pq_pop = 0;
      // unsigned long long counter_pq_pop = 0;
      // unsigned long long clk_set_candidates_checked = 0;
      // unsigned long long counter_set_candidates_checked = 0;
      // unsigned long long clk_check_queue = 0;
      // unsigned long long counter_check_queue = 0;
      // unsigned long long clk_pq_push = 0;
      // unsigned long long counter_pq_push = 0;
      //
      // unsigned long long clk_set_neighbors_found = 0;
      // unsigned long long counter_set_neighbors_found = 0;
      //
      counters["metrics_counter_threadIdx_x0"] = statistics.counter_threadIdx_x0;
      counters["metrics_clk_distance_computation"] = statistics.clk_distance_computation;
      counters["metrics_distance_computation_counter"] = statistics.distance_computation_counter;
      counters["metrics_distance_computation_counter_upper_layers"] = statistics.distance_computation_counter_upper_layers;
      counters["metrics_num_queries"] = statistics.num_queries;
      counters["metrics_clk_check_visited_table"] = statistics.clk_check_visited_table;
      counters["metrics_counter_check_visited_table"] = statistics.counter_check_visited_table;
      counters["metrics_clk_set_visited_table"] = statistics.clk_set_visited_table;
      counters["metrics_counter_set_visited_table"] = statistics.counter_set_visited_table;
      counters["metrics_clk_get_candidates"] = statistics.clk_get_candidates;
      counters["metrics_counter_get_candidates"] = statistics.counter_get_candidates;
      counters["metrics_clk_pq_pop"] = statistics.clk_pq_pop;
      counters["metrics_counter_pq_pop"] = statistics.counter_pq_pop;
      counters["metrics_clk_set_candidates_checked"] = statistics.clk_set_candidates_checked;
      counters["metrics_counter_set_candidates_checked"] = statistics.counter_set_candidates_checked;
      counters["metrics_clk_check_queue"] = statistics.clk_check_queue;
      counters["metrics_counter_check_queue"] = statistics.counter_check_queue;
      counters["metrics_clk_pq_push"] = statistics.clk_pq_push;
      counters["metrics_counter_pq_push"] = statistics.counter_pq_push;
      counters["metrics_clk_set_neighbors_found"] = statistics.clk_set_neighbors_found;
      counters["metrics_counter_set_neighbors_found"] = statistics.counter_set_neighbors_found;
    }
    return counters;
  }

  void print_metrics() const override
  {
    if constexpr (cuvs::bench::collect_metrics) {
      auto& statistics = ::cuhnsw::Statistics::GetInstance();
      statistics.print();
    }
  }

  void reset_metrics() override
  {
    if constexpr (cuvs::bench::collect_metrics) {
      auto& statistics = ::cuhnsw::Statistics::GetInstance();
      statistics.reset();
    }
  }
 private:
  std::shared_ptr<algo<float>> impl_;
};

class cuhnsw_impl : public algo<float>, public algo_gpu {
 public:
  using search_param_base = typename algo<float>::search_param;

  cuhnsw_impl(Metric metric, int dim, const typename cuhnsw::build_param& param)
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
    auto r = std::make_unique<cuhnsw_impl>(*this);
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

  void set_random_levels();

 private:
  enum DIST_TYPE dist_type()
  {
    switch (metric_) {
      case Metric::kEuclidean: return DIST_TYPE::L2;
      case Metric::kInnerProduct: return DIST_TYPE::DOT;
      default: throw std::runtime_error("unsupported metric");
    }
  }
  using algo<float>::metric_;
  using algo<float>::dim_;

  using cuhnswgpu_instance = ::cuhnsw::CuHNSW;
  std::shared_ptr<cuhnswgpu_instance> cuhnsw_;
  typename cuhnsw::build_param build_param_;
  typename cuhnsw::search_param search_param_;
  cudaStream_t stream_;
  const float* base_dataset_ = nullptr;
  size_t base_n_rows_        = 0;
};

cuhnsw::cuhnsw(Metric metric, int dim, const build_param& param) : algo<float>(metric, dim)
{
  impl_ = std::make_shared<cuhnsw_impl>(metric, dim, param);
}

void cuhnsw_impl::set_random_levels()
{
  // cuhnsw_->SetRandomLevecuhnsw/pyhnsw.py:65
  // def set_random_levels(self):
  //   np.random.seed(self.opt.seed)
  //   num_data = self.data.shape[0]
  //   levels = np.random.uniform(size=num_data)
  //   levels = np.maximum(levels, EPS)
  //   levels = (-np.log(levels) * self.opt.level_mult).astype(np.int32)
  //   self.obj.set_random_levels(levels)ls();
  std::random_device rd;
  std::mt19937 gen(rd());
  int* levels     = new int[base_n_rows_];
  const float EPS = 1e-10;
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  const float level_mult = 1.0f / std::log(build_param_.max_m);
  for (size_t i = 0; i < base_n_rows_; ++i) {
    levels[i] = static_cast<int>(-std::log(std::max(dis(gen), EPS)) * level_mult);
  }
  cuhnsw_->SetRandomLevels(levels);
  delete[] levels;
}

void cuhnsw_impl::build(const float* dataset, size_t nrow)
{
  base_dataset_ = dataset;
  base_n_rows_  = nrow;
  cuhnsw_       = std::make_shared<cuhnswgpu_instance>();
  cuhnsw_->Init(build_param_.max_m,
                build_param_.max_m0,
                build_param_.save_remains,
                build_param_.ef_construction,
                build_param_.block_dim,
                build_param_.hyper_threads,
                build_param_.visited_table_size,
                build_param_.visited_list_size,
                build_param_.heuristic_coef,
                dist_type(),
                build_param_.reverse_cand,
                build_param_.log_level);
  cuhnsw_->SetData(base_dataset_, static_cast<int>(base_n_rows_), dim_);
  set_random_levels();
  cuhnsw_->BuildGraph();
}

void cuhnsw_impl::set_search_dataset(const float* dataset, size_t nrow)
{
  if (base_dataset_ != dataset || base_n_rows_ != nrow) {
    base_dataset_ = dataset;
    base_n_rows_  = nrow;
    cuhnsw_->SetData(base_dataset_, static_cast<int>(base_n_rows_), dim_);
    cuhnsw_->SetDevDataIfNullptr();
  }
}

void cuhnsw_impl::set_search_param(const search_param_base& param)
{
  search_param_ = dynamic_cast<const typename cuhnsw::search_param&>(param);
  cuhnsw_->SetBlockDim(search_param_.block_dim_search, search_param_.hyper_threads_search);
}

void cuhnsw_impl::search(const float* queries,
                         int batch_size,
                         int k,
                         algo_base::index_type* neighbors,
                         float* distances) const
{
  int* nns       = new int[batch_size * k];
  int* found_cnt = new int[batch_size];
  cuhnsw_->SearchGraph(queries, batch_size, k, search_param_.ef_search, nns, distances, found_cnt);
  // std::cout << "batch: " << batch_size << " k: " << k << std::endl;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < k; ++j) {
      neighbors[i * k + j] = nns[i * k + j];
    }
  }
  delete[] nns;
  delete[] found_cnt;
}

void cuhnsw_impl::save(const std::string& file) const { cuhnsw_->SaveIndex(file); }

void cuhnsw_impl::load(const std::string& file)
{
  cuhnsw_ = std::make_shared<cuhnswgpu_instance>();
  cuhnsw_->Init(build_param_.max_m,
                build_param_.max_m0,
                build_param_.save_remains,
                build_param_.ef_construction,
                build_param_.block_dim,
                build_param_.hyper_threads,
                build_param_.visited_table_size,
                build_param_.visited_list_size,
                build_param_.heuristic_coef,
                dist_type(),
                build_param_.reverse_cand,
                build_param_.log_level);
  cuhnsw_->SetData(base_dataset_, static_cast<int>(base_n_rows_), dim_);
  cuhnsw_->LoadIndex(file);
}

}  // namespace cuvs::bench
