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
#include "ggnn/utils/cuda_knn_utils.cuh"

// #include <raft/util/cudart_utils.hpp>

#include <memory>

namespace cuvs::bench {

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
class ggnn_impl;

template <typename T>
class ggnn : public algo<T>, public algo_gpu {
 public:
  struct build_param {
    int k_build{24};       // KBuild
    int segment_size{32};  // S
    int num_layers{4};     // L
    float tau_build{0.5};
    int refine_iterations{2};
    int k;  // GGNN requires to know k during building
  };

  using search_param_base = typename algo<T>::search_param;
  struct search_param : public search_param_base {
    float tau_query;
    int block_dim{32};
    int max_iterations{400};
    int cache_size{512};
    int sorted_size{256};
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
  };

  ggnn(Metric metric, int dim, const build_param& param);

  void build(const T* dataset, size_t nrow) override { impl_->build(dataset, nrow); }

  void set_search_param(const search_param_base& param) override { impl_->set_search_param(param); }
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    impl_->search(queries, batch_size, k, neighbors, distances);
  }

  benchmark::UserCounters get_custom_counters() const override;
  void print_metrics() const override;
  void reset_metrics() override;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return dynamic_cast<algo_gpu*>(impl_.get())->get_sync_stream();
  }

  void save(const std::string& file) const override { impl_->save(file); }
  void load(const std::string& file) override { impl_->load(file); }
  std::unique_ptr<algo<T>> copy() override { return std::make_unique<ggnn<T>>(*this); };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    return impl_->get_preference();
  }

  void set_search_dataset(const T* dataset, size_t nrow) override
  {
    impl_->set_search_dataset(dataset, nrow);
  };

 private:
  template<int DIM, int K>
  void create_impl_k(Metric metric, int dim, const build_param& param);
  template<int DIM>
  void create_impl(Metric metric, int dim, const build_param& param);

  std::shared_ptr<algo<T>> impl_;
};

extern template class ggnn<float>;
}  // namespace cuvs::bench
