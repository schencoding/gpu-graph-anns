#pragma once

#include "../common/ann_types.hpp"
#include "../common/util.hpp"
#include "ggnn/cuda_knn_ggnn_gpu_instance.cuh"
#include "ggnn_wrapper.cuh"

namespace cuvs::bench {
template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
class ggnn_impl : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  ggnn_impl(Metric metric, int dim, const typename ggnn<T>::build_param& param);

  void build(const T* dataset, size_t nrow) override;

  void set_search_param(const search_param_base& param) override;
  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override { return stream_; }

  void save(const std::string& file) const override;
  void load(const std::string& file) override;
  std::unique_ptr<algo<T>> copy() override
  {
    auto r = std::make_unique<ggnn_impl<T, measure, D, KBuild, KQuery, S>>(*this);
    // set the thread-local stream to the copied handle.
    r->stream_ = cuvs::bench::get_stream_from_global_pool();
    return r;
  };

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kDevice;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }

  void set_search_dataset(const T* dataset, size_t nrow) override;

  benchmark::UserCounters get_custom_counters() const override { return {}; }

  void print_metrics() const override {}

  void reset_metrics() override {}

 private:
  using algo<T>::metric_;
  using algo<T>::dim_;

  using ggnngpu_instance = GGNNGPUInstance<measure,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           T /* BaseT */,
                                           size_t /* BAddrT */,
                                           D,
                                           KBuild,
                                           KBuild / 2 /* KF */,
                                           KQuery,
                                           S>;
  std::shared_ptr<ggnngpu_instance> ggnn_;
  typename ggnn<T>::build_param build_param_;
  typename ggnn<T>::search_param search_param_;
  cudaStream_t stream_;
  const T* base_dataset_                 = nullptr;
  size_t base_n_rows_                    = 0;
  std::optional<std::string> graph_file_ = std::nullopt;

  void load_impl()
  {
    if (base_dataset_ == nullptr) { return; }
    if (base_n_rows_ == 0) { return; }
    int device;
    cudaGetDevice(&device);
    ggnn_ = std::make_shared<ggnngpu_instance>(
      device, base_n_rows_, build_param_.num_layers, true, build_param_.tau_build);
    ggnn_->set_base_data(base_dataset_);
    ggnn_->set_stream(get_sync_stream());
    if (graph_file_.has_value()) {
      auto& ggnn_host   = ggnn_->ggnn_cpu_buffers.at(0);
      auto& ggnn_device = ggnn_->ggnn_shards.at(0);
      ggnn_->set_stream(get_sync_stream());

      ggnn_host.load(graph_file_.value());
      ggnn_host.uploadAsync(ggnn_device);
      cudaStreamSynchronize(ggnn_device.stream);
    }
  }
};

}  // namespace cuvs::bench
