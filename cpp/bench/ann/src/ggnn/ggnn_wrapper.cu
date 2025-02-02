#include "ggnn/statistics.h"
#include "ggnn_impl.cuh"
#include "ggnn_wrapper.cuh"

namespace cuvs::bench {

template <typename T>
benchmark::UserCounters ggnn<T>::get_custom_counters() const
{
  benchmark::UserCounters counters;
  if constexpr (cuvs::bench::collect_metrics) {
    auto& statistics = GGNNStatistics::getInstance();

    counters["metrics_clk_init"]                 = statistics.clk_init;
    counters["metrics_clk_pop"]                  = statistics.clk_pop;
    counters["metrics_clk_distance_computation"] = statistics.clk_distance_computation;
    counters["metrics_clk_distance_computation_load_keys_start"] =
      statistics.clk_distance_computation_load_keys_start;

    counters["metrics_clk_push"]                     = statistics.clk_push;
    counters["metrics_clk_final"]                    = statistics.clk_final;
    counters["metrics_distance_computation_counter"] = statistics.distance_computation_counter;
    counters["metrics_num_queries"]                  = statistics.num_queries;
  }
  return counters;
}

template <typename T>
void ggnn<T>::print_metrics() const
{
  if constexpr (cuvs::bench::collect_metrics) { GGNNStatistics::getInstance().print(); }
}

template <typename T>
void ggnn<T>::reset_metrics()
{
  if constexpr (cuvs::bench::collect_metrics) { GGNNStatistics::getInstance().reset(); }
}

template <typename T>
template <int DIM, int K>
void ggnn<T>::create_impl_k(Metric metric, int dim, const build_param& param)
{
    if (param.k_build == 24 && param.segment_size == 32) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 24, K, 32>>(metric, dim, param);
    } else if (param.k_build == 24 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 24, K, 64>>(metric, dim, param);
    } else if (param.k_build == 48 && param.segment_size == 32) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 48, K, 32>>(metric, dim, param);
    } else if (param.k_build == 48 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 48, K, 64>>(metric, dim, param);
    } else if (param.k_build == 64 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 64, K, 64>>(metric, dim, param);
    } else if (param.k_build == 96 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 96, K, 64>>(metric, dim, param);
    }
}

template <typename T>
template <int DIM>
void ggnn<T>::create_impl(Metric metric, int dim, const build_param& param)
{
  if (metric == Metric::kEuclidean && dim == DIM && param.k == 10) {
    create_impl_k<DIM, 10>(metric, dim, param);
  } else if (metric == Metric::kEuclidean && dim == DIM && param.k == 100) {
    create_impl_k<DIM, 100>(metric, dim, param);
  }
}

template <typename T>
ggnn<T>::ggnn(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  create_impl<100>(metric, dim, param);
  if (impl_) { return; }
  create_impl<128>(metric, dim, param);
  if (impl_) { return; }
  create_impl<784>(metric, dim, param);
  if (impl_) { return; }
  create_impl<960>(metric, dim, param);
  if (impl_) { return; }
  // ggnn/src/deep1b_multi_gpu.cu, and adapt it deep1B
  // else if (metric == Metric::kEuclidean && dim == 96 && param.k_build == 24 && param.k == 10 &&
  //          param.segment_size == 32) {
  //   impl_ = std::make_shared<ggnn_impl<T, Euclidean, 96, 24, 10, 32>>(metric, dim, param);
  // } else if (metric == Metric::kInnerProduct && dim == 96 && param.k_build == 24 && param.k == 10
  // &&
  //            param.segment_size == 32) {
  //   impl_ = std::make_shared<ggnn_impl<T, Cosine, 96, 24, 10, 32>>(metric, dim, param);
  // }
  // else if (metric == Metric::kInnerProduct && dim == 96 && param.k_build == 96 && param.k == 10
  // &&
  //            param.segment_size == 64) {
  //   impl_ = std::make_shared<ggnn_impl<T, Cosine, 96, 96, 10, 64>>(metric, dim, param);
  // }
  // ggnn/src/glove200.cu, adapt it to glove100
  // else if (metric == Metric::kInnerProduct && dim == 100 && param.k_build == 96 && param.k == 10
  // &&
  //          param.segment_size == 64) {
  //   impl_ = std::make_shared<ggnn_impl<T, Cosine, 100, 96, 10, 64>>(metric, dim, param);
  // }
  {
    throw std::runtime_error(
      "ggnn: not supported combination of metric, dim and build param; "
      "see Ggnn's constructor in ggnn_wrapper.cuh for available combinations");
  }
}

template class ggnn<float>;
}  // namespace cuvs::bench
