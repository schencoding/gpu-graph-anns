#include "ggnn_impl.cuh"
#include "ggnn_wrapper.cuh"

namespace cuvs::bench {

  template <typename T>
template <int DIM>
void ggnn<T>::create_impl(Metric metric, int dim, const build_param& param)
{
  if (metric == Metric::kEuclidean && dim == DIM && param.k == 10) {
    if (param.k_build == 24 && param.segment_size == 32) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 24, 10, 32>>(metric, dim, param);
    } else if (param.k_build == 24 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 24, 10, 64>>(metric, dim, param);
    } else if (param.k_build == 48 && param.segment_size == 32) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 48, 10, 32>>(metric, dim, param);
    } else if (param.k_build == 48 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 48, 10, 64>>(metric, dim, param);
    } else if (param.k_build == 64 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 64, 10, 64>>(metric, dim, param);
    } else if (param.k_build == 96 && param.segment_size == 64) {
      impl_ = std::make_shared<ggnn_impl<T, Euclidean, DIM, 96, 10, 64>>(metric, dim, param);
    }
  }
}

template <typename T>
ggnn<T>::ggnn(Metric metric, int dim, const build_param& param) : algo<T>(metric, dim)
{
  create_impl<100>(metric, dim, param);
  if (impl_) { return; }
  create_impl<128>(metric, dim, param);
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
