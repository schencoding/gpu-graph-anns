#pragma once

#include "ggnn_impl.cuh"

namespace cuvs::bench {
#define INSTANTIATE_EXTERN(DIM, KQUERY)                                          \
  extern template class ggnn_impl<float, Euclidean, DIM, 24, KQUERY, 32>; \
  extern template class ggnn_impl<float, Euclidean, DIM, 24, KQUERY, 64>; \
  extern template class ggnn_impl<float, Euclidean, DIM, 48, KQUERY, 32>; \
  extern template class ggnn_impl<float, Euclidean, DIM, 48, KQUERY, 64>; \
  extern template class ggnn_impl<float, Euclidean, DIM, 64, KQUERY, 64>; \
  extern template class ggnn_impl<float, Euclidean, DIM, 96, KQUERY, 64>;

INSTANTIATE_EXTERN(100, 10);
INSTANTIATE_EXTERN(128, 10);
INSTANTIATE_EXTERN(784, 10);
INSTANTIATE_EXTERN(960, 10);
INSTANTIATE_EXTERN(100, 100);
INSTANTIATE_EXTERN(128, 100);
INSTANTIATE_EXTERN(784, 100);
INSTANTIATE_EXTERN(960, 100);
#undef INSTANTIATE_EXTERN
}  // namespace cuvs::bench
