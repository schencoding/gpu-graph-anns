#include "ggnn_impl.cuh"

namespace cuvs::bench {
template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
ggnn_impl<T, measure, D, KBuild, KQuery, S>::ggnn_impl(Metric metric,
                                                       int dim,
                                                       const typename ggnn<T>::build_param& param)
  : algo<T>(metric, dim), build_param_(param), stream_(cuvs::bench::get_stream_from_global_pool())
{
  if (metric_ == Metric::kInnerProduct) {
    if (measure != Cosine) { throw std::runtime_error("mis-matched metric"); }
  } else if (metric_ == Metric::kEuclidean) {
    if (measure != Euclidean) { throw std::runtime_error("mis-matched metric"); }
  } else {
    throw std::runtime_error(
      "ggnn supports only metric type of InnerProduct, Cosine and Euclidean");
  }

  if (dim != D) { throw std::runtime_error("mis-matched dim"); }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::build(const T* dataset, size_t nrow)
{
  base_dataset_ = dataset;
  base_n_rows_  = nrow;
  graph_file_   = std::nullopt;
  load_impl();
  ggnn_->build(0);
  for (int i = 0; i < build_param_.refine_iterations; ++i) {
    ggnn_->refine();
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::set_search_dataset(const T* dataset, size_t nrow)
{
  if (base_dataset_ != dataset || base_n_rows_ != nrow) {
    base_dataset_ = dataset;
    base_n_rows_  = nrow;
    load_impl();
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::set_search_param(const search_param_base& param)
{
  search_param_ = dynamic_cast<const typename ggnn<T>::search_param&>(param);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(sizeof(size_t) == sizeof(int64_t), "sizes of size_t and GGNN's KeyT are different");
  if (k != KQuery) {
    throw std::runtime_error(
      "k = " + std::to_string(k) +
      ", but this GGNN instance only supports k = " + std::to_string(KQuery));
  }

  ggnn_->set_stream(get_sync_stream());
  cudaMemcpyToSymbol(c_tau_query, &search_param_.tau_query, sizeof(float));

  const int block_dim      = search_param_.block_dim;
  const int max_iterations = search_param_.max_iterations;
  const int cache_size     = search_param_.cache_size;
  const int sorted_size    = search_param_.sorted_size;
  // default value
  if (block_dim == 32 && max_iterations == 400 && cache_size == 512 && sorted_size == 256) {
    ggnn_->template queryLayer<32, 400, 512, 256, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else if (block_dim == 32 && max_iterations == 1000 && cache_size == 512 && sorted_size == 256) {
    ggnn_->template queryLayer<32, 1000, 512, 256, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/sift1m.cu
  else if (block_dim == 32 && max_iterations == 200 && cache_size == 256 && sorted_size == 64) {
    ggnn_->template queryLayer<32, 200, 256, 64, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/sift1m.cu
  else if (block_dim == 32 && max_iterations == 400 && cache_size == 448 && sorted_size == 64) {
    ggnn_->template queryLayer<32, 400, 448, 64, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // ggnn/src/glove200.cu
  else if (block_dim == 128 && max_iterations == 2000 && cache_size == 2048 && sorted_size == 32) {
    ggnn_->template queryLayer<128, 2000, 2048, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  }
  // for glove100
  else if (block_dim == 64 && max_iterations == 400 && cache_size == 512 && sorted_size == 32) {
    ggnn_->template queryLayer<64, 400, 512, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else if (block_dim == 128 && max_iterations == 1000 && cache_size == 512 &&
             sorted_size == 256) {
    ggnn_->template queryLayer<128, 1000, 512, 256, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else if (block_dim == 128 && max_iterations == 1000 && cache_size == 1024 &&
             sorted_size == 32) {
    ggnn_->template queryLayer<128, 1000, 1024, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else if (block_dim == 128 && max_iterations == 2000 && cache_size == 1024 &&
             sorted_size == 32) {
    ggnn_->template queryLayer<128, 2000, 1024, 32, false>(
      queries, batch_size, reinterpret_cast<int64_t*>(neighbors), distances);
  } else {
    throw std::runtime_error("ggnn: not supported search param");
  }
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::save(const std::string& file) const
{
  auto& ggnn_host   = ggnn_->ggnn_cpu_buffers.at(0);
  auto& ggnn_device = ggnn_->ggnn_shards.at(0);
  ggnn_->set_stream(get_sync_stream());

  ggnn_host.downloadAsync(ggnn_device);
  cudaStreamSynchronize(ggnn_device.stream);
  ggnn_host.store(file);
}

template <typename T, DistanceMeasure measure, int D, int KBuild, int KQuery, int S>
void ggnn_impl<T, measure, D, KBuild, KQuery, S>::load(const std::string& file)
{
  if (!graph_file_.has_value() || graph_file_.value() != file) {
    graph_file_ = file;
    load_impl();
  }
}

#define INSTANTIATE(DIM)                                       \
  template class ggnn_impl<float, Euclidean, DIM, 24, 10, 32>; \
  template class ggnn_impl<float, Euclidean, DIM, 24, 10, 64>; \
  template class ggnn_impl<float, Euclidean, DIM, 48, 10, 32>; \
  template class ggnn_impl<float, Euclidean, DIM, 48, 10, 64>; \
  template class ggnn_impl<float, Euclidean, DIM, 64, 10, 64>; \
  template class ggnn_impl<float, Euclidean, DIM, 96, 10, 64>;

INSTANTIATE(100);
INSTANTIATE(128);
INSTANTIATE(768);
INSTANTIATE(960);
#undef INSTANTIATE
}  // namespace cuvs::bench
