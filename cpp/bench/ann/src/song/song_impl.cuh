#pragma once
#include "../common/ann_types.hpp"
#include "../common/util.hpp"
#include "song/kernelgraph.h"
#include "song_wrapper.cuh"

namespace cuvs::bench {

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
class song_impl : public algo<data_value_t>, public algo_gpu {
 public:
  using search_param_base = typename algo<data_value_t>::search_param;

  song_impl(Metric metric, int dim, const typename song::build_param& param)
    : algo<data_value_t>(metric, dim),
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
  void set_search_dataset(const data_value_t* dataset, size_t nrow) override;

  std::unique_ptr<algo<data_value_t>> copy() override
  {
    auto r = std::make_unique<song_impl>(*this);
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

 private:
  void add_data(const float* dataset, size_t nrow, bool add_vertex);

  using algo<data_value_t>::dim_;

  using song_instance = KernelFixedDegreeGraph<metric_type, DIM, visited_table_type>;
  std::shared_ptr<song_instance> song_;
  std::shared_ptr<Data> data_;
  std::string file_;
  typename song::build_param build_param_;
  typename song::search_param search_param_;
  cudaStream_t stream_;
  const float* base_dataset_ = nullptr;
  size_t base_n_rows_        = 0;
};

#define INSTANTIATE_EXTERN_SONG_IMPL(dim)                                                       \
  extern template class song_impl<SongMetricType::L2, dim, VisitedTableType::kBloomFilter>;     \
  extern template class song_impl<SongMetricType::L2, dim, VisitedTableType::kCuckooFilter>;    \
  extern template class song_impl<SongMetricType::L2, dim, VisitedTableType::kHashTable>;       \
  extern template class song_impl<SongMetricType::L2, dim, VisitedTableType::kHashTableSel>;    \
  extern template class song_impl<SongMetricType::L2, dim, VisitedTableType::kHashTableSelDel>; \
  extern template class song_impl<SongMetricType::IP, dim, VisitedTableType::kBloomFilter>;     \
  extern template class song_impl<SongMetricType::IP, dim, VisitedTableType::kCuckooFilter>;    \
  extern template class song_impl<SongMetricType::IP, dim, VisitedTableType::kHashTable>;       \
  extern template class song_impl<SongMetricType::IP, dim, VisitedTableType::kHashTableSel>;    \
  extern template class song_impl<SongMetricType::IP, dim, VisitedTableType::kHashTableSelDel>;

INSTANTIATE_EXTERN_SONG_IMPL(96)
INSTANTIATE_EXTERN_SONG_IMPL(100)
INSTANTIATE_EXTERN_SONG_IMPL(128)
INSTANTIATE_EXTERN_SONG_IMPL(256)
INSTANTIATE_EXTERN_SONG_IMPL(784)
INSTANTIATE_EXTERN_SONG_IMPL(960)
}  // namespace cuvs::bench
