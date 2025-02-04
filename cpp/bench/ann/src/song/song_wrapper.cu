#include "song_wrapper.cuh"
#include "song_impl.cuh"

namespace cuvs::bench {
template <Metric metric>
void song::make_impl(int dim, const build_param& param, VisitedTableType visited_table_type)
{
  switch (visited_table_type) {
    case VisitedTableType::kBloomFilter:
      make_impl<metric, VisitedTableType::kBloomFilter>(dim, param);
      break;
    case VisitedTableType::kCuckooFilter:
      make_impl<metric, VisitedTableType::kCuckooFilter>(dim, param);
      break;
    case VisitedTableType::kHashTable:
      make_impl<metric, VisitedTableType::kHashTable>(dim, param);
      break;
    case VisitedTableType::kHashTableSel:
      make_impl<metric, VisitedTableType::kHashTableSel>(dim, param);
      break;
    case VisitedTableType::kHashTableSelDel:
      make_impl<metric, VisitedTableType::kHashTableSelDel>(dim, param);
      break;
  }
}

template <Metric metric, VisitedTableType visited_table_type>
void song::make_impl(int dim, const build_param& param)
{
  switch (dim) {
    case 96:
      impl_ =
        std::make_shared<song_impl<ConvertMetricTrait<metric>::metric, 96, visited_table_type>>(
          metric, dim, param);
      break;
    case 100:
      impl_ =
        std::make_shared<song_impl<ConvertMetricTrait<metric>::metric, 100, visited_table_type>>(
          metric, dim, param);
      break;
    case 128:
      impl_ =
        std::make_shared<song_impl<ConvertMetricTrait<metric>::metric, 128, visited_table_type>>(
          metric, dim, param);
      break;
    case 256:
      impl_ =
        std::make_shared<song_impl<ConvertMetricTrait<metric>::metric, 256, visited_table_type>>(
          metric, dim, param);
      break;
    case 784:
      impl_ =
        std::make_shared<song_impl<ConvertMetricTrait<metric>::metric, 784, visited_table_type>>(
          metric, dim, param);
      break;
    case 960:
      impl_ =
        std::make_shared<song_impl<ConvertMetricTrait<metric>::metric, 960, visited_table_type>>(
          metric, dim, param);
      break;
  }
}

song::song(Metric metric, int dim, const build_param& param, VisitedTableType visited_table_type)
  : algo<float>(metric, dim)
{
  switch (metric) {
    case Metric::kEuclidean: make_impl<Metric::kEuclidean>(dim, param, visited_table_type); break;
    case Metric::kInnerProduct:
      make_impl<Metric::kInnerProduct>(dim, param, visited_table_type);
      break;
  }
  if (!impl_) { throw std::runtime_error("unsupported metric or dim"); }
}
}  // namespace cuvs::bench
