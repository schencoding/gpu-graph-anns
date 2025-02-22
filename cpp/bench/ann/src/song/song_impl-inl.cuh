#pragma once
#include "song_impl.cuh"
#include <thread>
#include <vector>

namespace cuvs::bench {

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::add_data(const float* dataset,
                                                               size_t nrow,
                                                               bool add_vertex)
{
  std::vector<std::thread> threads;
  size_t step = nrow / build_param_.num_threads;
  for (int i = 0; i < build_param_.num_threads; ++i) {
    size_t start = i * step;
    size_t end   = (i == build_param_.num_threads - 1) ? nrow : (i + 1) * step;
    threads.push_back(std::thread([this, dataset, start, end, add_vertex] {
      for (size_t row = start; row < end; ++row) {
        std::vector<std::pair<int, value_t>> vec;
        vec.reserve(dim_);
        const value_t* ptr = dataset + dim_ * row;
        for (int col = 0; col < dim_; ++col) {
          vec.emplace_back(col, ptr[col]);
        }
        // load data
        data_->add(row, vec);
        // build graph
        if (add_vertex) { song_->add_vertex(row, vec); }
      }
    }));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::build(const float* dataset, size_t nrow)
{
  base_dataset_ = dataset;
  base_n_rows_  = nrow;
  data_         = std::make_shared<Data>(nrow, dim_);
  song_         = std::make_shared<song_instance>(data_.get(), build_param_.degree);

  add_data(dataset, nrow, true);
}

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::set_search_param(
  const search_param_base& param)
{
  search_param_ = dynamic_cast<const typename song::search_param&>(param);
}

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::search(const float* queries,
                                                             int batch_size,
                                                             int k,
                                                             algo_base::index_type* neighbors,
                                                             float* distances) const
{
  std::vector<std::vector<std::pair<int, value_t>>> queries_vec;
  std::vector<std::vector<idx_t>> results;
  queries_vec.reserve(batch_size);

  for (int row = 0; row < batch_size; ++row) {
    const value_t* ptr = queries + row * dim_;
    std::vector<std::pair<int, value_t>> vec_row;
    vec_row.reserve(dim_);
    for (int col = 0; col < dim_; ++col) {
      vec_row.emplace_back(col, ptr[col]);
    }
    queries_vec.push_back(vec_row);
  }
  results.reserve(batch_size);
  switch (search_param_.pq_size) {
    case 10:
      song_->template search_top_k_batch<10>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 20:
      song_->template search_top_k_batch<20>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 30:
      song_->template search_top_k_batch<30>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 40:
      song_->template search_top_k_batch<40>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 50:
      song_->template search_top_k_batch<50>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 60:
      song_->template search_top_k_batch<60>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 70:
      song_->template search_top_k_batch<70>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 80:
      song_->template search_top_k_batch<80>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 90:
      song_->template search_top_k_batch<90>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 100:
      song_->template search_top_k_batch<100>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 150:
      song_->template search_top_k_batch<150>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 200:
      song_->template search_top_k_batch<200>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 400:
      song_->template search_top_k_batch<400>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 800:
      song_->template search_top_k_batch<800>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 1600:
      song_->template search_top_k_batch<1600>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 3200:
      song_->template search_top_k_batch<3200>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    case 6400:
      song_->template search_top_k_batch<6400>(queries_vec, k, results, search_param_.finish_cnt);
      break;
    default: throw std::runtime_error("unsupported pq_size");
  }
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < k; ++j) {
      // if (i < 10) { std::cout << results[i][j] << " "; }
      neighbors[i * k + j] = results[i][j];
    }
    // if (i < 10) { std::cout << std::endl; }
  }
}

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::save(const std::string& file) const
{
  song_->dump(file);
}

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::load(const std::string& file)
{
  file_ = file;
}

template <SongMetricType metric_type, int DIM, VisitedTableType visited_table_type>
void song_impl<metric_type, DIM, visited_table_type>::set_search_dataset(
  const data_value_t* dataset, size_t nrow)
{
  if (file_.empty()) { throw std::runtime_error("file is not set"); }
  data_ = std::make_shared<Data>(nrow, dim_);
  add_data(dataset, nrow, false);
  song_ = std::make_shared<song_instance>(data_.get(), build_param_.degree);
  song_->load(file_);
  song_->set_device_data_and_graph_if_nullptr();
};

#define INSTANTIATE_SONG_IMPL(dim) \
  template class song_impl<SongMetricType::L2, dim, VisitedTableType::kBloomFilter>; \
  template class song_impl<SongMetricType::L2, dim, VisitedTableType::kCuckooFilter>; \
  template class song_impl<SongMetricType::L2, dim, VisitedTableType::kHashTable>; \
  template class song_impl<SongMetricType::L2, dim, VisitedTableType::kHashTableSel>; \
  template class song_impl<SongMetricType::L2, dim, VisitedTableType::kHashTableSelDel>; \
  template class song_impl<SongMetricType::IP, dim, VisitedTableType::kBloomFilter>; \
  template class song_impl<SongMetricType::IP, dim, VisitedTableType::kCuckooFilter>; \
  template class song_impl<SongMetricType::IP, dim, VisitedTableType::kHashTable>; \
  template class song_impl<SongMetricType::IP, dim, VisitedTableType::kHashTableSel>; \
  template class song_impl<SongMetricType::IP, dim, VisitedTableType::kHashTableSelDel>;
}  // namespace cuvs::bench
