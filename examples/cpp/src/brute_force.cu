#include <cstdint>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/matrix.hpp>

template <typename T>
auto read_bin(raft::resources &handle, const std::string &filename,
              int32_t subset_size = -1) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file");
  }
  std::cout << "Reading " << filename << ", subset_size=" << subset_size
            << std::endl;

  int32_t nvecs, dim;
  file.read(reinterpret_cast<char *>(&nvecs), sizeof(int32_t));
  file.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));

  if (subset_size >= 0) {
    nvecs = std::min(nvecs, subset_size);
  }

  auto res = raft::make_managed_matrix<T>(handle, static_cast<uint32_t>(nvecs),
                                          static_cast<uint32_t>(dim));

  file.read(reinterpret_cast<char *>(res.data_handle()),
            nvecs * dim * sizeof(T));

  return res;
}

template <typename T, typename MATRIX_VIEW>
void write_bin(const std::string &filename, MATRIX_VIEW data) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file");
  }

  int32_t nrows = data.extent(0);
  int32_t ncols = data.extent(1);
  file.write(reinterpret_cast<char *>(&nrows), sizeof(int32_t));
  file.write(reinterpret_cast<char *>(&ncols), sizeof(int32_t));
  file.write(reinterpret_cast<char *>(data.data_handle()),
             nrows * ncols * sizeof(T));
}

int main(int argc, char **argv) {
  cxxopts::Options options("brute_force", "Description of your program");

  options.add_options()("d,dataset", "Dataset file",
                        cxxopts::value<std::string>())(
      "q,queries", "Queries file", cxxopts::value<std::string>())(
      "s,subset_size", "Subset size", cxxopts::value<int32_t>());

  auto result = options.parse(argc, argv);
  if (!result.count("dataset") || !result.count("queries")) {
    std::cerr << "Usage: " << argv[0]
              << " --dataset <dataset> --queries <queries> [--subset_size"
                 "<subset_size>]"
              << std::endl;
    return 1;
  }
  std::string dataset_file = result["dataset"].as<std::string>();
  std::string queries_file = result["queries"].as<std::string>();
  int32_t subset_size = -1;
  if (result.count("subset_size")) {
    subset_size = result["subset_size"].as<int32_t>();
  }

  using namespace cuvs::neighbors;
  raft::resources handle;
  // L2 expaned
  brute_force::index_params index_params{
      /** Distance type. */
      cuvs::distance::DistanceType::L2Expanded,
      /** The argument used by some distance metrics. */
      2.0f};
  auto dataset = read_bin<float>(handle, dataset_file, subset_size);
  raft::device_matrix_view<const float, int64_t, raft::row_major> dataset_view =
      raft::make_device_matrix_view<const float, int64_t, row_major>(
          dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  std::cout << "Dataset size: " << dataset.extent(0) << "x" << dataset.extent(1)
            << " ptr: " << dataset.data_handle() << std::endl;
  brute_force::index<float, float> index =
      brute_force::build(handle, index_params, dataset_view);
  brute_force::search_params search_params;

  auto queries = read_bin<float>(handle, queries_file);
  std::cout << "Queries size: " << queries.extent(0) << "x" << queries.extent(1)
            << " ptr: " << queries.data_handle() << std::endl;
  // create output arrays
  int64_t topk = 100;
  auto queries_view =
      raft::make_device_matrix_view<const float, int64_t, row_major>(
          queries.data_handle(), queries.extent(0), queries.extent(1));

  int64_t n_queries = queries.extent(0);
  // create output arrays
  auto neighbors =
      raft::make_device_matrix<int64_t, int64_t>(handle, n_queries, topk);
  auto distances =
      raft::make_device_matrix<float, int64_t>(handle, n_queries, topk);
  const cuvs::neighbors::filtering::none_sample_filter filter{};
  brute_force::search(handle, search_params, index, queries_view,
                      neighbors.view(), distances.view(), filter);

  auto distances_host = raft::make_host_matrix<float>(handle, n_queries, topk);
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::update_host(distances_host.data_handle(), distances.data_handle(),
                    n_queries * topk, stream);
  stream.synchronize();
  auto neighbors_host_int32 =
      raft::make_host_matrix<int32_t>(handle, n_queries, topk);
  for (int i = 0; i < n_queries; i++) {
    for (int j = 0; j < topk; j++) {
      neighbors_host_int32(i, j) = static_cast<int32_t>(neighbors(i, j));
    }
  }

  for (int i = 0; i < 5; i++) {
    std::cout << "Query " << i << ": " << std::endl;
    std::cout << "  Neighbors:\n";
    for (int j = 0; j < topk; j++) {
      std::cout << neighbors_host_int32(i, j) << " ";
    }
    std::cout << "\n  Distances:\n";
    for (int j = 0; j < topk; j++) {
      std::cout << distances_host(i, j) << " ";
    }
    std::cout << std::endl;
  }

  write_bin<int32_t>("groundtruth.neighbors.ibin", neighbors_host_int32.view());
  write_bin<float>("groundtruth.distances.fbin", distances_host.view());
  if (subset_size > 0) {
    std::string subset_file = "subset." + std::to_string(subset_size) + ".fbin";
    write_bin<float>(subset_file, dataset.view());
  }
  return 0;
}
