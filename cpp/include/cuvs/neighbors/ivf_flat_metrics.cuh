#pragma once
#include <cstdint>
#include <iostream>

namespace cuvs::neighbors::ivf_flat::detail {
struct IVFFlatMetrics {
  uint64_t scan_distance_computation_counter = 0;
  uint64_t nlist_distance_computation_counter = 0;

  static IVFFlatMetrics& get_instance() {
    static IVFFlatMetrics instance;
    return instance;
  }

  void reset()
  {
    scan_distance_computation_counter = 0;
    nlist_distance_computation_counter = 0;
  }

  void print() {
    std::cout << "scan_distance_computation_counter: " << scan_distance_computation_counter << std::endl;
    std::cout << "nlist_distance_computation_counter: " << nlist_distance_computation_counter << std::endl;
  }
};
}  // namespace cuvs::neighbors::ivf_flat::detail
