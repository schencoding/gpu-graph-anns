#pragma once
#include <cstdint>
#include <iostream>

namespace cuvs::neighbors::cagra::detail {

struct CagraMetrics {
  std::uint64_t clk_init;
  std::uint64_t clk_compute_1st_distance;
  std::uint64_t clk_topk;
  std::uint64_t clk_reset_hash;
  std::uint64_t clk_pickup_parents;
  std::uint64_t clk_restore_hash;
  std::uint64_t clk_insert_hashmap;
  std::uint64_t clk_compute_distance;

  uint64_t clk_counter;

  uint64_t global_distance_calculation_counter1;
  uint64_t global_distance_calculation_counter2;
  uint64_t global_distance_calculation_counter3;
  uint64_t global_distance_calculation_counter4;
  uint64_t global_distance_calculation_counter3_4_counter;

  __host__ __device__ void reset()
  {
    clk_init                 = 0;
    clk_compute_1st_distance = 0;
    clk_topk                 = 0;
    clk_reset_hash           = 0;
    clk_pickup_parents       = 0;
    clk_restore_hash         = 0;
    clk_insert_hashmap       = 0;
    clk_compute_distance     = 0;
    clk_counter              = 0;

    global_distance_calculation_counter1           = 0;
    global_distance_calculation_counter2           = 0;
    global_distance_calculation_counter3           = 0;
    global_distance_calculation_counter4           = 0;
    global_distance_calculation_counter3_4_counter = 0;
  }
};

enum class CagraKernelType : int {
  kUnknown   = 0,
  kSingleCta = 1,
  kMultiCta  = 2,
};

struct CagraMetricsAccumulator {
 private:
  CagraMetricsAccumulator() { metrics.reset(); }

 public:
  CagraMetricsAccumulator(const CagraMetricsAccumulator&)            = delete;
  CagraMetricsAccumulator(CagraMetricsAccumulator&&)                 = delete;
  CagraMetricsAccumulator& operator=(const CagraMetricsAccumulator&) = delete;
  CagraMetricsAccumulator& operator=(CagraMetricsAccumulator&&)      = delete;

  static CagraMetricsAccumulator& get_instance()
  {
    static CagraMetricsAccumulator instance;
    return instance;
  }

  CagraMetrics metrics{};
  uint64_t num_executed_iterations{};
  uint64_t num_queries{};
  CagraKernelType kernel_type{CagraKernelType::kUnknown};

  void accumulate(const CagraMetrics& m,
                  uint32_t* const num_executed_iterations,
                  const uint32_t num_queries,
                  CagraKernelType kernel_type)
  {
    metrics.clk_init += m.clk_init;
    metrics.clk_compute_1st_distance += m.clk_compute_1st_distance;
    metrics.clk_topk += m.clk_topk;
    metrics.clk_reset_hash += m.clk_reset_hash;
    metrics.clk_pickup_parents += m.clk_pickup_parents;
    metrics.clk_restore_hash += m.clk_restore_hash;
    metrics.clk_insert_hashmap += m.clk_insert_hashmap;
    metrics.clk_compute_distance += m.clk_compute_distance;
    metrics.clk_counter += m.clk_counter;

    metrics.global_distance_calculation_counter1 += m.global_distance_calculation_counter1;
    metrics.global_distance_calculation_counter2 += m.global_distance_calculation_counter2;
    metrics.global_distance_calculation_counter3 += m.global_distance_calculation_counter3;
    metrics.global_distance_calculation_counter4 += m.global_distance_calculation_counter4;
    metrics.global_distance_calculation_counter3_4_counter +=
      m.global_distance_calculation_counter3_4_counter;

    for (uint32_t i = 0; i < num_queries; ++i) {
      this->num_executed_iterations += num_executed_iterations[i];
    }
    this->num_queries += num_queries;
    this->kernel_type = kernel_type;
  }

  void reset()
  {
    metrics.reset();
    num_executed_iterations = 0;
    num_queries             = 0;
    kernel_type             = CagraKernelType::kUnknown;
  }

  void print_metrics() const {
    std::cout << "clk_init: " << metrics.clk_init << std::endl;
    std::cout << "clk_compute_1st_distance: " << metrics.clk_compute_1st_distance << std::endl;
    std::cout << "clk_topk: " << metrics.clk_topk << std::endl;
    std::cout << "clk_reset_hash: " << metrics.clk_reset_hash << std::endl;
    std::cout << "clk_pickup_parents: " << metrics.clk_pickup_parents << std::endl;
    std::cout << "clk_restore_hash: " << metrics.clk_restore_hash << std::endl;
    std::cout << "clk_insert_hashmap: " << metrics.clk_insert_hashmap << std::endl;
    std::cout << "clk_compute_distance: " << metrics.clk_compute_distance << std::endl;
    std::cout << "clk_counter: " << metrics.clk_counter << std::endl;
    std::cout << "global_distance_calculation_counter1: " << metrics.global_distance_calculation_counter1 << std::endl;
    std::cout << "global_distance_calculation_counter2: " << metrics.global_distance_calculation_counter2 << std::endl;
    std::cout << "global_distance_calculation_counter3: " << metrics.global_distance_calculation_counter3 << std::endl;
    std::cout << "global_distance_calculation_counter4: " << metrics.global_distance_calculation_counter4 << std::endl;
    std::cout << "global_distance_calculation_counter3_4_counter: " << metrics.global_distance_calculation_counter3_4_counter << std::endl;
    std::cout << "num_executed_iterations: " << num_executed_iterations << std::endl;
    std::cout << "num_queries: " << num_queries << std::endl;
    std::cout << "kernel_type: " << static_cast<int>(kernel_type) << std::endl;
  }
};

}  // namespace cuvs::neighbors::cagra::detail
