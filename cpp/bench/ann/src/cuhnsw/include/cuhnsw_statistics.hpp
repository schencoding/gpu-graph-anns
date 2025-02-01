#pragma once
#include <iostream>
#define _CLK_BREAKDOWN
namespace cuhnsw {
struct Statistics {
  unsigned long long distance_computation_counter = 0;
  unsigned long long distance_computation_counter_upper_layers = 0;
  unsigned long long num_queries = 0;

  unsigned long long clk_check_queue = 0;
  unsigned long long clk_distance_computation = 0;
  unsigned long long clk_update_priority_queue = 0;
  unsigned long long clk_check_visited_table = 0;
  unsigned long long clk_get_candidates = 0;
  unsigned long long clk_final = 0;

  static Statistics& GetInstance() {
    static Statistics instance;
    return instance;
  }

  void reset() {
    distance_computation_counter = 0;
    distance_computation_counter_upper_layers = 0;
    num_queries = 0;
    clk_check_queue = 0;
    clk_distance_computation = 0;
    clk_update_priority_queue = 0;
    clk_check_visited_table = 0;
    clk_get_candidates = 0;
    clk_final = 0;
  }
  void print() const {
    std::cout << "distance_computation_counter: " << distance_computation_counter
              << std::endl;
    std::cout << "distance_computation_counter_upper_layers: " << distance_computation_counter_upper_layers
              << std::endl;
    std::cout << "num_queries: " << num_queries << std::endl;
    std::cout << "clk_check_queue: " << clk_check_queue << std::endl;
    std::cout << "clk_distance_computation: " << clk_distance_computation << std::endl;
    std::cout << "clk_update_priority_queue: " << clk_update_priority_queue << std::endl;
    std::cout << "clk_check_visited_table: " << clk_check_visited_table << std::endl;
    std::cout << "clk_get_candidates: " << clk_get_candidates << std::endl;
    std::cout << "clk_final: " << clk_final << std::endl;
  }

  void accumulate(const Statistics& other) {
    distance_computation_counter += other.distance_computation_counter;
    distance_computation_counter_upper_layers += other.distance_computation_counter_upper_layers;
    num_queries += other.num_queries;
    clk_check_queue += other.clk_check_queue;
    clk_distance_computation += other.clk_distance_computation;
    clk_update_priority_queue += other.clk_update_priority_queue;
    clk_check_visited_table += other.clk_check_visited_table;
    clk_get_candidates += other.clk_get_candidates;
    clk_final += other.clk_final;
  }
};
}  // namespace cuhnsw
