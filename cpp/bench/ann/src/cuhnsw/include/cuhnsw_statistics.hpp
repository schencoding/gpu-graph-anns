#pragma once
#include <iostream>
// #define _CLK_BREAKDOWN
namespace cuhnsw {
struct Statistics {
  unsigned long long counter_threadIdx_x0 = 0;
  unsigned long long clk_distance_computation = 0;
  unsigned long long distance_computation_counter = 0;
  unsigned long long distance_computation_counter_upper_layers = 0;
  unsigned long long num_queries = 0;

  // _visited_table
  unsigned long long clk_check_visited_table = 0;
  unsigned long long counter_check_visited_table = 0;
  unsigned long long clk_set_visited_table = 0;
  unsigned long long counter_set_visited_table = 0;

  // ef_search_pq
  unsigned long long clk_get_candidates = 0;
  unsigned long long counter_get_candidates = 0;
  unsigned long long clk_pq_pop = 0;
  unsigned long long counter_pq_pop = 0;
  unsigned long long clk_set_candidates_checked = 0;
  unsigned long long counter_set_candidates_checked = 0;
  unsigned long long clk_check_queue = 0;
  unsigned long long counter_check_queue = 0;
  unsigned long long clk_pq_push = 0;
  unsigned long long counter_pq_push = 0;

  unsigned long long clk_set_neighbors_found = 0;
  unsigned long long counter_set_neighbors_found = 0;

  static Statistics& GetInstance() {
    static Statistics instance;
    return instance;
  }

  void reset() {
    counter_threadIdx_x0 = 0;
    clk_distance_computation = 0;
    distance_computation_counter = 0;
    distance_computation_counter_upper_layers = 0;
    num_queries = 0;

    clk_check_visited_table = 0;
    counter_check_visited_table = 0;
    clk_set_visited_table = 0;
    counter_set_visited_table = 0;

    clk_get_candidates = 0;
    counter_get_candidates = 0;
    clk_pq_pop = 0;
    counter_pq_pop = 0;
    clk_set_candidates_checked = 0;
    counter_set_candidates_checked = 0;
    clk_check_queue = 0;
    counter_check_queue = 0;
    clk_pq_push = 0;
    counter_pq_push = 0;

    clk_set_neighbors_found = 0;
    counter_set_neighbors_found = 0;
  }
  void print() const {
    std::cout << "counter_threadIdx_x0: " << counter_threadIdx_x0 << std::endl;
    std::cout << "clk_distance_computation: " << clk_distance_computation << std::endl;
    std::cout << "distance_computation_counter: " << distance_computation_counter << std::endl;
    std::cout << "distance_computation_counter_upper_layers: " << distance_computation_counter_upper_layers << std::endl;
    std::cout << "num_queries: " << num_queries << std::endl;

    std::cout << "clk_check_visited_table: " << clk_check_visited_table << std::endl;
    std::cout << "counter_check_visited_table: " << counter_check_visited_table << std::endl;
    std::cout << "clk_set_visited_table: " << clk_set_visited_table << std::endl;
    std::cout << "counter_set_visited_table: " << counter_set_visited_table << std::endl;

    std::cout << "clk_get_candidates: " << clk_get_candidates << std::endl;
    std::cout << "counter_get_candidates: " << counter_get_candidates << std::endl;
    std::cout << "clk_pq_pop: " << clk_pq_pop << std::endl;
    std::cout << "counter_pq_pop: " << counter_pq_pop << std::endl;
    std::cout << "clk_set_candidates_checked: " << clk_set_candidates_checked << std::endl;
    std::cout << "counter_set_candidates_checked: " << counter_set_candidates_checked << std::endl;
    std::cout << "clk_check_queue: " << clk_check_queue << std::endl;
    std::cout << "counter_check_queue: " << counter_check_queue << std::endl;
    std::cout << "clk_pq_push: " << clk_pq_push << std::endl;
    std::cout << "counter_pq_push: " << counter_pq_push << std::endl;

    std::cout << "clk_set_neighbors_found: " << clk_set_neighbors_found << std::endl;
    std::cout << "counter_set_neighbors_found: " << counter_set_neighbors_found << std::endl;
  }

  void accumulate(const Statistics& other) {
    counter_threadIdx_x0 += other.counter_threadIdx_x0;
    clk_distance_computation += other.clk_distance_computation;
    distance_computation_counter += other.distance_computation_counter;
    distance_computation_counter_upper_layers += other.distance_computation_counter_upper_layers;
    num_queries += other.num_queries;

    clk_check_visited_table += other.clk_check_visited_table;
    counter_check_visited_table += other.counter_check_visited_table;
    clk_set_visited_table += other.clk_set_visited_table;
    counter_set_visited_table += other.counter_set_visited_table;

    clk_get_candidates += other.clk_get_candidates;
    counter_get_candidates += other.counter_get_candidates;
    clk_pq_pop += other.clk_pq_pop;
    counter_pq_pop += other.counter_pq_pop;
    clk_set_candidates_checked += other.clk_set_candidates_checked;
    counter_set_candidates_checked += other.counter_set_candidates_checked;
    clk_check_queue += other.clk_check_queue;
    counter_check_queue += other.counter_check_queue;
    clk_pq_push += other.clk_pq_push;
    counter_pq_push += other.counter_pq_push;

    clk_set_neighbors_found += other.clk_set_neighbors_found;
    counter_set_neighbors_found += other.counter_set_neighbors_found;
  }
};
}  // namespace cuhnsw
