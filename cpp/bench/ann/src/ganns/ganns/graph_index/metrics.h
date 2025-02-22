#pragma once
#include <iostream>

namespace ganns {
struct Metrics {
  unsigned long long counter_thread_idx_x0 = 0;
  unsigned long long counter_iterations = 0;

  unsigned long long stage_init = 0;
  unsigned long long stage_init_distance_computation = 0;
  unsigned long long stage_1 = 0;
  unsigned long long stage_2 = 0;
  unsigned long long stage_3_distance_computation = 0;
  unsigned long long stage_4 = 0;
  unsigned long long stage_5 = 0;
  unsigned long long stage_6 = 0;
  unsigned long long counter_stage1 = 0;
  unsigned long long counter_stage2 = 0;
  unsigned long long counter_stage4 = 0;
  unsigned long long counter_stage5 = 0;
  unsigned long long counter_stage6 = 0;
  unsigned long long stage_final = 0;

  unsigned long long distance_computation_counter = 0;
  unsigned long long metric_queries = 0;

  static Metrics& get_instance() {
    static Metrics instance;
    return instance;
  }

  void reset() {
    counter_thread_idx_x0 = 0;
    counter_iterations = 0;

    stage_init = 0;
    stage_init_distance_computation = 0;
    stage_1 = 0;
    stage_2 = 0;
    stage_3_distance_computation = 0;
    stage_4 = 0;
    stage_5 = 0;
    stage_6 = 0;
    counter_stage1 = 0;
    counter_stage2 = 0;
    counter_stage4 = 0;
    counter_stage5 = 0;
    counter_stage6 = 0;
    stage_final = 0;

    distance_computation_counter = 0;
    metric_queries = 0;
  }

  void accumulate(const Metrics& other) {
    counter_thread_idx_x0 += other.counter_thread_idx_x0;
    counter_iterations += other.counter_iterations;

    stage_init += other.stage_init;
    stage_init_distance_computation += other.stage_init_distance_computation;
    stage_1 += other.stage_1;
    stage_2 += other.stage_2;
    stage_3_distance_computation += other.stage_3_distance_computation;
    stage_4 += other.stage_4;
    stage_5 += other.stage_5;
    stage_6 += other.stage_6;
    counter_stage1 += other.counter_stage1;
    counter_stage2 += other.counter_stage2;
    counter_stage4 += other.counter_stage4;
    counter_stage5 += other.counter_stage5;
    counter_stage6 += other.counter_stage6;
    stage_final += other.stage_final;

    distance_computation_counter += other.distance_computation_counter;
    metric_queries += other.metric_queries;
  }

  void print_metrics() {
    std::cout << "counter_thread_idx_x0: " << counter_thread_idx_x0 << std::endl;
    std::cout << "counter_iterations: " << counter_iterations << std::endl;

    std::cout << "stage_init: " << stage_init << std::endl;
    std::cout << "stage_init_distance_computation: " << stage_init_distance_computation << std::endl;
    std::cout << "stage_1: " << stage_1 << std::endl;
    std::cout << "stage_2: " << stage_2 << std::endl;
    std::cout << "stage_3_distance_computation: " << stage_3_distance_computation << std::endl;
    std::cout << "stage_4: " << stage_4 << std::endl;
    std::cout << "stage_5: " << stage_5 << std::endl;
    std::cout << "stage_6: " << stage_6 << std::endl;
    std::cout << "counter_stage1: " << counter_stage1 << std::endl;
    std::cout << "counter_stage2: " << counter_stage2 << std::endl;
    std::cout << "counter_stage4: " << counter_stage4 << std::endl;
    std::cout << "counter_stage5: " << counter_stage5 << std::endl;
    std::cout << "counter_stage6: " << counter_stage6 << std::endl;
    std::cout << "stage_final: " << stage_final << std::endl;

    std::cout << "distance_computation_counter: " << distance_computation_counter << std::endl;
    std::cout << "metric_queries: " << metric_queries << std::endl;
  }
};

}  // namespace ganns
