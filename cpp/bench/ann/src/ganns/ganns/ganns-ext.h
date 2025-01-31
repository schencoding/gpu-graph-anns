#pragma once
#include "data.h"
#include "ganns.h"
#include "graph_index/hierarchical_navigable_small_world.h"
#include "graph_index/navigable_small_world.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

using namespace std;
namespace ganns {
GANNS::~GANNS() {
  if (graph_ != nullptr) {
    delete graph_;
    graph_ = nullptr;
  }
}

void GANNS::Dump(std::string graph_name) { return graph_->Dump(graph_name); }
void GANNS::Establishment(int num_of_initial_neighbors, int num_of_candidates) {
  return graph_->Establishment(num_of_initial_neighbors, num_of_candidates);
}
void GANNS::SearchTopKonDevice(float *queries, int num_of_topk, int *&results,
                               int num_of_query_points, int num_of_candidates) {
  return graph_->SearchTopKonDevice(queries, num_of_topk, results,
                                    num_of_query_points, num_of_candidates);
}
void GANNS::DisplayGraphParameters(int num_of_candidates) {
  return graph_->DisplayGraphParameters(num_of_candidates);
}
void GANNS::DisplaySearchParameters(int num_of_topk, int num_of_candidates) {
  return graph_->DisplaySearchParameters(num_of_topk, num_of_candidates);
}

template <ganns::MetricType metric_type, int DIM, bool collect_metrics>
void GANNS::AddGraph(std::string graph_type, Data *points) {
  if (graph_type == "nsw") {
    graph_ =
        new NavigableSmallWorldGraphWithFixedDegree<metric_type, DIM, collect_metrics>(points);
  } else if (graph_type == "hnsw") {
    graph_ = new HierarchicalNavigableSmallWorld<metric_type, DIM, collect_metrics>(points);
  }
}


void GANNS::Load(std::string graph_path) { graph_->Load(graph_path); }

void GANNS::FreeResults(int* results)
  {
    graph_->FreeResults(results);
  }
} // namespace ganns
