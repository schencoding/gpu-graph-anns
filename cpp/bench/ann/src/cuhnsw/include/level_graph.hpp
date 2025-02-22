// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <set>
#include <unordered_set>
#include <random>
#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <queue>
#include <functional>
#include <vector>
#include <unordered_map>

#include "log.hpp"

namespace cuhnsw {

class LevelGraph {
 public:
  LevelGraph() {
    logger_ = CuHNSWLogger().get_logger();
  }

  ~LevelGraph() {
    if (dev_deg_ != nullptr) {
      cudaFree(dev_deg_);
      dev_deg_ = nullptr;
    }
    if (dev_neighbors_ != nullptr) {
      cudaFree(dev_neighbors_);
      dev_neighbors_ = nullptr;
    }
    if (dev_upper_node_ != nullptr) {
      cudaFree(dev_upper_node_);
      dev_upper_node_ = nullptr;
    }
  }

  void SetNodes(std::vector<int>& nodes, int num_data, int ef_construction) {
    nodes_ = nodes;
    num_nodes_ = nodes_.size();
    neighbors_.clear();
    neighbors_.resize(num_nodes_);
    nodes_idmap_.resize(num_data);
    std::fill(nodes_idmap_.begin(), nodes_idmap_.end(), -1);
    for (int i = 0; i < num_nodes_; ++i)
      nodes_idmap_[nodes[i]] = i;
  }

  const std::vector<std::pair<float, int>>& GetNeighbors(int node) const  {
    int nodeid = GetNodeId(node);
    return neighbors_[nodeid];
  }

  const std::vector<int>& GetNodes() const {
    return nodes_;
  }

  void ClearEdges(int node) {
    neighbors_[GetNodeId(node)].clear();
  }

  void AddEdge(int src, int dst, float dist) {
    if (src == dst) return;
    int srcid = GetNodeId(src);
    neighbors_[srcid].emplace_back(dist, dst);
  }

  inline int GetNodeId(int node) const {
    int nodeid = nodes_idmap_.at(node);
    if (not(nodeid >= 0 and nodeid < num_nodes_)) {
      throw std::runtime_error(
          fmt::format("[{}:{}] invalid nodeid: {}, node: {}, num_nodes: {}",
            __FILE__, __LINE__, nodeid, node, num_nodes_));
    }
    return nodeid;
  }

  void ShowGraph() {
    for (int i = 0; i < num_nodes_; ++i) {
      std::cout << std::string(50, '=') << std::endl;
      printf("nodeid %d: %d\n", i, nodes_[i]);
      for (auto& nb: GetNeighbors(nodes_[i])) {
        printf("neighbor id: %d, dist: %f\n",
            nb.second, nb.first);
      }
      std::cout << std::string(50, '=') << std::endl;
    }
  }

  void set_dev_data_if_nullptr(int max_m) {
    if (dev_deg_ != nullptr || dev_neighbors_ != nullptr || dev_upper_node_ != nullptr) {
      throw std::runtime_error("dev_deg_ or dev_neighbor_  or dev_upper_node_ is not nullptr");
    }
    int upper_size = nodes_.size();
    std::vector<int> deg(upper_size);
    std::vector<int> neighbors(upper_size * max_m);
    for (int i = 0; i < upper_size; ++i) {
      const std::vector<std::pair<float, int>>& _neighbors = GetNeighbors(nodes_[i]);
      deg[i]                                               = _neighbors.size();
      int offset                                           = max_m * i;
      for (int j = 0; j < deg[i]; ++j) {
        neighbors[offset + j] = GetNodeId(_neighbors[j].second);
      }
    }

    cudaMalloc(&dev_upper_node_, sizeof(int) * upper_size);
    cudaMalloc(&dev_deg_, sizeof(int) * upper_size);
    cudaMalloc(&dev_neighbors_, sizeof(int) * upper_size * max_m);
    cudaMemcpy(dev_upper_node_, nodes_.data(), sizeof(int) * upper_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_deg_, deg.data(), sizeof(int) * upper_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_neighbors_, neighbors.data(), sizeof(int) * upper_size * max_m, cudaMemcpyHostToDevice);
    logger_->info("set_dev_data_if_nullptr. upper_size={} ", upper_size);
  }

 private:
  std::shared_ptr<spdlog::logger> logger_;
  std::vector<int> nodes_;
  std::vector<std::vector<std::pair<float, int>>> neighbors_;
  int num_nodes_ = 0;
  std::vector<int> nodes_idmap_;
 public:
  int* dev_deg_ = nullptr;
  int* dev_neighbors_ = nullptr;
  int* dev_upper_node_ = nullptr;
};  // class LevelGraph

} // namespace cuhnsw
