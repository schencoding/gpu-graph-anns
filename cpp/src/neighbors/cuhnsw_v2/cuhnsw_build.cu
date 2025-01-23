// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <unordered_set>
#include <vector>

#include "cuda_build_kernels.cuh"
#include "cuda_search_kernels.cuh"
#include "cuvs/neighbors/cuhnsw_v2.hpp"

namespace cuhnsw_v2 {

template <class QueryDataAccessor>
void CuHNSW::GetEntryPoints(QueryDataAccessor qdata_accessor,
                            const int min_level,
                            std::vector<int>& entries)
{
  // assert(qdata);
  thrust::device_vector<int> dev_visited_list(visited_list_size_ * block_cnt_);
  for (int level = max_level_; level > min_level; --level) {
    int size = qdata_accessor.size();

    // process input data for kernel
    LevelGraph& graph                   = level_graphs_[level];
    const std::vector<int>& upper_nodes = graph.GetNodes();
    int upper_size                      = upper_nodes.size();
    std::vector<int> deg(upper_size);
    std::vector<int> neighbors(upper_size * max_m_);
    for (int i = 0; i < upper_size; ++i) {
      const std::vector<std::pair<float, int>>& _neighbors = graph.GetNeighbors(upper_nodes[i]);
      deg[i]                                               = _neighbors.size();
      int offset                                           = max_m_ * i;
      for (int j = 0; j < deg[i]; ++j) {
        neighbors[offset + j] = graph.GetNodeId(_neighbors[j].second);
      }
    }
    for (int i = 0; i < size; ++i) {
      int entryid = graph.GetNodeId(entries[i]);
      entries[i]  = entryid;
    }

    // copy to gpu mem
    thrust::device_vector<int> dev_entries(size);
    thrust::device_vector<int> dev_upper_nodes(upper_size), dev_deg(upper_size);
    thrust::device_vector<int> dev_neighbors(upper_size * max_m_);
    thrust::copy(entries.begin(), entries.end(), dev_entries.begin());
    thrust::copy(upper_nodes.begin(), upper_nodes.end(), dev_upper_nodes.begin());
    thrust::copy(deg.begin(), deg.end(), dev_deg.begin());
    thrust::copy(neighbors.begin(), neighbors.end(), dev_neighbors.begin());

    thrust::device_vector<bool> dev_visited(upper_size * block_cnt_, false);

    // BuildQueryDataAccessor target_data_accessor(thrust::raw_pointer_cast(device_data_.data()),
    //                                             thrust::raw_pointer_cast(dev_upper_nodes.data()),
    //                                             upper_size);

    // run kernel
    GetEntryPointsKernel<QueryDataAccessor>
      <<<block_cnt_, block_dim_>>>(qdata_accessor,
                                   thrust::raw_pointer_cast(device_data_.data()),
                                   thrust::raw_pointer_cast(dev_upper_nodes.data()),
                                   num_dims_,
                                   upper_size,
                                   max_m_,
                                   dist_type_,
                                   thrust::raw_pointer_cast(dev_neighbors.data()),
                                   thrust::raw_pointer_cast(dev_deg.data()),
                                   thrust::raw_pointer_cast(dev_visited.data()),
                                   thrust::raw_pointer_cast(dev_visited_list.data()),
                                   visited_list_size_,
                                   thrust::raw_pointer_cast(dev_entries.data()));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    // el_[GPU] += sw_[GPU].CheckPoint();
    thrust::copy(dev_entries.begin(), dev_entries.end(), entries.begin());

    // set output
    for (int i = 0; i < size; ++i) {
      entries[i] = upper_nodes[entries[i]];
    }
  }
}

void CuHNSW::BuildGraph()
{
  // visited_ = new bool[batch_size_ * num_data_];
  for (int level = max_level_; level >= 0; --level) {
    std::unordered_set<int> upper_nodes;
    LevelGraph& graph             = level_graphs_[level];
    const std::vector<int>& nodes = graph.GetNodes();
    int size                      = nodes.size();
    int max_m                     = level > 0 ? max_m_ : max_m0_;
    thrust::host_vector<int> graph_vec(size * max_m, 0);
    thrust::host_vector<int> deg(size, 0);
    if (level < max_level_) {
      LevelGraph& upper_graph = level_graphs_[level + 1];
      for (auto& node : upper_graph.GetNodes()) {
        upper_nodes.insert(node);
        int srcid = graph.GetNodeId(node);
        int idx   = 0;
        for (auto& nb : upper_graph.GetNeighbors(node)) {
          int dstid                          = graph.GetNodeId(nb.second);
          graph_vec[max_m * srcid + (idx++)] = dstid;
        }
        deg[srcid] = idx;
      }
    }

    std::vector<int> new_nodes;
    for (auto& node : graph.GetNodes()) {
      if (upper_nodes.count(node)) continue;
      new_nodes.push_back(node);
    }

    // initialize entries
    std::vector<int> entries(new_nodes.size(), enter_point_);

    thrust::device_vector<int> dev_nodes(new_nodes.size());
    thrust::copy(new_nodes.begin(), new_nodes.end(), dev_nodes.begin());
    BuildQueryDataAccessor qdata_accessor(thrust::raw_pointer_cast(device_data_.data()),
                                          thrust::raw_pointer_cast(dev_nodes.data()),
                                          new_nodes.size());

    GetEntryPoints<BuildQueryDataAccessor>(qdata_accessor, level, entries);
    for (size_t i = 0; i < new_nodes.size(); ++i) {
      int srcid                = graph.GetNodeId(new_nodes[i]);
      int dstid                = graph.GetNodeId(entries[i]);
      graph_vec[max_m * srcid] = dstid;
      deg[srcid]               = 1;
    }

    thrust::device_vector<NeighborIdxT> device_graph(max_m * size);
    thrust::device_vector<float> device_distances(max_m * size);
    thrust::device_vector<int> device_deg(size);
    thrust::device_vector<int> device_nodes(size);
    thrust::device_vector<int> device_visited_table(visited_table_size_ * block_cnt_, -1);
    thrust::device_vector<int> device_visited_list(visited_list_size_ * block_cnt_);
    thrust::device_vector<int> device_mutex(size, 0);
    thrust::device_vector<Neighbor<NeighborIdxT>> device_neighbors(ef_construction_ * block_cnt_);
    thrust::device_vector<int> device_cand_nodes(ef_construction_ * block_cnt_);
    thrust::device_vector<cuda_scalar> device_cand_distances(ef_construction_ * block_cnt_);
    thrust::device_vector<int> device_backup_neighbors(max_m * block_cnt_);
    thrust::device_vector<cuda_scalar> device_backup_distances(max_m * block_cnt_);
    thrust::device_vector<bool> device_went_through_heuristic(size, false);

    thrust::copy(graph_vec.begin(), graph_vec.end(), device_graph.begin());
    thrust::copy(deg.begin(), deg.end(), device_deg.begin());
    thrust::copy(nodes.begin(), nodes.end(), device_nodes.begin());

    BuildLevelGraphKernel<NeighborIdxT>
      <<<block_cnt_, block_dim_>>>(thrust::raw_pointer_cast(device_data_.data()),
                                   thrust::raw_pointer_cast(device_nodes.data()),
                                   num_dims_,
                                   size,
                                   max_m,
                                   dist_type_,
                                   save_remains_,
                                   ef_construction_,
                                   thrust::raw_pointer_cast(device_graph.data()),
                                   thrust::raw_pointer_cast(device_distances.data()),
                                   thrust::raw_pointer_cast(device_deg.data()),
                                   thrust::raw_pointer_cast(device_visited_table.data()),
                                   thrust::raw_pointer_cast(device_visited_list.data()),
                                   visited_table_size_,
                                   visited_list_size_,
                                   thrust::raw_pointer_cast(device_mutex.data()),
                                   reverse_cand_,
                                   thrust::raw_pointer_cast(device_neighbors.data()),
                                   thrust::raw_pointer_cast(device_cand_nodes.data()),
                                   thrust::raw_pointer_cast(device_cand_distances.data()),
                                   heuristic_coef_,
                                   thrust::raw_pointer_cast(device_backup_neighbors.data()),
                                   thrust::raw_pointer_cast(device_backup_distances.data()),
                                   thrust::raw_pointer_cast(device_went_through_heuristic.data()));
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    thrust::copy(device_deg.begin(), device_deg.end(), deg.begin());
    thrust::copy(device_graph.begin(), device_graph.end(), graph_vec.begin());
    std::vector<float> distances(max_m * size);
    thrust::copy(device_distances.begin(), device_distances.end(), distances.begin());

    for (auto& node : graph.GetNodes()) {
      graph.ClearEdges(node);
    }
    for (int i = 0; i < size; ++i) {
      int src = nodes[i];
      for (int j = 0; j < deg[i]; ++j) {
        int dst    = nodes[graph_vec[i * max_m + j]];
        float dist = distances[i * max_m + j];
        graph.AddEdge(src, dst, dist);
      }
    }
  }
}

void CuHNSW::SearchGraph(raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                         const int num_queries,
                         const int topk,
                         const int ef_search,
                         raft::device_matrix_view<NeighborIdxT, int64_t, raft::row_major> nns,
                         float* distances,
                         raft::device_vector_view<int> found_cnt)
{
  std::vector<int> entries(num_queries, enter_point_);
  SearchQueryDataAccessor qdata_accessor(queries.data_handle(), static_cast<size_t>(num_queries));
  GetEntryPoints<SearchQueryDataAccessor>(qdata_accessor, 0, entries);
  std::vector<int> graph_vec(max_m0_ * num_data_);
  std::vector<int> deg(num_data_);
  LevelGraph graph = level_graphs_[0];
  for (int i = 0; i < num_data_; ++i) {
    const std::vector<std::pair<float, int>>& neighbors = graph.GetNeighbors(i);
    int nbsize                                          = neighbors.size();
    int offset                                          = i * max_m0_;
    for (int j = 0; j < nbsize; ++j)
      graph_vec[offset + j] = neighbors[j].second;
    deg[i] = nbsize;
  }

  thrust::device_vector<int> device_graph(max_m0_ * num_data_);
  thrust::device_vector<int> device_deg(num_data_);
  thrust::device_vector<int> device_entries(num_queries);
  thrust::device_vector<float> device_distances(num_queries * topk);
  thrust::device_vector<int> device_visited_table(visited_table_size_ * block_cnt_, -1);
  thrust::device_vector<int> device_visited_list(visited_list_size_ * block_cnt_);
  thrust::device_vector<Neighbor<NeighborIdxT>> device_neighbors(ef_search * block_cnt_);
  thrust::device_vector<int> device_cand_nodes(ef_search * block_cnt_);
  thrust::device_vector<cuda_scalar> device_cand_distances(ef_search * block_cnt_);

  int* global_cand_nodes_ptr             = thrust::raw_pointer_cast(device_cand_nodes.data());
  cuda_scalar* global_cand_distances_ptr = thrust::raw_pointer_cast(device_cand_distances.data());

  thrust::copy(graph_vec.begin(), graph_vec.end(), device_graph.begin());
  thrust::copy(deg.begin(), deg.end(), device_deg.begin());
  thrust::copy(entries.begin(), entries.end(), device_entries.begin());
  SearchGraphKernel<<<block_cnt_, block_dim_>>>(
    queries.data_handle(),
    num_queries,
    thrust::raw_pointer_cast(device_data_.data()),
    num_data_,
    num_dims_,
    max_m0_,
    dist_type_,
    ef_search,
    thrust::raw_pointer_cast(device_entries.data()),
    thrust::raw_pointer_cast(device_graph.data()),
    thrust::raw_pointer_cast(device_deg.data()),
    topk,
    nns.data_handle(),
    thrust::raw_pointer_cast(device_distances.data()),
    found_cnt.data_handle(),
    thrust::raw_pointer_cast(device_visited_table.data()),
    thrust::raw_pointer_cast(device_visited_list.data()),
    visited_table_size_,
    visited_list_size_,
    reverse_cand_,
    thrust::raw_pointer_cast(device_neighbors.data()),
    global_cand_nodes_ptr,
    global_cand_distances_ptr);
  RAFT_CUDA_TRY(cudaGetLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  thrust::copy(device_distances.begin(), device_distances.end(), distances);
  // TODO(jiangyinzuo): add label support
  // if (labelled_)
  //   for (int i = 0; i < num_queries * topk; ++i)
  //     nns[i] = labels_[nns[i]];
}

}  // namespace cuhnsw_v2
