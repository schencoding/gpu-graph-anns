// Copyright (c) 2020 Jisang Yoon
// All rights reserved.
//
// This source code is licensed under the Apache 2.0 license found in the
// LICENSE file in the root directory of this source tree.
#include <cuda_runtime_api.h>
#include <fstream>
#include <numeric>
#include <raft/util/cuda_rt_essentials.hpp>

#include "cuvs/neighbors/cuhnsw_v2.hpp"

namespace cuhnsw_v2 {

CuHNSW::CuHNSW() : cores_(-1)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  cudaDeviceProp prop;
  RAFT_CUDA_TRY(cudaGetDeviceProperties(&prop, dev_id));
  const int mp_cnt = prop.multiProcessorCount;
  // reference: https://stackoverflow.com/a/32531982
  switch (prop.major) {
    case 2:  // Fermi
      if (prop.minor == 1)
        cores_ = mp_cnt * 48;
      else
        cores_ = mp_cnt * 32;
      break;
    case 3:  // Kepler
      cores_ = mp_cnt * 192;
      break;
    case 5:  // Maxwell
      cores_ = mp_cnt * 128;
      break;
    case 6:  // Pascal
      if (prop.minor == 1 or prop.minor == 2)
        cores_ = mp_cnt * 128;
      else if (prop.minor == 0)
        cores_ = mp_cnt * 64;
      else {
        RAFT_LOG_ERROR("Unknown device type");
      }
      break;
    case 7:  // Volta and Turing
      if (prop.minor == 0 or prop.minor == 5)
        cores_ = mp_cnt * 64;
      else {
        RAFT_LOG_ERROR("Unknown device type");
      }
      break;
    case 8:  // Ampere
      if (prop.minor == 0)
        cores_ = mp_cnt * 64;
      else if (prop.minor == 6)
        cores_ = mp_cnt * 128;
      else {
        RAFT_LOG_ERROR("Unknown device type");
      }
      break;
    default: RAFT_LOG_ERROR("Unknown device type"); throw;
  }
  if (cores_ == -1) cores_ = mp_cnt * 128;
  RAFT_LOG_INFO(
    "cuda device info, device_id: %d, major: %d, minor: %d, multi processors: %d, cores: %d",
    dev_id,
    prop.major,
    prop.minor,
    mp_cnt,
    cores_);
}

CuHNSW::~CuHNSW() {}

bool CuHNSW::Init(int max_m,
                  int max_m0,
                  bool save_remains,
                  int ef_construction,
                  // float level_mult,
                  // int batch_size,
                  int block_dim,
                  int hyper_threads,
                  int visited_table_size,
                  int visited_list_size,
                  float heuristic_coef,
                  enum DIST_TYPE dist_type,
                  bool reverse_cand,
                  int log_level)
{
  max_m_           = max_m;
  max_m0_          = max_m0;
  save_remains_    = save_remains;
  ef_construction_ = ef_construction;
  // level_mult_         = level_mult;
  // batch_size_         = batch_size;
  block_dim_          = block_dim;
  hyper_threads_      = hyper_threads;
  visited_table_size_ = visited_table_size;
  visited_list_size_  = visited_list_size;
  if (not visited_table_size_) visited_table_size_ = visited_list_size_ * 2;
  heuristic_coef_ = heuristic_coef;
  dist_type_      = dist_type;
  reverse_cand_   = reverse_cand;
  RAFT_LOG_DEBUG(
    "max_m: %d, max_m0: %d, save_remains: %d, ef_construction: %d, level_mult: N/A, dist_type: N/A",
    max_m_,
    max_m0_,
    save_remains_,
    ef_construction_);
  // dist_type);
  // level_mult_,
  return true;
}

void CuHNSW::SetData(const float* data, int num_data, int num_dims)
{
  num_data_ = num_data;
  num_dims_ = num_dims;
  // block_cnt_ = opt_["hyper_threads"].number_value() * (cores_ / block_dim_);
  block_cnt_ = hyper_threads_ * (cores_ / block_dim_);
  RAFT_LOG_DEBUG("copy data (%d x %d), block_cnt: %d, block_dim: %d",
                 num_data,
                 num_dims,
                 block_cnt_,
                 block_dim_);
  device_data_.resize(num_data * num_dims);
#ifdef HALF_PRECISION
  // DEBUG0("fp16")
  std::vector<cuda_scalar> hdata(num_data * num_dims);
  for (int i = 0; i < num_data * num_dims; ++i) {
    hdata[i] = conversion(data[i]);
    // DEBUG("hdata i: {}, scalar: {}", i, out_scalar(hdata[i]));
  }
  thrust::copy(hdata.begin(), hdata.end(), device_data_.begin());
#else
  thrust::copy(data, data + num_data * num_dims, device_data_.begin());
#endif
  data_ = data;
}

void CuHNSW::SetRandomLevels(const int* levels)
{
  levels_.resize(num_data_);
  RAFT_LOG_DEBUG("set levels of data (length: %d)", num_data_);
  max_level_ = 0;
  std::vector<std::vector<int>> level_nodes(1);
  for (int i = 0; i < num_data_; ++i) {
    levels_[i] = levels[i];
    if (levels[i] > max_level_) {
      max_level_ = levels[i];
      level_nodes.resize(max_level_ + 1);
      enter_point_ = i;
    }
    for (int l = 0; l <= levels[i]; ++l)
      level_nodes[l].push_back(i);
  }
  RAFT_LOG_DEBUG("max level: %d", max_level_);
  for (int i = 0; i <= max_level_; ++i) {
    RAFT_LOG_DEBUG("number of data in level %d: %lu", i, level_nodes[i].size());
  }
  level_graphs_.clear();
  for (int i = 0; i <= max_level_; ++i) {
    LevelGraph graph = LevelGraph();
    graph.SetNodes(level_nodes[i], num_data_, ef_construction_);
    level_graphs_.push_back(graph);
  }
}

// save graph compatible with hnswlib (https://github.com/nmslib/hnswlib)
void CuHNSW::SaveIndex(std::string fpath)
{
  std::ofstream output(fpath);
  RAFT_LOG_DEBUG("save index to %d", fpath);

  // write meta values
  size_t data_size              = num_dims_ * sizeof(scalar);
  size_t max_elements           = num_data_;
  size_t cur_element_count      = num_data_;
  size_t M                      = max_m_;
  size_t maxM                   = max_m_;
  size_t maxM0                  = max_m0_;
  int maxlevel                  = max_level_;
  size_t size_links_level0      = maxM0 * sizeof(tableint) + sizeof(sizeint);
  size_t size_links_per_element = maxM * sizeof(tableint) + sizeof(sizeint);
  size_t size_data_per_element  = size_links_level0 + data_size + sizeof(labeltype);
  size_t ef_construction        = ef_construction_;
  // double mult = level_mult_;
  size_t offsetData        = size_links_level0;
  size_t label_offset      = size_links_level0 + data_size;
  size_t offsetLevel0      = 0;
  tableint enterpoint_node = enter_point_;

  writeBinaryPOD(output, offsetLevel0);
  writeBinaryPOD(output, max_elements);
  writeBinaryPOD(output, cur_element_count);
  writeBinaryPOD(output, size_data_per_element);
  writeBinaryPOD(output, label_offset);
  writeBinaryPOD(output, offsetData);
  writeBinaryPOD(output, maxlevel);
  writeBinaryPOD(output, enterpoint_node);
  writeBinaryPOD(output, maxM);
  writeBinaryPOD(output, maxM0);
  writeBinaryPOD(output, M);
  // writeBinaryPOD(output, mult);
  writeBinaryPOD(output, ef_construction);

  // write level0 links and data
  char* data_level0_memory = (char*)malloc(cur_element_count * size_data_per_element);
  LevelGraph& graph        = level_graphs_[0];
  std::vector<tableint> links;
  links.reserve(max_m0_);
  size_t offset = 0;
  for (size_t i = 0; i < cur_element_count; ++i) {
    links.clear();
    for (const auto& pr : graph.GetNeighbors(i))
      links.push_back(static_cast<tableint>(pr.second));

    sizeint size = links.size();
    memcpy(data_level0_memory + offset, &size, sizeof(sizeint));
    offset += sizeof(sizeint);
    if (size > 0) memcpy(data_level0_memory + offset, &links[0], sizeof(tableint) * size);
    offset += maxM0 * sizeof(tableint);
    memcpy(data_level0_memory + offset, &data_[i * num_dims_], data_size);
    offset += data_size;
    labeltype label = i;
    memcpy(data_level0_memory + offset, &label, sizeof(labeltype));
    offset += sizeof(labeltype);
  }
  output.write(data_level0_memory, cur_element_count * size_data_per_element);

  // write upper layer links
  for (int i = 0; i < num_data_; ++i) {
    unsigned int size = size_links_per_element * levels_[i];
    writeBinaryPOD(output, size);
    char* mem = (char*)malloc(size);
    offset    = 0;
    if (size) {
      for (int j = 1; j <= levels_[i]; ++j) {
        links.clear();
        LevelGraph& upper_graph = level_graphs_[j];
        for (const auto& pr : upper_graph.GetNeighbors(i))
          links.push_back(static_cast<tableint>(pr.second));
        sizeint link_size = links.size();
        memcpy(mem + offset, &link_size, sizeof(sizeint));
        offset += sizeof(sizeint);
        if (link_size > 0) memcpy(mem + offset, &links[0], sizeof(tableint) * link_size);
        offset += sizeof(tableint) * maxM;
      }
      output.write(mem, size);
    }
  }

  output.close();
}

// load graph compatible with hnswlib (https://github.com/nmslib/hnswlib)
void CuHNSW::LoadIndex(std::string fpath)
{
  std::ifstream input(fpath, std::ios::binary);
  RAFT_LOG_DEBUG("load index from %d", fpath);

  // reqd meta values
  size_t offsetLevel0, max_elements, cur_element_count;
  size_t size_data_per_element, label_offset, offsetData;
  int maxlevel;
  tableint enterpoint_node = enter_point_;
  size_t maxM, maxM0, M;
  // double mult;
  size_t ef_construction;

  readBinaryPOD(input, offsetLevel0);
  readBinaryPOD(input, max_elements);
  readBinaryPOD(input, cur_element_count);
  readBinaryPOD(input, size_data_per_element);
  readBinaryPOD(input, label_offset);
  readBinaryPOD(input, offsetData);
  readBinaryPOD(input, maxlevel);
  readBinaryPOD(input, enterpoint_node);
  readBinaryPOD(input, maxM);
  readBinaryPOD(input, maxM0);
  readBinaryPOD(input, M);
  // readBinaryPOD(input, mult);
  readBinaryPOD(input, ef_construction);
  size_t size_per_link = maxM * sizeof(tableint) + sizeof(sizeint);
  num_data_            = cur_element_count;
  max_m_               = maxM;
  max_m0_              = maxM0;
  enter_point_         = enterpoint_node;
  ef_construction_     = ef_construction;
  max_level_           = maxlevel;
  // level_mult_ = mult;
  num_dims_ = (label_offset - offsetData) / sizeof(scalar);
  RAFT_LOG_DEBUG(
    "meta values loaded, num_data: {}, num_dims: {}, max_m: {}, max_m0: {}, enter_point: {}, "
    "max_level: {}",
    num_data_,
    num_dims_,
    max_m_,
    max_m0_,
    enter_point_,
    max_level_);

  char* data_level0_memory = (char*)malloc(max_elements * size_data_per_element);
  input.read(data_level0_memory, cur_element_count * size_data_per_element);

  // reset level graphs
  level_graphs_.clear();
  level_graphs_.shrink_to_fit();
  level_graphs_.resize(max_level_ + 1);

  // load data and level0 links
  RAFT_LOG_DEBUG("level0 count: %d", cur_element_count);
  std::vector<float> data(num_data_ * num_dims_);
  size_t offset = 0;
  std::vector<tableint> links(max_m0_);
  std::vector<scalar> vec_data(num_dims_);
  LevelGraph& graph0 = level_graphs_[0];
  std::vector<std::vector<int>> nodes(max_level_ + 1);
  nodes[0].resize(cur_element_count);
  std::iota(nodes[0].begin(), nodes[0].end(), 0);
  graph0.SetNodes(nodes[0], num_data_, ef_construction_);
  labels_.clear();
  labelled_ = true;
  for (size_t i = 0; i < cur_element_count; ++i) {
    sizeint deg;
    memcpy(&deg, data_level0_memory + offset, sizeof(sizeint));
    offset += sizeof(sizeint);
    memcpy(&links[0], data_level0_memory + offset, sizeof(tableint) * max_m0_);
    for (sizeint j = 0; j < deg; ++j)
      graph0.AddEdge(i, links[j], 0);
    offset += sizeof(tableint) * max_m0_;
    memcpy(&vec_data[0], data_level0_memory + offset, sizeof(scalar) * num_dims_);
    for (int j = 0; j < num_dims_; ++j)
      data[num_dims_ * i + j] = vec_data[j];
    offset += sizeof(scalar) * num_dims_;
    labeltype label;
    memcpy(&label, data_level0_memory + offset, sizeof(labeltype));
    labels_.push_back(static_cast<int>(label));
    offset += sizeof(labeltype);
  }
  SetData(&data[0], num_data_, num_dims_);

  // load upper layer links
  std::vector<std::vector<std::pair<int, int>>> links_data(max_level_ + 1);
  links.resize(max_m_);
  levels_.resize(cur_element_count);
  for (size_t i = 0; i < cur_element_count; ++i) {
    unsigned int linksize;
    readBinaryPOD(input, linksize);
    if (not linksize) continue;
    char* buffer = (char*)malloc(linksize);
    input.read(buffer, linksize);
    size_t levels = linksize / size_per_link;
    size_t offset = 0;
    levels_[i]    = levels + 1;
    for (size_t j = 1; j <= levels; ++j) {
      nodes[j].push_back(i);
      sizeint deg;
      memcpy(&deg, buffer + offset, sizeof(sizeint));
      offset += sizeof(sizeint);
      memcpy(&links[0], buffer + offset, sizeof(tableint) * deg);
      offset += sizeof(tableint) * max_m_;
      for (sizeint k = 0; k < deg; ++k)
        links_data[j].emplace_back(i, links[k]);
    }
  }

  for (int i = 1; i <= max_level_; ++i) {
    LevelGraph& graph = level_graphs_[i];
    RAFT_LOG_DEBUG("level {} count: {}", i, nodes[i].size());
    graph.SetNodes(nodes[i], num_data_, ef_construction_);
    for (const auto& pr : links_data[i]) {
      graph.AddEdge(pr.first, pr.second, 0);
    }
  }

  input.close();
}

}  // namespace cuhnsw_v2
