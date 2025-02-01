
/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ggnn_gpu_instance-ext.cuh"

template struct GGNNGPUInstance<Euclidean,
                               int64_t /* KeyT */,
                               float /* ValueT */,
                               size_t /* GAddrT */,
                               float /* BaseT */,
                               size_t /* BAddrT */,
                               960,
                               64,
                               64 / 2 /* KF */,
                               10,
                               64>;

    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 400, 512, 256, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 400, 512, 256, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 1000, 512, 256, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 1000, 512, 256, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 200, 256, 64, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 200, 256, 64, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 400, 448, 64, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<32, 400, 448, 64, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<128, 2000, 2048, 32, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<128, 2000, 2048, 32, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<64, 400, 512, 32, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<64, 400, 512, 32, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<128, 2000, 1024, 32, true>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    
    template void GGNNGPUInstance<Euclidean,
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           float /* BaseT */,
                                           size_t /* BAddrT */,
                                           960,
                                           64,
                                           64 / 2 /* KF */,
                                           10,
                                           64>::queryLayer<128, 2000, 1024, 32, false>(const float* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    