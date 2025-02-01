header1 = """
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

#pragma once
#include <ggnn/cuda_knn_ggnn_gpu_instance.cuh>
#include "ggnn/utils/cuda_knn_constants.cuh"
"""

header2 = """
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
"""
with open("ggnn_gpu_instance-ext.cuh", "w") as f:
    f.write(header1)
    for metric in ["Cosine", "Euclidean"]:
        for base_type in ["float"]:
            for dim in [100, 128, 784, 960]:
                for k_build in [24, 64, 96]:
                    for k_query in [10]:
                        for segment_size in [32, 64]:
                            f.write(
f"""
extern template struct GGNNGPUInstance<{metric},
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           {base_type} /* BaseT */,
                                           size_t /* BAddrT */,
                                           {dim},
                                           {k_build},
                                           {k_build} / 2 /* KF */,
                                           {k_query},
                                           {segment_size}>;
""")
                            for (block_dim, max_iterations, cache_size, sorted_size) in [
                                    (32, 400, 512, 256),
                                    (32, 1000, 512, 256),
                                    (32, 200, 256, 64),
                                    (32, 400, 448, 64),
                                    (128, 2000, 2048, 32),
                                    (64, 400, 512, 32),
                                    (128, 2000, 1024, 32),
                                ]:
                                for dist_stats in ["true", "false"]:
                                    f.write(f"""
    extern template void GGNNGPUInstance<{metric},
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           {base_type} /* BaseT */,
                                           size_t /* BAddrT */,
                                           {dim},
                                           {k_build},
                                           {k_build} / 2 /* KF */,
                                           {k_query},
                                           {segment_size}>::queryLayer<{block_dim}, {max_iterations}, {cache_size}, {sorted_size}, {dist_stats}>(const {base_type}* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
                                    """)
                            with open(f"ggnn_gpu_instance_{metric}_{dim}_{k_build}_{k_query}_{segment_size}.cu", "w") as f2:
                                f2.write(header2)
                                f2.write(
f"""
template struct GGNNGPUInstance<{metric},
                               int64_t /* KeyT */,
                               float /* ValueT */,
                               size_t /* GAddrT */,
                               {base_type} /* BaseT */,
                               size_t /* BAddrT */,
                               {dim},
                               {k_build},
                               {k_build} / 2 /* KF */,
                               {k_query},
                               {segment_size}>;
""")
                                for (block_dim, max_iterations, cache_size, sorted_size) in [
                                        (32, 400, 512, 256),
                                        (32, 1000, 512, 256),
                                        (32, 200, 256, 64),
                                        (32, 400, 448, 64),
                                        (128, 2000, 2048, 32),
                                        (64, 400, 512, 32),
                                        (128, 2000, 1024, 32),
                                    ]:
                                    for dist_stats in ["true", "false"]:
                                        f2.write(
    f"""
    template void GGNNGPUInstance<{metric},
                                           int64_t /* KeyT */,
                                           float /* ValueT */,
                                           size_t /* GAddrT */,
                                           {base_type} /* BaseT */,
                                           size_t /* BAddrT */,
                                           {dim},
                                           {k_build},
                                           {k_build} / 2 /* KF */,
                                           {k_query},
                                           {segment_size}>::queryLayer<{block_dim}, {max_iterations}, {cache_size}, {sorted_size}, {dist_stats}>(const {base_type}* d_query, int batch_size, int64_t* d_query_result_ids, float* d_query_result_dists, const int shard_id) const;
    """)
