#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


###############################################################################
#                                 Utilities                                   #
###############################################################################

dtype_sizes = {
    "float": 4,
    "fp8": 1,
    "half": 2,
}


###############################################################################
#                              cuVS constraints                               #
###############################################################################


def cuvs_cagra_build(params, dims):
    if "graph_degree" in params and "intermediate_graph_degree" in params:
        return params["graph_degree"] <= params["intermediate_graph_degree"]
    return True


def cuvs_ivf_pq_build(params, dims):
    if "pq_dim" in params:
        return params["pq_dim"] <= dims
    return True


def cuvs_ivf_pq_search(params, build_params, k, batch_size):
    ret = True
    if "internalDistanceDtype" in params and "smemLutDtype" in params:
        ret = (
            dtype_sizes[params["smemLutDtype"]]
            <= dtype_sizes[params["internalDistanceDtype"]]
        )

    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


def cuvs_cagra_search(params, build_params, k, batch_size):
    if "itopk" in params:
        return params["itopk"] >= k
    return True


###############################################################################
#                              FAISS constraints                              #
###############################################################################


def faiss_gpu_ivf_pq_build(params, dims):
    ret = True
    # M must be defined
    ret = params["M"] <= dims and dims % params["M"] == 0
    if "use_cuvs" in params and params["use_cuvs"]:
        return ret
    pq_bits = 8
    if "bitsPerCode" in params:
        pq_bits = params["bitsPerCode"]
    lookup_table_size = 4
    if "useFloat16" in params and params["useFloat16"]:
        lookup_table_size = 2
    # FAISS constraint to check if lookup table fits in shared memory
    # for now hard code maximum shared memory per block to 49 kB
    # (the value for A100 and V100)
    return ret and lookup_table_size * params["M"] * (2**pq_bits) <= 49152


def faiss_gpu_ivf_pq_search(params, build_params, k, batch_size):
    ret = True
    if "nlist" in build_params and "nprobe" in params:
        ret = ret and build_params["nlist"] >= params["nprobe"]
    return ret


###############################################################################
#                              hnswlib constraints                            #
###############################################################################


def hnswlib_search(params, build_params, k, batch_size):
    if "ef" in params:
        return params["ef"] >= k


###############################################################################
#                              cuhnsw constraints                             #
###############################################################################


def cuhnsw_search(params, build_params, k, batch_size):
    if "ef_search" in params:
        return params["ef_search"] >= k


_GGNN_VALID_BUILD_PARAMS = {
        # dims, k_build, k_query, segment_size
        (24, 10, 32),
        (24, 10, 64),
        (48, 10, 32),
        (48, 10, 64),
        (64, 10, 64),
        (96, 10, 64),
        (96, 10, 64),
        }


def ggnn_build(params, dims):
    return dims in [96, 100, 128, 784, 960] and  params["segment_size"] > params["k_build"] / 2 and (params["k_build"], params["k_query"], params["segment_size"]) in _GGNN_VALID_BUILD_PARAMS


_GGNN_VALID_SEARCH_PARAMS = {
        (32, 400, 512, 256),
        (32, 1000, 512, 256),
        (128, 1000, 512, 256),
        (128, 2000, 1024, 32),
        }


def ggnn_search(params, build_params, k, batch_size):
    return k == build_params["k_query"] and (params["block_dim"], params["max_iterations"], params["cache_size"], params["sorted_size"]) in _GGNN_VALID_SEARCH_PARAMS

###############################################################################
#                               ganns constraints                             #
###############################################################################


def ganns_build(params, dims):
    return dims <= 960


def ganns_search(params, build_params, k, batch_size):
    if "num_of_candidates_search" in params:
        return params["num_of_candidates_search"] >= k
    return False


###############################################################################
#                               song constraints                              #
###############################################################################


def song_build(params, dims):
    return True


def song_search(params, build_params, k, batch_size):
    return "pq_size" in params and params["pq_size"] >= k
