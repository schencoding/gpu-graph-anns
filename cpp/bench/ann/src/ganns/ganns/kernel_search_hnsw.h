#pragma once

#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include<chrono>
#include<iostream>
#include "structure_on_device.h"
#include "metric_type.h"
#include "graph_index/metrics.h"

template <ganns::MetricType metric_type, int DIM, bool collect_metrics>
__global__
void SearchDevice(float* d_data, float* d_query, int* d_result, int* d_graph, int total_num_of_points, int num_of_query_points, int num_of_final_neighbors, 
                    int num_of_candidates, int num_of_results, int num_of_explored_points, int num_of_layers, int* prefix_sum_of_num_array_of_each_layer, ganns::Metrics* metrics) {
  auto stage_init_start = clock64();
	int t_id = threadIdx.x;
    int b_id = blockIdx.x;
    int size_of_warp = 32;

    extern __shared__ KernelPair<float, int> shared_memory_space_s[];
    KernelPair<float, int>* neighbors_array = shared_memory_space_s;
    int* flags = (int*)(shared_memory_space_s + num_of_candidates + num_of_final_neighbors);
    
    int crt_point_id = b_id;
    int* crt_result = d_result + crt_point_id * num_of_results;
    
    float q[(DIM + 31) / 32];
#pragma unroll
    for (int i = 0; i < (DIM + 31) / 32; ++i) {
        q[i] = 0;
        if (t_id + 32 * i < DIM) {
            q[i] = d_query[crt_point_id * DIM + t_id + i * 32];
        }
    }

    int step_id;
    int substep_id;

    int num_of_visited_points_one_batch;
    int length_of_compared_list;
    
    int flag_all_blocks = 1;

    int temporary_flag;
    int first_position_of_flag = 0;
    KernelPair<float, int> temporary_neighbor;
    int crt_layer = num_of_layers - 1;

    for (int i = 0; i < (num_of_candidates + num_of_final_neighbors + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;

        if (unrollt_id < num_of_candidates + num_of_final_neighbors) {
            flags[unrollt_id] = 0;

            neighbors_array[unrollt_id].first = Max;
            neighbors_array[unrollt_id].second = total_num_of_points;
        }
    }

    if (t_id == 0) {
        neighbors_array[0].second = 0;
        flags[0] = 1;
    }

    __syncthreads();
    if constexpr (collect_metrics) {
      if (t_id == 0) {
            atomicAdd(&metrics->stage_init, clock64() - stage_init_start);
      }
    }
    auto stage_init_distance_computation = clock64();

    int target_point_id = 0;
    
    float p[(DIM + 31) / 32];
#pragma unroll
    for (int i = 0; i < (DIM + 31) / 32; ++i) {
        p[i] = 0;
        if (t_id + 32 * i < DIM) {
            p[i] = d_data[target_point_id * DIM + t_id + i * 32];
        }
    }

    float delta[(DIM + 31) / 32];
    float p_l2[(DIM + 31) / 32];
    float q_l2[(DIM + 31) / 32];
    if constexpr (metric_type == ganns::MetricType::L2) {
#pragma unroll
        for (int i = 0; i < (DIM + 31) / 32; ++i) {
            delta[i] = (p[i] - q[i]) * (p[i] - q[i]);
        }
    } else if (metric_type == ganns::MetricType::IP) {
#pragma unroll
        for (int i = 0 ; i < (DIM + 31) / 32; ++i) {
            delta[i] = p[i] * q[i];
        }
    } else if (metric_type == ganns::MetricType::COS) {
#pragma unroll
        for (int i = 0; i < (DIM + 31) / 32; ++i) {
            delta[i] = p[i] * q[i];
            p_l2[i] = p[i] * p[i];
            q_l2[i] = q[i] * q[i];
        }
    }

    float dist = 0;
    float p_l2_sum = 0;
    float q_l2_sum = 0;
    if constexpr (metric_type == ganns::MetricType::L2) {
#pragma unroll
        for (int i = 0; i < (DIM + 31) / 32; ++i) {
            dist += delta[i];
        }
    } else if constexpr (metric_type == ganns::MetricType::IP) {
#pragma unroll
        for (int i = 0; i < (DIM + 31) / 32; ++i) {
            dist += delta[i];
        }
    } else if constexpr (metric_type == ganns::MetricType::COS) {
        for (int i = 0; i < (DIM + 31) / 32; ++i) {
            dist += delta[i];
            p_l2_sum += p_l2[i];
            q_l2_sum += q_l2[i];
        }
    }

    if constexpr (metric_type == ganns::MetricType::L2) {
        dist += __shfl_down_sync(FULL_MASK, dist, 16);
        dist += __shfl_down_sync(FULL_MASK, dist, 8);
        dist += __shfl_down_sync(FULL_MASK, dist, 4);
        dist += __shfl_down_sync(FULL_MASK, dist, 2);
        dist += __shfl_down_sync(FULL_MASK, dist, 1);
    } else if constexpr (metric_type == ganns::MetricType::IP) {
        dist += __shfl_down_sync(FULL_MASK, dist, 16);
        dist += __shfl_down_sync(FULL_MASK, dist, 8);
        dist += __shfl_down_sync(FULL_MASK, dist, 4);
        dist += __shfl_down_sync(FULL_MASK, dist, 2);
        dist += __shfl_down_sync(FULL_MASK, dist, 1);
    } else if constexpr (metric_type == ganns::MetricType::COS) {
        dist += __shfl_down_sync(FULL_MASK, dist, 16);
        dist += __shfl_down_sync(FULL_MASK, dist, 8);
        dist += __shfl_down_sync(FULL_MASK, dist, 4);
        dist += __shfl_down_sync(FULL_MASK, dist, 2);
        dist += __shfl_down_sync(FULL_MASK, dist, 1);

        p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 16);
        p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 8);
        p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 4);
        p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 2);
        p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 1);

        q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 16);
        q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 8);
        q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 4);
        q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 2);
        q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 1);
    }

    if constexpr (metric_type == ganns::MetricType::L2) {
        dist = sqrt(dist);
    } else if constexpr (metric_type == ganns::MetricType::IP) {
        dist = -dist;
    } else if constexpr (metric_type == ganns::MetricType::COS) {
        p_l2_sum = sqrt(p_l2_sum);
        q_l2_sum = sqrt(q_l2_sum);
        dist = dist / (p_l2_sum * q_l2_sum);
        if (t_id == 0) {
            if(!(dist == dist)){
                dist = 2;
            } else {
                dist = 1 - dist;
            }
        }
    }

    if (t_id == 0) {
        neighbors_array[0].first = dist;
    }
    if constexpr (collect_metrics) {
        if (t_id == 0) {
            atomicAdd(&metrics->stage_init_distance_computation, clock64() - stage_init_distance_computation);
            atomicAdd(&metrics->distance_computation_counter, 1);
        }
    }
   	
    while (flag_all_blocks) {

        if (t_id == 0) {
            flags[first_position_of_flag] = 0;
        }

        int layer_offset = prefix_sum_of_num_array_of_each_layer[crt_layer] * num_of_final_neighbors;
        int offset_within_layer = neighbors_array[first_position_of_flag].second * num_of_final_neighbors;

        auto stage2_start = clock64();

        length_of_compared_list = num_of_candidates;
        if (num_of_final_neighbors < num_of_candidates) {
            length_of_compared_list = num_of_final_neighbors;
        }
        num_of_visited_points_one_batch = num_of_final_neighbors;

        for (int i = 0; i < (num_of_final_neighbors + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;

            if (unrollt_id < num_of_final_neighbors) {
                neighbors_array[num_of_candidates + unrollt_id].second = (d_graph + layer_offset + offset_within_layer)[unrollt_id];
                flags[num_of_candidates + unrollt_id] = 1;
            }
        }

        auto stage2_end = clock64();
        if constexpr (collect_metrics) {
            if(t_id == 0){
                atomicAdd(&metrics->stage_2, stage2_end - stage2_start);
            }
        }
        auto stage3_start = clock64();

        for (int i = 0; i < num_of_final_neighbors; i++) {
            int target_point_id = neighbors_array[num_of_candidates + i].second;
            
            if (target_point_id >= total_num_of_points) {
                neighbors_array[num_of_candidates + i].first = Max;
                continue;
            }
            float p[(DIM + 31) / 32];
#pragma unroll
            for (int k = 0; k < (DIM + 31) / 32; ++k) {
                p[k] = 0;
                if (t_id + 32 * k < DIM) {
                    p[k] = d_data[target_point_id * DIM + t_id + k * 32];
                }
            }

            float delta[(DIM + 31) / 32];
            float p_l2[(DIM + 31) / 32];
            float q_l2[(DIM + 31) / 32];
            if constexpr (metric_type == ganns::MetricType::L2) {
#pragma unroll
                for (int k = 0; k < (DIM + 31) / 32; ++k) {
                    delta[k] = (p[k] - q[k]) * (p[k] - q[k]);
                }
            } else if (metric_type == ganns::MetricType::IP) {
#pragma unroll
                for (int k = 0 ; k < (DIM + 31) / 32; ++k) {
                    delta[k] = p[k] * q[k];
                }
            } else if (metric_type == ganns::MetricType::COS) {
#pragma unroll
                for (int k = 0; k < (DIM + 31) / 32; ++k) {
                    delta[k] = p[k] * q[k];
                    p_l2[k] = p[k] * p[k];
                    q_l2[k] = q[k] * q[k];
                }
            }

            float dist = 0;
            float p_l2_sum = 0;
            float q_l2_sum = 0;
            if constexpr (metric_type == ganns::MetricType::L2) {
#pragma unroll
                for (int k = 0; k < (DIM + 31) / 32; ++k) {
                    dist += delta[k];
                }
            } else if constexpr (metric_type == ganns::MetricType::IP) {
#pragma unroll
                for (int k = 0; k < (DIM + 31) / 32; ++k) {
                    dist += delta[k];
                }
            } else if constexpr (metric_type == ganns::MetricType::COS) {
                for (int k = 0; k < (DIM + 31) / 32; ++k) {
                    dist += delta[k];
                    p_l2_sum += p_l2[k];
                    q_l2_sum += q_l2[k];
                }
            }

            if constexpr (metric_type == ganns::MetricType::L2) {
                dist += __shfl_down_sync(FULL_MASK, dist, 16);
                dist += __shfl_down_sync(FULL_MASK, dist, 8);
                dist += __shfl_down_sync(FULL_MASK, dist, 4);
                dist += __shfl_down_sync(FULL_MASK, dist, 2);
                dist += __shfl_down_sync(FULL_MASK, dist, 1);
            } else if constexpr (metric_type == ganns::MetricType::IP) {
                dist += __shfl_down_sync(FULL_MASK, dist, 16);
                dist += __shfl_down_sync(FULL_MASK, dist, 8);
                dist += __shfl_down_sync(FULL_MASK, dist, 4);
                dist += __shfl_down_sync(FULL_MASK, dist, 2);
                dist += __shfl_down_sync(FULL_MASK, dist, 1);
            } else if constexpr (metric_type == ganns::MetricType::COS) {
                dist += __shfl_down_sync(FULL_MASK, dist, 16);
                dist += __shfl_down_sync(FULL_MASK, dist, 8);
                dist += __shfl_down_sync(FULL_MASK, dist, 4);
                dist += __shfl_down_sync(FULL_MASK, dist, 2);
                dist += __shfl_down_sync(FULL_MASK, dist, 1);

                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 16);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 8);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 4);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 2);
                p_l2_sum += __shfl_down_sync(FULL_MASK, p_l2_sum, 1);

                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 16);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 8);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 4);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 2);
                q_l2_sum += __shfl_down_sync(FULL_MASK, q_l2_sum, 1);
            }

            if constexpr (metric_type == ganns::MetricType::L2) {
                dist = sqrt(dist);
            } else if constexpr (metric_type == ganns::MetricType::IP) {
                dist = -dist;
            } else if constexpr (metric_type == ganns::MetricType::COS) {
                p_l2_sum = sqrt(p_l2_sum);
                q_l2_sum = sqrt(q_l2_sum);
                dist = dist / (p_l2_sum * q_l2_sum);
                if (t_id == 0) {
                    if(!(dist == dist)){
                        dist = 2;
                    } else {
                        dist = 1 - dist;
                    }
                }
            }

                
                if (t_id == 0) {
                    neighbors_array[num_of_candidates+i].first = dist;

                    if constexpr (collect_metrics) {
                        atomicAdd(&metrics->distance_computation_counter, 1);
                    }
                }

        }

        auto stage3_end = clock64();
        if constexpr (collect_metrics) {
            if(t_id == 0){
                atomicAdd(&metrics->stage_3_distance_computation, stage3_end - stage3_start);
            }
        }

        auto stage4_start = clock64();

for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch + size_of_warp - 1) / size_of_warp; temparory_id++) {
    int unrollt_id = t_id + size_of_warp * temparory_id;
    if (unrollt_id < num_of_visited_points_one_batch) {
        float target_distance = neighbors_array[num_of_candidates+unrollt_id].first;
        int flag_of_find = -1;
        int low_end = 0;
        int high_end = num_of_candidates - 1;
        int middle_end;
        while (low_end <= high_end) {
            middle_end = (high_end + low_end) / 2;
            if (target_distance == neighbors_array[middle_end].first) {
                if (middle_end > 0 && neighbors_array[middle_end - 1].first == neighbors_array[middle_end].first) {
                    high_end = middle_end - 1;
                } else {
                    flag_of_find = middle_end;
                    break;
                }
            } else if (target_distance < neighbors_array[middle_end].first) {
                high_end = middle_end - 1;
            } else {
                low_end = middle_end + 1;
            }
        }
        if (flag_of_find != -1) {
            if (neighbors_array[num_of_candidates + unrollt_id].second == neighbors_array[flag_of_find].second) {
                neighbors_array[num_of_candidates + unrollt_id].first = Max;
            } else {
                int position_of_find_element = flag_of_find + 1;

                while (neighbors_array[position_of_find_element].first == neighbors_array[num_of_candidates + unrollt_id].first) {
                    if (neighbors_array[num_of_candidates + unrollt_id].second == neighbors_array[position_of_find_element].second) {
                        neighbors_array[num_of_candidates + unrollt_id].first = Max;
                        break;
                    }
                    position_of_find_element++;
                }
            }
        }
    }
}

        auto stage4_end = clock64();
        if constexpr (collect_metrics) {
            if(t_id == 0){
                // crt_time_breakdown[3] += stage4_end - stage4_start;
                atomicAdd(&metrics->stage_4, stage4_end - stage4_start);
            }
        }

        auto stage5_start = clock64();

step_id = 1;
substep_id = 1;

for (; step_id <= num_of_visited_points_one_batch / 2; step_id *= 2) {
    substep_id = step_id;

    for (; substep_id >= 1; substep_id /= 2) {
        for (int temparory_id = 0; temparory_id < (num_of_visited_points_one_batch/2+size_of_warp-1) / size_of_warp; temparory_id++) {
            int unrollt_id = num_of_candidates + ((t_id + size_of_warp * temparory_id) / substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
            
            if (unrollt_id < num_of_candidates + num_of_visited_points_one_batch) {
                if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                    if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                } else {
                    if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                        temporary_neighbor = neighbors_array[unrollt_id];
                        neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                        neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    }
                }
            }
        }
    }
}
        
        auto stage5_end = clock64();
        if constexpr (collect_metrics) {
            if(t_id == 0){
                // crt_time_breakdown[4] += stage5_end - stage5_start;
                atomicAdd(&metrics->stage_5, stage5_end - stage5_start);
            }
        }
        
        auto stage6_start = clock64();

        
for (int temparory_id = 0; temparory_id < (length_of_compared_list + size_of_warp - 1) / size_of_warp; temparory_id++) {
    int unrollt_id = num_of_candidates - length_of_compared_list + t_id + size_of_warp * temparory_id;
    if (unrollt_id < num_of_candidates) {
        if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + num_of_visited_points_one_batch].first) {
            temporary_neighbor = neighbors_array[unrollt_id];
            neighbors_array[unrollt_id] = neighbors_array[unrollt_id + num_of_visited_points_one_batch];
            neighbors_array[unrollt_id + num_of_visited_points_one_batch] = temporary_neighbor;
            
            temporary_flag = flags[unrollt_id];
            flags[unrollt_id] = flags[unrollt_id + num_of_visited_points_one_batch];
            flags[unrollt_id + num_of_visited_points_one_batch] = temporary_flag;
        }
    }
}

step_id = num_of_candidates / 2;
substep_id = num_of_candidates / 2;
for (; substep_id >= 1; substep_id /= 2) {
    for (int temparory_id = 0; temparory_id < (num_of_candidates / 2 + size_of_warp - 1) / size_of_warp; temparory_id++) {
        int unrollt_id = ((t_id + size_of_warp * temparory_id)/ substep_id) * 2 * substep_id + ((t_id + size_of_warp * temparory_id) & (substep_id - 1));
        if (unrollt_id < num_of_candidates) {
            if (((t_id + size_of_warp * temparory_id) / step_id) % 2 == 0) {
                if (neighbors_array[unrollt_id].first > neighbors_array[unrollt_id + substep_id].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = temporary_neighbor;

                    temporary_flag = flags[unrollt_id];
                    flags[unrollt_id] = flags[unrollt_id + substep_id];
                    flags[unrollt_id + substep_id] = temporary_flag;
                }
            } else {
                if (neighbors_array[unrollt_id].first < neighbors_array[unrollt_id + substep_id].first) {
                    temporary_neighbor = neighbors_array[unrollt_id];
                    neighbors_array[unrollt_id] = neighbors_array[unrollt_id + substep_id];
                    neighbors_array[unrollt_id + substep_id] = temporary_neighbor;
                    
                    temporary_flag = flags[unrollt_id];
                    flags[unrollt_id] = flags[unrollt_id + substep_id];
                    flags[unrollt_id + substep_id] = temporary_flag;
                }
            }
        }
    }
}

        auto stage6_end = clock64();
        if constexpr (collect_metrics) {
            if(t_id == 0){
                // crt_time_breakdown[5] += stage6_end - stage6_start;
                atomicAdd(&metrics->stage_6, stage6_end - stage6_start);
            }
        }
        
        auto stage1_start = clock64();

        for (int i = 0; i < (num_of_explored_points + size_of_warp - 1) / size_of_warp; i++) {
            int unrollt_id = t_id + size_of_warp * i;
            int crt_flag = 0;

            if(unrollt_id < num_of_explored_points){
                crt_flag = flags[unrollt_id];
            }
            first_position_of_flag = __ballot_sync(FULL_MASK, crt_flag);

            if(first_position_of_flag != 0){
                first_position_of_flag = size_of_warp * i + __ffs(first_position_of_flag) - 1;
                break;
            }else if(i == (num_of_explored_points + size_of_warp - 1) / size_of_warp - 1){
                if (crt_layer > 0) {
                    crt_layer--;
                    first_position_of_flag = 0;

                    for (int j = 0; j < (num_of_explored_points + size_of_warp - 1) / size_of_warp; j++) {
                        int unrollt_id2 = t_id + size_of_warp * j;

                        if (unrollt_id2 < num_of_explored_points && neighbors_array[unrollt_id2].first != Max) {
                            flags[unrollt_id2] = 1;
                        }
                    }
                } else {
                    flag_all_blocks = 0;
                }
            }
        }

        auto stage1_end = clock64();
        if constexpr (collect_metrics) {
            if(t_id == 0){
                // crt_time_breakdown[0] += stage1_end - stage1_start;
                atomicAdd(&metrics->stage_1, stage1_end - stage1_start);
            }
        }
    }

    auto stage_final = clock64();
    // printf("num_of_results: %d, (num_of_results + 31) / 32 = %d", num_of_results, (num_of_results + 31) / 32);
    for (int i = 0; i < (num_of_results + size_of_warp - 1) / size_of_warp; i++) {
        int unrollt_id = t_id + size_of_warp * i;
        // printf("unrollt_id=%d\n", unrollt_id);
        if (unrollt_id < num_of_results) {
            crt_result[unrollt_id] = neighbors_array[unrollt_id].second;
            // printf("set result!\n");
        }
    }
    if constexpr (collect_metrics) {
        if (t_id == 0) {
            atomicAdd(&metrics->stage_final, stage_final - stage_init_start);
        }
    }
}
