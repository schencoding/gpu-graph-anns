#pragma once

#include <iostream>
#include "kernel_pair.h"
#include"data.h"
#include<vector>
#include"config.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda_runtime.h>
#include"cublas_v2.h"
#include<chrono>

#include"smmh2.h"
#include"bin_heap.h"
#include"cuckoofilter.h"
#include"bloomfilter.h"
#include"blocked_bloomfilter.h"
#include"vanilla_list.h"
#include"fixhash.h"

#ifndef __ENABLE_BLOCKED_BLOOM_FILTER
	#define BlockedBloomFilter BloomFilter
#endif

#define FULL_MASK 0xffffffff
#define N_THREAD_IN_WARP 32
#define N_MULTIQUERY 1
#define CRITICAL_STEP (N_THREAD_IN_WARP/N_MULTIQUERY)
#define N_MULTIPROBE 1
// #define FINISH_CNT 1

// #define __ENABLE_MEASURE

struct Measure {
  unsigned long long iteration_counter = 0;
  unsigned long long stage_distance_computation = 0;
  unsigned long long stage_final = 0;
  unsigned long long distance_computation_counter = 0;

  unsigned long long clock_threadidx_x_counter = 0;
  unsigned long long clk_new_local_memory = 0;

  unsigned long long clk_smmh2_pop_min = 0;
  unsigned long long smmh2_pop_min_counter = 0;

  unsigned long long clk_smmh2_pop_max = 0;
  unsigned long long smmh2_pop_max_counter = 0;

  unsigned long long clk_smmh2_push = 0;
  unsigned long long smmh2_push_counter = 0;

  unsigned long long clk_binheap_push = 0;
  unsigned long long binheap_push_counter = 0;

  unsigned long long clk_binheap_pop = 0;
  unsigned long long binheap_pop_counter = 0;

  unsigned long long clk_pbf_add = 0;
  unsigned long long pbf_add_counter = 0;

  unsigned long long clk_pbf_test = 0;
  unsigned long long pbf_test_counter = 0;

  unsigned long long clk_pbf_del = 0;
  unsigned long long pbf_del_counter = 0;

  unsigned long long clk_finished_array_access = 0;

  unsigned long long clk_init_distance_computation = 0;

  unsigned long long metric_queries = 0;

  static Measure& get_instance(){
    static Measure instance;
    return instance;
  }

  void print_metrics()
  {
    printf("iteration_counter: %llu\n", iteration_counter);
    printf("stage_distance_computation: %llu\n", stage_distance_computation);
    printf("stage_final: %llu\n", stage_final);
    printf("distance_computation_counter: %llu\n", distance_computation_counter);
    printf("clock_threadidx_x_counter: %llu\n", clock_threadidx_x_counter);
    printf("clk_new_local_memory: %llu\n", clk_new_local_memory);
    printf("clk_smmh2_pop_min: %llu\n", clk_smmh2_pop_min);
    printf("smmh2_pop_min_counter: %llu\n", smmh2_pop_min_counter);
    printf("clk_smmh2_pop_max: %llu\n", clk_smmh2_pop_max);
    printf("smmh2_pop_max_counter: %llu\n", smmh2_pop_max_counter);
    printf("clk_smmh2_push: %llu\n", clk_smmh2_push);
    printf("smmh2_push_counter: %llu\n", smmh2_push_counter);
    printf("clk_binheap_push: %llu\n", clk_binheap_push);
    printf("binheap_push_counter: %llu\n", binheap_push_counter);
    printf("clk_binheap_pop: %llu\n", clk_binheap_pop);
    printf("binheap_pop_counter: %llu\n", binheap_pop_counter);
    printf("clk_pbf_add: %llu\n", clk_pbf_add);
    printf("pbf_add_counter: %llu\n", pbf_add_counter);
    printf("clk_pbf_test: %llu\n", clk_pbf_test);
    printf("pbf_test_counter: %llu\n", pbf_test_counter);
    printf("clk_pbf_del: %llu\n", clk_pbf_del);
    printf("pbf_del_counter: %llu\n", pbf_del_counter);
    printf("clk_finished_array_access: %llu\n", clk_finished_array_access);
    printf("clk_init_distance_computation: %llu\n", clk_init_distance_computation);
    printf("metric_queries: %llu\n", metric_queries);
  }
  void reset()
  {
    iteration_counter = 0;
    stage_distance_computation = 0;
    stage_final = 0;
    distance_computation_counter = 0;
    clock_threadidx_x_counter = 0;
    clk_new_local_memory = 0;
    clk_smmh2_pop_min = 0;
    smmh2_pop_min_counter = 0;
    clk_smmh2_pop_max = 0;
    smmh2_pop_max_counter = 0;
    clk_smmh2_push = 0;
    smmh2_push_counter = 0;
    clk_binheap_push = 0;
    binheap_push_counter = 0;
    clk_binheap_pop = 0;
    binheap_pop_counter = 0;
    clk_pbf_add = 0;
    pbf_add_counter = 0;
    clk_pbf_test = 0;
    pbf_test_counter = 0;
    clk_pbf_del = 0;
    pbf_del_counter = 0;
    clk_finished_array_access = 0;
    clk_init_distance_computation = 0;
    metric_queries = 0;
  }

  void accumulate(const Measure& other)
  {
    iteration_counter += other.iteration_counter;
    stage_distance_computation += other.stage_distance_computation;
    stage_final += other.stage_final;
    distance_computation_counter += other.distance_computation_counter;
    clock_threadidx_x_counter += other.clock_threadidx_x_counter;
    clk_new_local_memory += other.clk_new_local_memory;
    clk_smmh2_pop_min += other.clk_smmh2_pop_min;
    smmh2_pop_min_counter += other.smmh2_pop_min_counter;
    clk_smmh2_pop_max += other.clk_smmh2_pop_max;
    smmh2_pop_max_counter += other.smmh2_pop_max_counter;
    clk_smmh2_push += other.clk_smmh2_push;
    smmh2_push_counter += other.smmh2_push_counter;
    clk_binheap_push += other.clk_binheap_push;
    binheap_push_counter += other.binheap_push_counter;
    clk_binheap_pop += other.clk_binheap_pop;
    binheap_pop_counter += other.binheap_pop_counter;
    clk_pbf_add += other.clk_pbf_add;
    pbf_add_counter += other.pbf_add_counter;
    clk_pbf_test += other.clk_pbf_test;
    pbf_test_counter += other.pbf_test_counter;
    clk_pbf_del += other.clk_pbf_del;
    pbf_del_counter += other.pbf_del_counter;
    clk_finished_array_access += other.clk_finished_array_access;
    clk_init_distance_computation += other.clk_init_distance_computation;
    metric_queries += other.metric_queries;
  }
};

enum class VisitedTableType {
  kBloomFilter,
  kCuckooFilter,
  kHashTable, // hashtablenosel
  kHashTableSel,
  kHashTableSelDel,
};

constexpr bool _enable_fixhash(VisitedTableType visited_table_type) {
  return visited_table_type == VisitedTableType::kHashTable || visited_table_type == VisitedTableType::kHashTableSel || visited_table_type == VisitedTableType::kHashTableSelDel;
}
constexpr bool _define_disable_select_insert(VisitedTableType visited_table_type) {
  return visited_table_type == VisitedTableType::kHashTable;
}
constexpr bool _not_define_disable_select_insert(VisitedTableType visited_table_type) {
  return !_define_disable_select_insert(visited_table_type);
}
constexpr bool _enable_visited_del(VisitedTableType visited_table_type) {
  return visited_table_type == VisitedTableType::kHashTableSelDel || visited_table_type == VisitedTableType::kCuckooFilter;
}
constexpr bool _enable_cuckoo_filter(VisitedTableType visited_table_type) {
  return visited_table_type == VisitedTableType::kCuckooFilter;
}

template<VisitedTableType visited_table_type>
struct VisitedTableTypeTrait ;

template<>
struct VisitedTableTypeTrait<VisitedTableType::kBloomFilter> {
  template<int TOPK>
    static auto __device__ new_pbf() {
      constexpr auto BLOOM_FILTER_BIT64 = BloomFilterConfig<TOPK>::BLOOM_FILTER_BIT64;
      constexpr auto BLOOM_FILTER_BIT_SHIFT = BloomFilterConfig<TOPK>::BLOOM_FILTER_BIT_SHIFT;
      constexpr auto BLOOM_FILTER_NUM_HASH = BloomFilterConfig<TOPK>::BLOOM_FILTER_NUM_HASH;
      BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>* pbf = new BlockedBloomFilter<BLOOM_FILTER_BIT64,BLOOM_FILTER_BIT_SHIFT,BLOOM_FILTER_NUM_HASH>();
      return pbf;
    }
};

// #-D__ENABLE_CUCKOO_FILTER -D__ENABLE_VISITED_DEL #cuckoofilter
template<>
struct VisitedTableTypeTrait<VisitedTableType::kCuckooFilter> {
  template<int TOPK>
    static __device__ auto new_pbf() {
      constexpr auto CUCKOO_CAPACITY = BloomFilterConfig<TOPK>::BLOOM_FILTER_BIT64 * 2;
      CuckooFilter<CUCKOO_CAPACITY>* pbf = new CuckooFilter<CUCKOO_CAPACITY>();
      return pbf;
    }
};

// #-D__ENABLE_FIXHASH -D__DISABLE_SELECT_INSERT # hashtablenosel
template<>
struct VisitedTableTypeTrait<VisitedTableType::kHashTable> {
  template<int TOPK>
    static __device__ auto new_pbf() {
      static_assert(_define_disable_select_insert(VisitedTableType::kHashTable), "disable select insert");
      constexpr auto HASH_TABLE_CAPACITY = TOPK*4*16 + 500;
      auto* pbf = new FixHash<int,HASH_TABLE_CAPACITY, _enable_visited_del(VisitedTableType::kHashTable)>();
      return pbf;
    }
};

// #-D__ENABLE_FIXHASH #hashtable
template<>
struct VisitedTableTypeTrait<VisitedTableType::kHashTableSel> {
  template<int TOPK>
    static __device__ auto new_pbf() {
      static_assert(!_enable_visited_del(VisitedTableType::kHashTableSel), "enable visited del");
      constexpr auto HASH_TABLE_CAPACITY = TOPK*4*16;
      auto* pbf = new FixHash<int,HASH_TABLE_CAPACITY, _enable_visited_del(VisitedTableType::kHashTableSel)>();
      return pbf;
    }
};

// #-D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL.  # hashtabledel
template<>
struct VisitedTableTypeTrait<VisitedTableType::kHashTableSelDel> {
  template<int TOPK>
    static __device__ auto new_pbf() {
      static_assert(_enable_visited_del(VisitedTableType::kHashTableSelDel), "enable visited del");
      constexpr auto HASH_TABLE_CAPACITY = TOPK*4*2;
      auto * pbf = new FixHash<int,HASH_TABLE_CAPACITY, _enable_visited_del(VisitedTableType::kHashTableSelDel)>();
      return pbf;
    }
};

template<SongMetricType metric_type, int DIM, int TOPK, VisitedTableType visited_table_type>
__global__
void warp_independent_search_kernel(value_t* d_data,value_t* d_query,idx_t* d_result,idx_t* d_graph,int num_query,int vertex_offset_shift, size_t finish_cnt
#ifdef __ENABLE_MEASURE
,Measure* measure
#endif
){
#ifdef __ENABLE_MEASURE
  if (threadIdx.x == 0) {
    atomicAdd(&measure->clock_threadidx_x_counter,1);
  }
  auto clk_new_local_memory_start = clock64();
#endif
  const int QUEUE_SIZE = TOPK;
  int bid = blockIdx.x * N_MULTIQUERY;
  const int step = N_THREAD_IN_WARP;
  int tid = threadIdx.x;
  int cid = tid / CRITICAL_STEP;
  int subtid = tid % CRITICAL_STEP;


  decltype(VisitedTableTypeTrait<visited_table_type>::template new_pbf<TOPK>()) pbf = nullptr;

  KernelPair<dist_t,idx_t>* q;
  KernelPair<dist_t,idx_t>* topk;
	value_t* dist_list;
	if(subtid == 0){
		dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE];
		q= new KernelPair<dist_t,idx_t>[QUEUE_SIZE + 2];
		topk = new KernelPair<dist_t,idx_t>[TOPK + 1];
    pbf = VisitedTableTypeTrait<visited_table_type>::template new_pbf<TOPK>();
	}
    __shared__ int heap_size[N_MULTIQUERY];
	int topk_heap_size;

    __shared__ value_t query_point[N_MULTIQUERY][DIM];

	__shared__ int finished[N_MULTIQUERY];
	__shared__ idx_t index_list[N_MULTIQUERY][FIXED_DEGREE * N_MULTIPROBE];
	__shared__ char index_list_len[N_MULTIQUERY];
	value_t start_distance;
	__syncthreads();

#ifdef __ENABLE_MEASURE
  auto clk_new_local_memory_end = clock64();
  if(threadIdx.x == 0) {
    atomicAdd(&measure->clk_new_local_memory,clk_new_local_memory_end- clk_new_local_memory_start);
  }
  auto clk_init_distance_computation_start = clock64();
#endif

	value_t tmp[N_MULTIQUERY];
  // if constexpr (metric_type == SongMetricType::COS) {
	  value_t tmp_data_len[N_MULTIQUERY];
  // }
	for(int j = 0;j < N_MULTIQUERY;++j){
		tmp[j] = 0;
    if constexpr (metric_type == SongMetricType::COS) {
		  tmp_data_len[j] = 0;
		}
		for(int i = tid;i < DIM;i += step){
			query_point[j][i] = d_query[(bid + j) * DIM + i];
      if constexpr (metric_type == SongMetricType::L2) {
			  tmp[j] += (query_point[j][i] - d_data[i]) * (query_point[j][i] - d_data[i]); 
      } else if constexpr (metric_type == SongMetricType::IP) {
			  tmp[j] += query_point[j][i] * d_data[i]; 
      } else if constexpr (metric_type == SongMetricType::COS) {
        //negative cosine
        tmp[j] += query_point[j][i] * d_data[i]; 
        tmp_data_len[j] += d_data[i] * d_data[i];
      }
		}
		for (int offset = 16; offset > 0; offset /= 2){
      if constexpr (metric_type == SongMetricType::L2) {
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
      } else if constexpr (metric_type == SongMetricType::IP) {
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
      } else if constexpr (metric_type == SongMetricType::COS) {
				//negative cosine
				tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
				tmp_data_len[j] += __shfl_xor_sync(FULL_MASK, tmp_data_len[j], offset);
      }
		}
	}
	if(subtid == 0){
    if constexpr (metric_type == SongMetricType::L2) {
		  start_distance = tmp[cid];
    } else if constexpr (metric_type == SongMetricType::IP) {
		  start_distance = -tmp[cid];
    } else if constexpr (metric_type == SongMetricType::COS) {
      //negative cosine
          int sign = tmp[cid] < 0 ? 1 : -1;
      if(tmp_data_len[cid] != 0)
        start_distance = sign * tmp[cid] * tmp[cid] / tmp_data_len[cid];
      else
        start_distance = 0;
    }
	}
	__syncthreads();
#ifdef __ENABLE_MEASURE
  auto clk_init_distance_computation_end = clock64();
  if(threadIdx.x == 0) {
    atomicAdd(&measure->clk_init_distance_computation,clk_init_distance_computation_end - clk_init_distance_computation_start);
    atomicAdd(&measure->distance_computation_counter,1);
  }
#endif

	if(subtid == 0){
#ifdef __ENABLE_MEASURE
    auto clk_smmh2_push_start = clock64();
#endif
    	heap_size[cid] = 1;
		topk_heap_size = 0;
		finished[cid] = false;
		dist_t d = start_distance;
		KernelPair<dist_t,idx_t> kp;
		kp.first = d;
		kp.second = 0;
		smmh2::insert(q,heap_size[cid],kp);
#ifdef __ENABLE_MEASURE
    auto clk_smmh2_push_end = clock64();
    if(threadIdx.x == 0) {
      atomicAdd(&measure->clk_smmh2_push,clk_smmh2_push_end - clk_smmh2_push_start);
      atomicAdd(&measure->smmh2_push_counter,1);
    }
    auto clk_pbf_add_start = clock64();
#endif
		pbf->add(0);
#ifdef __ENABLE_MEASURE
    auto clk_pbf_add_end = clock64();
    if(threadIdx.x == 0) {
      atomicAdd(&measure->clk_pbf_add,clk_pbf_add_end - clk_pbf_add_start);
      atomicAdd(&measure->pbf_add_counter,1);
    }
#endif
	}
	__syncthreads();

  while(heap_size[cid] > 1){
#ifdef __ENABLE_MEASURE
    if (threadIdx.x == 0) {
      atomicAdd(&measure->iteration_counter,1);
    }
#endif
		index_list_len[cid] = 0;
		int current_heap_elements = heap_size[cid] - 1;
		for(int k = 0;k < N_MULTIPROBE && k < current_heap_elements;++k){
			KernelPair<dist_t,idx_t> now;
			if(subtid == 0){
#ifdef __ENABLE_MEASURE
        auto clk_smmh2_pop_min_start = clock64();
#endif
				now = smmh2::pop_min(q,heap_size[cid]);
#ifdef __ENABLE_MEASURE
        auto clk_smmh2_pop_min_end = clock64();
        if (threadIdx.x == 0) {
          atomicAdd(&measure->clk_smmh2_pop_min,clk_smmh2_pop_min_end - clk_smmh2_pop_min_start);
          atomicAdd(&measure->smmh2_pop_min_counter,1);
        }
#endif
        if constexpr (_enable_visited_del(visited_table_type)) {
#ifdef __ENABLE_MEASURE
          auto clk_pbf_del_start = clock64();
#endif
				  pbf->del(now.second);
#ifdef __ENABLE_MEASURE
          auto clk_pbf_del_end = clock64();
          if (threadIdx.x == 0) {
            atomicAdd(&measure->clk_pbf_del,clk_pbf_del_end - clk_pbf_del_start);
            atomicAdd(&measure->pbf_del_counter,1);
          }
#endif
        }
#ifdef __ENABLE_MEASURE
        auto clk_finished_array_access_start = clock64();
#endif
				if(k == 0 && topk_heap_size == TOPK && (topk[0].first <= now.first)){
					++finished[cid];
				}
#ifdef __ENABLE_MEASURE
        auto clk_finished_array_access_end = clock64();
        if (threadIdx.x == 0) {
          atomicAdd(&measure->clk_finished_array_access,clk_finished_array_access_end - clk_finished_array_access_start);
        }
#endif
			}
			__syncthreads();
			if(finished[cid] >= finish_cnt)
				break;
			if(subtid == 0){
#ifdef __ENABLE_MEASURE
        auto clk_binheap_push_start = clock64();
#endif
				topk[topk_heap_size++] = now;
				push_heap(topk,topk + topk_heap_size);
#ifdef __ENABLE_MEASURE
        auto clk_binheap_push_end = clock64();
        if (threadIdx.x == 0) {
          atomicAdd(&measure->clk_binheap_push,clk_binheap_push_end - clk_binheap_push_start);
          atomicAdd(&measure->binheap_push_counter,1);
        }
#endif
        if constexpr (_enable_visited_del(visited_table_type)) {
#ifdef __ENABLE_MEASURE
          auto clk_pbf_add_start = clock64();
#endif
				  pbf->add(now.second);
#ifdef __ENABLE_MEASURE
          auto clk_pbf_add_end = clock64();
          if (threadIdx.x == 0) {
            atomicAdd(&measure->clk_pbf_add,clk_pbf_add_end - clk_pbf_add_start);
            atomicAdd(&measure->pbf_add_counter,1);
          }
#endif
        }
				if(topk_heap_size > TOPK){
          if constexpr (_enable_visited_del(visited_table_type)) {
#ifdef __ENABLE_MEASURE
            auto clk_pbf_del_start = clock64();
#endif
					  pbf->del(topk[0].second);
#ifdef __ENABLE_MEASURE
            auto clk_pbf_del_end = clock64();
            if (threadIdx.x == 0) {
              atomicAdd(&measure->clk_pbf_del,clk_pbf_del_end - clk_pbf_del_start);
              atomicAdd(&measure->pbf_del_counter,1);
            }
#endif
          }
#ifdef __ENABLE_MEASURE
          auto clk_binheap_pop_start = clock64();
#endif
					pop_heap(topk,topk + topk_heap_size);
					--topk_heap_size;
#ifdef __ENABLE_MEASURE
          auto clk_binheap_pop_end = clock64();
          if (threadIdx.x == 0) {
            atomicAdd(&measure->clk_binheap_pop,clk_binheap_pop_end - clk_binheap_pop_start);
            atomicAdd(&measure->binheap_pop_counter,1);
          }
#endif
				}
				auto offset = now.second << vertex_offset_shift;
				int degree = d_graph[offset];
				for(int i = 1;i <= degree;++i){
					auto idx = d_graph[offset + i];
					if(subtid == 0){
#ifdef __ENABLE_MEASURE
            auto clk_pbf_test_start = clock64();
#endif
            auto pbf_test_result = pbf->test(idx);
#ifdef __ENABLE_MEASURE
            auto clk_pbf_test_end = clock64();
            if (threadIdx.x == 0) {
              atomicAdd(&measure->clk_pbf_test,clk_pbf_test_end - clk_pbf_test_start);
              atomicAdd(&measure->pbf_test_counter,1);
            }
#endif
						if(pbf_test_result){
							continue;
						}
            if constexpr (_define_disable_select_insert(visited_table_type)) {
#ifdef __ENABLE_MEASURE
              auto clk_pbf_add_start = clock64();
#endif
              pbf->add(idx);
#ifdef __ENABLE_MEASURE
              auto clk_pbf_add_end = clock64();
              if (threadIdx.x == 0) {
                atomicAdd(&measure->clk_pbf_add,clk_pbf_add_end - clk_pbf_add_start);
                atomicAdd(&measure->pbf_add_counter,1);
              }
#endif
            }
						index_list[cid][index_list_len[cid]++] = idx;
					}
				}
			}
		}
		if(finished[cid] >= finish_cnt)
			break;
		__syncthreads();

#ifdef __ENABLE_MEASURE
    // NOTE(jiangyinzuo): distance computation
		if(threadIdx.x == 0) {
      atomicAdd(&measure->distance_computation_counter, index_list_len[cid]);
    }
		auto stage2_start = clock64();
#endif
		for(int nq = 0;nq < N_MULTIQUERY;++nq){
			for(int i = 0;i < index_list_len[nq];++i){
				//TODO: replace this atomic with reduction in CUB
				value_t tmp = 0;
				// #ifdef __USE_COS_DIST
				value_t tmp_data_len = 0;
				// #endif
				for(int j = tid;j < DIM;j += step){
          if constexpr (metric_type == SongMetricType::L2) {
					tmp += (query_point[nq][j] - d_data[index_list[nq][i] * DIM + j]) * (query_point[nq][j] - d_data[index_list[nq][i] * DIM + j]); 
          } else if constexpr (metric_type == SongMetricType::IP) {
					tmp += query_point[nq][j] * d_data[index_list[nq][i] * DIM + j]; 
          } else if constexpr (metric_type == SongMetricType::COS) {
            //negative cosine
            tmp += query_point[nq][j] * d_data[index_list[nq][i] * DIM + j]; 
            tmp_data_len += d_data[index_list[nq][i] * DIM + j] * d_data[index_list[nq][i] * DIM + j]; 
          }
				}
				for (int offset = 16; offset > 0; offset /= 2){
          if constexpr (metric_type == SongMetricType::L2) {
					  tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
          } else if constexpr (metric_type == SongMetricType::IP) {
					  tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
          } else if constexpr (metric_type == SongMetricType::COS) {
            //negative cosine
            tmp += __shfl_xor_sync(FULL_MASK, tmp, offset);
            tmp_data_len += __shfl_xor_sync(FULL_MASK, tmp_data_len, offset);
          }
				}
				if(tid == nq * CRITICAL_STEP){
          if constexpr (metric_type == SongMetricType::L2) {
					  dist_list[i] = tmp;
          } else if constexpr (metric_type == SongMetricType::IP) {
					  dist_list[i] = -tmp;
          } else if constexpr (metric_type == SongMetricType::COS) {
            //negative cosine
            int sign = tmp < 0 ? 1 : -1;
            if(tmp_data_len != 0)
              dist_list[i] = sign * tmp * tmp / tmp_data_len;
            else
              dist_list[i] = 0;
          }
				}
			}
		}

		__syncthreads();
#ifdef __ENABLE_MEASURE
		auto stage2_end = clock64();
		if(threadIdx.x == 0)
			atomicAdd(&measure->stage_distance_computation,stage2_end - stage2_start);	
		// auto stage3_start = clock64();
#endif

		if(subtid == 0){
			for(int i = 0;i < index_list_len[cid];++i){
#ifdef __ENABLE_MEASURE
        auto clk_smmh2_push_start = clock64();
#endif
				dist_t d = dist_list[i];
				KernelPair<dist_t,idx_t> kp;
				kp.first = d;
				kp.second = index_list[cid][i];

				if(heap_size[cid] >= QUEUE_SIZE + 1 && q[2].first < kp.first){
#ifdef __ENABLE_MEASURE
        auto clk_smmh2_push_end = clock64();
        if (threadIdx.x == 0) {
          // only add time, do not add counter
          atomicAdd(&measure->clk_smmh2_push,clk_smmh2_push_end - clk_smmh2_push_start);
        }
#endif
					continue;
				}
#ifdef __ENABLE_MULTIPROBE_DOUBLE_CHECK
				if(pbf->test(kp.second))
					continue;
#endif
				smmh2::insert(q,heap_size[cid],kp);
#ifdef __ENABLE_MEASURE
        auto clk_smmh2_push_end = clock64();
        if (threadIdx.x == 0) {
          atomicAdd(&measure->clk_smmh2_push,clk_smmh2_push_end - clk_smmh2_push_start);
          atomicAdd(&measure->smmh2_push_counter,1);
        }
#endif
        if constexpr (_not_define_disable_select_insert(visited_table_type)) {
#ifdef __ENABLE_MEASURE
          auto clk_pbf_add_start = clock64();
#endif
				  pbf->add(kp.second);
#ifdef __ENABLE_MEASURE
          auto clk_pbf_add_end = clock64();
          if (threadIdx.x == 0) {
            atomicAdd(&measure->clk_pbf_add,clk_pbf_add_end - clk_pbf_add_start);
            atomicAdd(&measure->pbf_add_counter,1);
          }
#endif
        }
				if(heap_size[cid] >= QUEUE_SIZE + 2){
          if constexpr (_enable_visited_del(visited_table_type)) {
#ifdef __ENABLE_MEASURE
            auto clk_pbf_del_start = clock64();
#endif
					  pbf->del(q[2].second);
#ifdef __ENABLE_MEASURE
            auto clk_pbf_del_end = clock64();
            if (threadIdx.x == 0) {
              atomicAdd(&measure->clk_pbf_del,clk_pbf_del_end - clk_pbf_del_start);
              atomicAdd(&measure->pbf_del_counter,1);
            }
#endif
          }
#ifdef __ENABLE_MEASURE
          auto clk_smmh2_pop_max_start = clock64();
#endif
					smmh2::pop_max(q,heap_size[cid]);
#ifdef __ENABLE_MEASURE
          auto clk_smmh2_pop_max_end = clock64();
          if (threadIdx.x == 0) {
            atomicAdd(&measure->clk_smmh2_pop_max,clk_smmh2_pop_max_end - clk_smmh2_pop_max_start);
            atomicAdd(&measure->smmh2_pop_max_counter,1);
          }
#endif
				}
			}
		}
		__syncthreads();
#ifdef __ENABLE_MEASURE
		// auto stage3_end = clock64();
		// if(tid == 0)
		// 	atomicAdd(&measure->stage3,stage3_end - stage3_start);	
#endif
  }

#ifdef __ENABLE_MEASURE
	auto stage_final_start= clock64();
#endif
	if(subtid == 0){
		for(int i = 0;i < TOPK;++i){
			auto now = pop_heap(topk,topk + topk_heap_size - i);
			d_result[(bid + cid) * TOPK + TOPK - 1 - i] = now.second;
		}
		delete[] q;
		delete[] topk;
    	delete pbf;
    	delete[] dist_list;
	}
#ifdef __ENABLE_MEASURE
    auto stage_final_end = clock64();
    if(tid == 0)
      atomicAdd(&measure->stage_final,stage_final_end - stage_final_start);
#endif
}

class WarpAStarAccelerator{
private:

public:
  // TOPK: priority queue size
 template <SongMetricType metric_type, int DIM, int TOPK, VisitedTableType visited_table_type>
 static void astar_multi_start_search_batch(
   const std::vector<std::vector<std::pair<int, value_t>>>& queries,
   int k,
   std::vector<std::vector<idx_t>>& results,
   value_t* d_data,
   idx_t* d_graph,
   int vertex_offset_shift,
   int num,
   int dim,
   size_t finish_cnt)
 {
   value_t* d_query;
   idx_t* d_result;

#ifdef __ENABLE_MEASURE
		Measure* d_measure;
		Measure h_measure;
		cudaMalloc(&d_measure,sizeof(Measure));
		cudaMemcpy(d_measure,&h_measure,sizeof(Measure),cudaMemcpyHostToDevice);
		auto time_begin = std::chrono::steady_clock::now();
#endif

		std::unique_ptr<value_t[]> h_query = std::unique_ptr<value_t[]>(new value_t[queries.size() * dim]);
		memset(h_query.get(),0,sizeof(value_t) * queries.size() * dim);
		for(size_t i = 0;i < queries.size();++i){
			for(auto p : queries[i]){
				*(h_query.get() + i * dim + p.first) = p.second;
			}
		}
		std::unique_ptr<idx_t[]> h_result = std::unique_ptr<idx_t[]>(new idx_t[queries.size() * TOPK]);

		cudaMalloc(&d_query,sizeof(value_t) * queries.size() * dim);
		cudaMalloc(&d_result,sizeof(idx_t) * queries.size() * TOPK);
		
		cudaMemcpy(d_query,h_query.get(),sizeof(value_t) * queries.size() * dim,cudaMemcpyHostToDevice);

#ifdef __ENABLE_MEASURE
		std::chrono::steady_clock::time_point mem_transfer = std::chrono::steady_clock::now();
		fprintf(stderr,"mem transfer %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(mem_transfer - time_begin).count());
		std::chrono::steady_clock::time_point kernel_begin = std::chrono::steady_clock::now();
#endif

		warp_independent_search_kernel<metric_type, DIM, TOPK, visited_table_type><<<queries.size()/N_MULTIQUERY,32>>>(d_data,d_query,d_result,d_graph,queries.size(),vertex_offset_shift, finish_cnt
#ifdef __ENABLE_MEASURE
, d_measure
#endif
		);

#ifdef __ENABLE_MEASURE
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "error in kernel: %s\n", cudaGetErrorString(err));
    }
		cudaDeviceSynchronize();
		std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();
		fprintf(stderr,"kernel takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count());
		std::chrono::steady_clock::time_point back_begin = std::chrono::steady_clock::now();
#endif
		cudaMemcpy(h_result.get(),d_result,sizeof(idx_t) * queries.size() * TOPK,cudaMemcpyDeviceToHost);

#ifdef __ENABLE_MEASURE
		std::chrono::steady_clock::time_point back_end = std::chrono::steady_clock::now();
		fprintf(stderr,"transfer back result takes %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(back_end - back_begin).count());

		cudaMemcpy(&h_measure,d_measure,sizeof(Measure),cudaMemcpyDeviceToHost);
		// auto stage_sum = h_measure.stage_init + h_measure.stage1 + h_measure.stage_distance_computation + h_measure.stage3 + h_measure.stage_final;
		// fprintf(stderr,"stages percentage %.2f %.2f %.2f %.2f %.2f\n",
  //       h_measure.stage_init * 100.0 / stage_sum,
  //       h_measure.stage1 * 100.0 / stage_sum,
  //       h_measure.stage_distance_computation * 100.0 / stage_sum,
  //       h_measure.stage3 * 100.0 / stage_sum,
  //       h_measure.stage_final * 100.0 / stage_sum);

    // accumulate
    h_measure.metric_queries = queries.size();
    Measure::get_instance().accumulate(h_measure);
#endif
		results.clear();
		for(size_t i = 0;i < queries.size();++i){
			std::vector<idx_t> v(TOPK);
			for(int j = 0;j < TOPK;++j)
				v[j] = h_result[i * TOPK+ j];
			results.push_back(v);
		}
#ifdef __ENABLE_MEASURE
		// std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		// fprintf(stderr,"using %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
		//printf("using %ld microseconds\n",std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
#endif
		cudaFree(d_query);
		cudaFree(d_result);
 }
};

