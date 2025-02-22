#pragma once

//#define __ENABLE_HASH

typedef float data_value_t;

#ifdef __ENABLE_HASH
typedef unsigned int value_t; 
typedef int dist_t; 
#else
typedef float value_t; 
typedef double dist_t; 
#endif
typedef size_t idx_t;
typedef int UINT;


//#define ACC_BATCH_SIZE 4096
#define ACC_BATCH_SIZE 1000000

//for GPU
#define FIXED_DEGREE 31
#define FIXED_DEGREE_SHIFT 5

//for CPU construction
#define SEARCH_DEGREE 15
#define CONSTRUCT_SEARCH_BUDGET 150

enum class SongMetricType {
  L2,
  IP,
  COS
};

template <int PQ_SIZE>
struct BloomFilterConfig;

template <>
struct BloomFilterConfig<10> {
  static constexpr int BLOOM_FILTER_BIT64 = 16;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 4;
  static constexpr int BLOOM_FILTER_NUM_HASH = 6;
};

template <>
struct BloomFilterConfig<20> {
  static constexpr int BLOOM_FILTER_BIT64 = 32;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 5;
  static constexpr int BLOOM_FILTER_NUM_HASH = 6;
};

template <>
struct BloomFilterConfig<30> {
  static constexpr int BLOOM_FILTER_BIT64 = 64;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 6;
  static constexpr int BLOOM_FILTER_NUM_HASH = 7;
};

template <>
struct BloomFilterConfig<40> {
  static constexpr int BLOOM_FILTER_BIT64 = 64;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 6;
  static constexpr int BLOOM_FILTER_NUM_HASH = 6;
};

template <>
struct BloomFilterConfig<50> {
  static constexpr int BLOOM_FILTER_BIT64 = 128;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 7;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<60> {
  static constexpr int BLOOM_FILTER_BIT64 = 128;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 7;
  static constexpr int BLOOM_FILTER_NUM_HASH = 7;
};

template <>
struct BloomFilterConfig<70> {
  static constexpr int BLOOM_FILTER_BIT64 = 128;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 7;
  static constexpr int BLOOM_FILTER_NUM_HASH = 6;
};

template <>
struct BloomFilterConfig<80> {
  static constexpr int BLOOM_FILTER_BIT64 = 128;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 7;
  static constexpr int BLOOM_FILTER_NUM_HASH = 6;
};

template <>
struct BloomFilterConfig<90> {
  static constexpr int BLOOM_FILTER_BIT64 = 256;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 8;
  static constexpr int BLOOM_FILTER_NUM_HASH = 10;
};

template <>
struct BloomFilterConfig<100> {
  static constexpr int BLOOM_FILTER_BIT64 = 256;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 8;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<150> {
  static constexpr int BLOOM_FILTER_BIT64 = 256;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 8;
  static constexpr int BLOOM_FILTER_NUM_HASH = 6;
};

template <>
struct BloomFilterConfig<200> {
  static constexpr int BLOOM_FILTER_BIT64 = 512;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 9;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<400> {
  static constexpr int BLOOM_FILTER_BIT64 = 1024;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 10;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<800> {
  static constexpr int BLOOM_FILTER_BIT64 = 2048;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 11;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<1600> {
  static constexpr int BLOOM_FILTER_BIT64 = 4096;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 12;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<3200> {
  static constexpr int BLOOM_FILTER_BIT64 = 8192;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 13;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};

template <>
struct BloomFilterConfig<6400> {
  static constexpr int BLOOM_FILTER_BIT64 = 16384;
  static constexpr int BLOOM_FILTER_BIT_SHIFT = 14;
  static constexpr int BLOOM_FILTER_NUM_HASH = 9;
};
