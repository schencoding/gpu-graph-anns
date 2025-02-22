#pragma once
// #define _GRAPH_QUALITY_ANALYSIS
// #define _CLK_BREAKDOWN
#define METRIC_THREAD_COND() ((threadIdx.x == 0) && (blockIdx.x == 0))
