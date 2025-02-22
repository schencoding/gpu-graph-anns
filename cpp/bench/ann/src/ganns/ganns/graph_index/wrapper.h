#pragma once

#include <cuda_runtime_api.h>
using namespace std;

class GraphWrapper{

public:
  virtual ~GraphWrapper() {}
	virtual void Dump(string graph_name) = 0;
	virtual void Establishment(int num_of_initial_neighbors, int num_of_candidates) = 0;
	virtual void Load(string graph_path) = 0;
	virtual void SearchTopKonDevice(float* queries, int num_of_topk, int* &results, int num_of_query_points, int num_of_candidates) = 0;
	virtual void DisplayGraphParameters(int num_of_candidates) = 0;
	virtual void DisplaySearchParameters(int num_of_topk, int num_of_candidates) = 0;
  virtual void PrepareDeviceData() = 0;
  void FreeResults(int* results)
  {
    cudaFreeHost(results);
  }
private:
	
};
