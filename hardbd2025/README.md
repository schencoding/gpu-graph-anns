# Build and Run

This repository is based on [rapidsai/cuvs](https://github.com/rapidsai/cuvs),
so please refer to the [cuvs build-from-source documentation](https://docs.rapids.ai/api/cuvs/stable/build/#build-from-source) as a starting point.

## Software & Hardware Requirements

Same as [rapidsai/cuvs](https://github.com/rapidsai/cuvs),
but if you want to benchmark the [GANNS](../cpp/bench/ann/src/ganns) algorithm,
please make sure your CUDA version is < 12, see https://github.com/yuyuanhang/GANNS/issues/2

## Compile the Benchmark

When building cuVS from source, the `build.sh` script offers a nice wrapper around the cmake commands to ease the burdens of manually configuring the various available cmake options.
You can run `build.sh -h` to see the available options.

For compiling the benchmark, you can use the following command:
```bash
./build.sh bench-ann
```

## Run Benchmarks

Our benchmarks are based on [cuVS Bench](https://docs.rapids.ai/api/cuvs/stable/cuvs_bench/),
please refer to the [cuvs_bench documentation](https://docs.rapids.ai/api/cuvs/stable/cuvs_bench/) at first.

