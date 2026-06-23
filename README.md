# GraphHET: A Fast Graph Database System on CPU/GPU Platforms

GraphHET is a lightweight graph database system that uses both CPUs and GPUs to accelerate common graph analytics workloads, including shortest paths, PageRank, and community detection.

GraphHET is designed for large graphs with billions of vertices and edges. On the [LDBC Graphalytics benchmark](https://ldbcouncil.org/benchmarks/graphalytics/) and graphs from [SNAP](https://snap.stanford.edu), the project reports performance that is up to **10x faster than [Neo4j](https://neo4j.com) on CPUs** and up to **50x faster than Neo4j on GPUs**.

## Graph Data Structures and Algorithms

GraphHET uses different graph representations for different hardware targets:

- **CPU memory**: [Adjacency lists](https://www.geeksforgeeks.org/adjacency-list-meaning-definition-in-dsa/)
- **GPU memory**: [Compressed Sparse Row (CSR)](https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/) and [GPMA+](https://github.com/desert0616/gpma_demo)

The repository currently includes implementations of six graph algorithms on both CPU and GPU backends:

- Breadth-First Search (BFS)
- Single-Source Shortest Paths (SSSP)
- Weakly Connected Components (WCC)
- PageRank (PR)
- Community Detection with Label Propagation (CDLP)
- Triangle Counting (TC)

Reference pseudocode for the LDBC Graphalytics workloads can be found in the [LDBC Graphalytics benchmark handbook](https://arxiv.org/pdf/2011.15028). The implementations in GraphHET are optimized for parallel execution, so they may differ substantially from the pseudocode.

## Repository Structure

- `include/`: Public headers
- `include/CPU_adj_list/`: CPU adjacency-list data structure and helpers
- `include/CPU_adj_list/CPU_adj_list.hpp`: CPU adjacency-list implementation
- `include/CPU_adj_list/algorithm/`: CPU graph algorithm implementations
- `include/GPU_csr/`: GPU CSR data structure and helpers
- `include/GPU_csr/GPU_csr.hpp`: GPU CSR implementation
- `include/GPU_csr/algorithm/`: GPU graph algorithm implementations for the CSR backend
- `include/GPU_gpma/`: GPU GPMA+ data structure and helpers
- `include/GPU_gpma/GPU_gpma.hpp`: GPU GPMA+ implementation
- `include/GPU_gpma/algorithm/`: GPU graph algorithm implementations for the GPMA+ backend
- `include/LDBC/`: Utilities for running the LDBC Graphalytics benchmark
- `src/`: Source files
- `src/CPU_adj_list/CPU_example.cpp`: CPU example program
- `src/GPU_csr/GPU_csr_example.cu`: GPU example program for the CSR backend
- `src/GPU_gpma/GPU_gpma_example.cu`: GPU example program for the GPMA+ backend
- `src/LDBC/LDBC_CPU_adj_list.cpp`: CPU benchmark entry point
- `src/LDBC/LDBC_GPU_csr.cu`: GPU benchmark entry point for the CSR backend
- `src/LDBC/LDBC_GPU_gpma.cu`: GPU benchmark entry point for the GPMA+ backend
- `LDBC-CPU.sh`: Batch script for CPU benchmark runs
- `LDBC-GPU-csr.sh`: Batch script for GPU CSR benchmark runs
- `LDBC-GPU-gpma.sh`: Batch script for GPU GPMA+ benchmark runs

## Build and Run

The project has been compiled and run successfully on a Linux server with CentOS 7.9 and an NVIDIA RTX A6000 GPU. The verified build environment is:

- `cmake --version`: 3.27.9
- `g++ --version`: 9.4.0
- CUDA Toolkit: 11.8
- NVIDIA Driver: 550.54.14
- GPU: NVIDIA RTX A6000

CUDA 12.4 is known to fail during compilation for this project.

For the server named `170`, the original instructions require two adjustments before compiling:

1. Run `source /opt/rh/devtoolset-11/enable` to switch the compiler toolchain.
2. Replace `cmake` with `cmake3` in the commands below.

Clone or copy the repository to a Linux machine, then build it from the repository root:

```shell
mkdir build
cd build
cmake .. -DBUILD_CPU=ON -DBUILD_GPU_CSR=ON -DBUILD_GPU_GPMA=ON
make
./bin_cpu/CPU_example
./bin_gpu/GPU_example_csr
./bin_gpu/GPU_example_gpma
./bin_cpu/Test_CPU
./bin_gpu/Test_GPU_CSR
./bin_gpu/Test_GPU_GPMA
```

### Build Options

- `-DBUILD_CPU=ON`: build the CPU adjacency-list backend
- `-DBUILD_GPU_CSR=ON`: build the GPU CSR backend
- `-DBUILD_GPU_GPMA=ON`: build the GPU GPMA+ backend

If GPUs are not available, build only the CPU backend:

```shell
cmake .. -DBUILD_CPU=ON -DBUILD_GPU_CSR=OFF -DBUILD_GPU_GPMA=OFF
```

### Executables

- `./bin_cpu/CPU_example`: runs `src/CPU_adj_list/CPU_example.cpp`
- `./bin_gpu/GPU_example_csr`: runs `src/GPU_csr/GPU_csr_example.cu`
- `./bin_gpu/GPU_example_gpma`: runs `src/GPU_gpma/GPU_gpma_example.cu`
- `./bin_cpu/Test_CPU`: runs `src/LDBC/LDBC_CPU_adj_list.cpp`
- `./bin_gpu/Test_GPU_CSR`: runs `src/LDBC/LDBC_GPU_csr.cu`
- `./bin_gpu/Test_GPU_GPMA`: runs `src/LDBC/LDBC_GPU_gpma.cu`

The three example programs can be executed without downloading any dataset. They print algorithm outputs directly to the terminal.

## Running the LDBC Graphalytics Benchmarks

Before running `Test_CPU`, `Test_GPU_CSR`, or `Test_GPU_GPMA`, download the [LDBC Graphalytics datasets](https://repository.surfsara.nl/datasets/cwi/graphalytics). When the benchmark binaries start, they will prompt for the dataset directory and graph name:

```shell
Please input the data directory:
/home/username/data
Please input the graph name:
datagen-7_5-fb
```

After the dataset path and graph name are provided, the program runs the corresponding benchmark workload and prints both the benchmark configuration and the execution time for each graph algorithm.

The repository also includes helper scripts for batch benchmark execution:

- `./LDBC-CPU.sh`
- `./LDBC-GPU-csr.sh`
- `./LDBC-GPU-gpma.sh`

These scripts iterate through a predefined dataset list and redirect benchmark logs to per-dataset output files.
