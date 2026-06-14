#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <GPU_gpma/GPU_gpma.hpp>

#include <GPU_gpma/algorithm/GPU_BFS_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_WCC_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_SSSP_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_PR_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_CDLP_gpma.cuh>

namespace {

void configure_stdio() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
}

graph_structure<double> build_sample_graph() {
    graph_structure<double> graph;
    graph.add_vertice("one");
    graph.add_vertice("two");
    graph.add_vertice("three");
    graph.add_vertice("four");
    graph.add_vertice("five");
    graph.add_vertice("R");

    graph.add_edge("one", "two", 0.8);
    graph.add_edge("two", "three", 1);
    graph.add_edge("two", "R", 1);
    graph.add_edge("two", "four", 0.1);
    graph.add_edge("R", "three", 1);
    graph.add_edge("one", "three", 1);
    graph.add_edge("one", "four", 1);
    graph.add_edge("four", "three", 1);
    graph.add_edge("four", "five", 1);

    graph.remove_vertice("R");
    graph.add_vertice("six");
    graph.remove_edge("two", "four");
    graph.add_edge("one", "six", 1);

    return graph;
}

template <typename LeftType, typename RightType>
void print_named_pairs(const std::string& title,
        const std::vector<std::pair<LeftType, RightType>>& results) {
    std::cout << title << std::endl;
    for (const auto& result : results) {
        std::cout << result.first << " " << result.second << std::endl;
    }
}

void build_gpma_from_graph(graph_structure<double>& graph, GPMA& gpma_graph_out, GPMA& gpma_graph_in) {
    const int num_nodes = graph.size();
    std::vector<KEY_TYPE> cpu_keys_out;
    std::vector<KEY_TYPE> cpu_keys_in;
    std::vector<VALUE_TYPE> cpu_values_out;
    std::vector<VALUE_TYPE> cpu_values_in;
    std::vector<SIZE_TYPE> cpu_row_offset_out(num_nodes + 1, 0);
    std::vector<SIZE_TYPE> cpu_row_offset_in(num_nodes + 1, 0);

    SIZE_TYPE current_pos_out = 0;
    SIZE_TYPE current_pos_in = 0;
    for (KEY_TYPE src = 0; src < static_cast<KEY_TYPE>(num_nodes); ++src) {
        cpu_row_offset_out[src] = current_pos_out;
        cpu_row_offset_in[src] = current_pos_in;

        for (const auto& edge : graph.OUTs[src]) {
            cpu_keys_out.push_back((src << 32) | static_cast<KEY_TYPE>(edge.first));
            cpu_values_out.push_back(static_cast<VALUE_TYPE>(edge.second));
            ++current_pos_out;
        }
        for (const auto& edge : graph.INs[src]) {
            cpu_keys_in.push_back((src << 32) | static_cast<KEY_TYPE>(edge.first));
            cpu_values_in.push_back(static_cast<VALUE_TYPE>(edge.second));
            ++current_pos_in;
        }
    }
    cpu_row_offset_out[num_nodes] = current_pos_out;
    cpu_row_offset_in[num_nodes] = current_pos_in;

    init_csr_gpma(gpma_graph_out, static_cast<SIZE_TYPE>(num_nodes));
    init_csr_gpma(gpma_graph_in, static_cast<SIZE_TYPE>(num_nodes));

    DEV_VEC_KEY dev_keys_out(cpu_keys_out.size());
    DEV_VEC_KEY dev_keys_in(cpu_keys_in.size());
    DEV_VEC_VALUE dev_values_out(cpu_values_out.size());
    DEV_VEC_VALUE dev_values_in(cpu_values_in.size());
    DEV_VEC_SIZE dev_row_offset_out(cpu_row_offset_out.size());
    DEV_VEC_SIZE dev_row_offset_in(cpu_row_offset_in.size());

    thrust::copy(cpu_keys_out.begin(), cpu_keys_out.end(), dev_keys_out.begin());
    thrust::copy(cpu_keys_in.begin(), cpu_keys_in.end(), dev_keys_in.begin());
    thrust::copy(cpu_values_out.begin(), cpu_values_out.end(), dev_values_out.begin());
    thrust::copy(cpu_values_in.begin(), cpu_values_in.end(), dev_values_in.begin());
    thrust::copy(cpu_row_offset_out.begin(), cpu_row_offset_out.end(), dev_row_offset_out.begin());
    thrust::copy(cpu_row_offset_in.begin(), cpu_row_offset_in.end(), dev_row_offset_in.begin());
    cudaDeviceSynchronize();

    gpma_graph_out.row_offset = dev_row_offset_out;
    gpma_graph_in.row_offset = dev_row_offset_in;
    init_gpma_from_csr(gpma_graph_out, dev_keys_out, dev_values_out);
    init_gpma_from_csr(gpma_graph_in, dev_keys_in, dev_values_in);
}

} // namespace

int main() {
    configure_stdio();

    graph_structure<double> graph = build_sample_graph();

    auto begin = std::chrono::high_resolution_clock::now();
    GPMA gpma_graph_out, gpma_graph_in;
    build_gpma_from_graph(graph, gpma_graph_out, gpma_graph_in);
    auto end = std::chrono::high_resolution_clock::now();
    const double graph_to_gpma_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    std::printf("LDBC to GPMA cost time: %f s\n", graph_to_gpma_time);

    std::cout << "Running BFS..." << std::endl;
    print_named_pairs("BFS result:", Cuda_BFS_optimized(graph, gpma_graph_out, "one"));

    std::cout << "Running SSSP..." << std::endl;
    print_named_pairs("SSSP result:", Cuda_SSSP_optimized(graph, gpma_graph_out, "one"));

    std::cout << "Running Connected Components..." << std::endl;
    print_named_pairs("Connected Components result:", Cuda_WCC_optimized(graph, gpma_graph_in));

    std::cout << "Running PageRank..." << std::endl;
    print_named_pairs("PageRank result:", Cuda_PR_optimized(graph, gpma_graph_in, gpma_graph_out, 10, 0.85));

    std::cout << "Running Community Detection..." << std::endl;
    print_named_pairs("Community Detection result:",
            Cuda_CDLP_optimized(graph, gpma_graph_in, gpma_graph_out, 10));

    return 0;
}
