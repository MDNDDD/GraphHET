#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <GPU_csr/GPU_csr.hpp>

#include <GPU_csr/algorithm/GPU_BFS_csr.cuh>
#include <GPU_csr/algorithm/GPU_WCC_csr.cuh>
#include <GPU_csr/algorithm/GPU_SSSP_csr.cuh>
#include <GPU_csr/algorithm/GPU_SSSP_pre_csr.cuh>
#include <GPU_csr/algorithm/GPU_PR_csr.cuh>
#include <GPU_csr/algorithm/GPU_CDLP_csr.cuh>

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

} // namespace

int main() {
    configure_stdio();

    graph_structure<double> graph = build_sample_graph();

    auto begin = std::chrono::high_resolution_clock::now();
    CSR_graph<double> csr_graph = toCSR(graph);
    auto end = std::chrono::high_resolution_clock::now();
    const double graph_to_csr_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    std::printf("LDBC to CSR cost time: %f s\n", graph_to_csr_time);

    std::cout << "Running BFS..." << std::endl;
    print_named_pairs("BFS result:", Cuda_BFS(graph, csr_graph, "one"));

    std::cout << "Running Connected Components..." << std::endl;
    print_named_pairs("Connected Components result:", Cuda_WCC(graph, csr_graph));

    std::cout << "Running SSSP..." << std::endl;
    std::vector<int> pre_v;
    print_named_pairs("SSSP result:",
            Cuda_SSSP_pre(graph, csr_graph, "one", pre_v, std::numeric_limits<double>::max()));

    std::cout << "Running PageRank..." << std::endl;
    print_named_pairs("PageRank result:", Cuda_PR(graph, csr_graph, 10, 0.85));

    std::cout << "Running Community Detection..." << std::endl;
    print_named_pairs("Community Detection result:", Cuda_CDLP(graph, csr_graph, 10));

    return 0;
}
