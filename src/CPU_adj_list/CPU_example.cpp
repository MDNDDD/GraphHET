#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <CPU_adj_list/algorithm/CPU_BFS.hpp>
#include <CPU_adj_list/algorithm/CPU_connected_components.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>
#include <CPU_adj_list/algorithm/CPU_PageRank.hpp>
#include <CPU_adj_list/algorithm/CPU_Community_Detection.hpp>

namespace {

void configure_stdio() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
}

graph_structure<double> build_sample_graph() {
    graph_structure<double> graph;
    graph.add_vertice("Andy");
    graph.add_vertice("Bob");
    graph.add_vertice("Tom");
    graph.add_vertice("Sam");
    graph.add_vertice("Kevin");
    graph.add_vertice("Leo");

    graph.add_edge("Andy", "Bob", 1);
    graph.add_edge("Bob", "Tom", 1);
    graph.add_edge("Bob", "Sam", 1);
    graph.add_edge("Andy", "Sam", 1);
    graph.add_edge("Bob", "Bob", 1);
    graph.add_edge("Andy", "Leo", 1);
    graph.add_edge("Leo", "Sam", 1);
    graph.add_edge("Sam", "Leo", 1);

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

    std::cout << "Running BFS..." << std::endl;
    print_named_pairs("BFS result:", CPU_Bfs(graph, "one"));

    std::cout << "Running Connected Components..." << std::endl;
    print_named_pairs("Connected Components result:", CPU_WCC(graph));

    std::cout << "Running SSSP..." << std::endl;
    print_named_pairs("SSSP result:", CPU_SSSP(graph, "one"));

    std::cout << "Running PageRank..." << std::endl;
    print_named_pairs("PageRank result:", CPU_PR(graph, 10, 0.85));

    std::cout << "Running Community Detection..." << std::endl;
    print_named_pairs("Community Detection result:", CPU_CDLP(graph, 10));

    return 0;
}
