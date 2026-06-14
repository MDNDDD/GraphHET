#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <CPU_adj_list/algorithm/CPU_BFS.hpp>
#include <CPU_adj_list/algorithm/CPU_BFS_pre.hpp>
#include <CPU_adj_list/algorithm/CPU_Community_Detection.hpp>
#include <CPU_adj_list/algorithm/CPU_PageRank.hpp>
#include <CPU_adj_list/algorithm/CPU_connected_components.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>
#include <CPU_adj_list/algorithm/CPU_sssp_pre.hpp>

#include <LDBC/checker.hpp>
#include <LDBC/ldbc.hpp>

namespace {

using EdgeList = std::vector<std::pair<std::string, std::string>>;
using ResultList = std::vector<std::pair<std::string, std::string>>;

EdgeList g_edges_to_add;
EdgeList g_edges_to_delete;

void configure_stdio() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
}

std::pair<std::string, std::string> read_input_paths() {
    std::string directory;
    std::cout << "Please input the data directory: " << std::endl;
    std::cin >> directory;
    if (!directory.empty() && directory.back() != '/') {
        directory += '/';
    }

    std::string graph_name;
    std::cout << "Please input the graph name: " << std::endl;
    std::cin >> graph_name;
    return {directory, graph_name};
}

template <typename Func>
double measure_average_seconds(int iterations, Func&& func) {
    const auto begin = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        func();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const double total_seconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    return total_seconds / static_cast<double>(iterations);
}

void append_result(ResultList& results, const std::string& name, const std::string& value) {
    results.emplace_back(name, value);
}

void print_result_row(const ResultList& results) {
    std::cout << "Result: " << std::endl;
    for (std::size_t index = 0; index < results.size(); ++index) {
        std::cout << results[index].second;
        if (index + 1 != results.size()) {
            std::cout << ',';
        }
    }
    std::cout << std::endl;
}

void read_edge_file(const std::string& graph_file_path) {
    std::ifstream infile_add(graph_file_path + "-add-edge.txt");
    std::ifstream infile_delete(graph_file_path + "-delete-edge.txt");

    std::string line;
    while (std::getline(infile_add, line)) {
        std::istringstream iss(line);
        std::string command;
        std::string src;
        std::string dst;
        iss >> command >> src >> dst;
        g_edges_to_add.emplace_back(src, dst);
    }

    while (std::getline(infile_delete, line)) {
        std::istringstream iss(line);
        std::string command;
        std::string src;
        std::string dst;
        iss >> command >> src >> dst;
        g_edges_to_delete.emplace_back(src, dst);
    }
}

void add_edge(graph_structure<double>& graph) {
    for (const auto& edge : g_edges_to_add) {
        graph.add_edge(edge.first, edge.second, 1);
    }
}

void delete_edge(graph_structure<double>& graph) {
    for (const auto& edge : g_edges_to_delete) {
        graph.remove_edge(edge.first, edge.second);
    }
}

template <typename Runner, typename Checker>
void run_checked_algorithm(ResultList& results, const std::string& result_name, const char* timing_label,
        int iterations, Runner&& runner, Checker&& checker) {
    try {
        auto output = runner();
        const double average_seconds = measure_average_seconds(iterations, [&]() {
            output = runner();
        });
        std::printf("%s cost time: %f s\n", timing_label, average_seconds);
        append_result(results, result_name, checker(output) ? std::to_string(average_seconds) : "wrong");
    } catch (...) {
        append_result(results, result_name, "failed!");
    }
}

} // namespace

int main() {
    configure_stdio();

    const auto [directory, graph_name] = read_input_paths();
    const std::string config_file_path = directory + graph_name + ".properties";

    LDBC<double> graph(directory, graph_name);
    graph.read_config(config_file_path);

    const double load_ldbc_time = measure_average_seconds(1, [&]() {
        graph.load_graph();
    });
    std::printf("load_ldbc_time cost time: %f s\n", load_ldbc_time);
    std::printf("CPU memory used: %.5lf MB\n", static_cast<double>(graph.getTotalMemory()) / (1024.0 * 1024.0));

    ResultList result_all;
    constexpr int kIterations = 1;

    if (graph.sup_bfs) {
        run_checked_algorithm(result_all, "BFS", "CPU BFS", kIterations,
                [&]() { return CPU_Bfs(graph, graph.bfs_src_name); },
                [&](auto& output) { return Bfs_checker(graph, output, graph.base_path + "-BFS"); });
    } else {
        append_result(result_all, "BFS", "N/A");
    }

    if (!graph.sup_sssp) {
        graph.sup_sssp = 1;
    }
    if (graph.sup_sssp) {
        run_checked_algorithm(result_all, "SSSP", "CPU SSSP", kIterations,
                [&]() { return CPU_SSSP(graph, graph.sssp_src_name); },
                [&](auto& output) { return SSSP_checker(graph, output, graph.base_path + "-SSSP"); });
    } else {
        append_result(result_all, "SSSP", "N/A");
    }

    if (graph.sup_wcc) {
        run_checked_algorithm(result_all, "WCC", "CPU WCC", kIterations,
                [&]() { return CPU_WCC(graph); },
                [&](auto& output) { return WCC_checker(graph, output, graph.base_path + "-WCC"); });
    } else {
        append_result(result_all, "WCC", "N/A");
    }

    if (graph.sup_pr) {
        run_checked_algorithm(result_all, "PageRank", "CPU PageRank", kIterations,
                [&]() { return CPU_PR(graph, graph.pr_its, graph.pr_damping); },
                [&](auto& output) { return PR_checker(graph, output, graph.base_path + "-PR"); });
    } else {
        append_result(result_all, "PageRank", "N/A");
    }

    if (graph.sup_cdlp) {
        run_checked_algorithm(result_all, "CommunityDetection", "CPU Community Detection", kIterations,
                [&]() { return CPU_CDLP(graph, graph.cdlp_max_its); },
                [&](auto& output) { return CDLP_checker(graph, output, graph.base_path + "-CDLP"); });
    } else {
        append_result(result_all, "CommunityDetection", "N/A");
    }

    print_result_row(result_all);
    graph.save_to_CSV(result_all, "./result-cpu.csv");
    return 0;
}
