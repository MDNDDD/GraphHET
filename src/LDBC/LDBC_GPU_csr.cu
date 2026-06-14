#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <GPU_csr/algorithm/GPU_BFS_csr.cuh>
#include <GPU_csr/algorithm/GPU_BFS_pre_csr.cuh>
#include <GPU_csr/algorithm/GPU_CDLP_csr.cuh>
#include <GPU_csr/algorithm/GPU_PR_csr.cuh>
#include <GPU_csr/algorithm/GPU_SSSP_csr.cuh>
#include <GPU_csr/algorithm/GPU_SSSP_pre_csr.cuh>
#include <GPU_csr/algorithm/GPU_WCC_csr.cuh>

#include <LDBC/checker.hpp>
#include <LDBC/ldbc.hpp>

namespace {

using ResultList = std::vector<std::pair<std::string, std::string>>;

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

void print_gpu_usage(std::size_t free_before, std::size_t total_before) {
    std::size_t free_after = 0;
    std::size_t total_after = 0;
    cudaMemGetInfo(&free_after, &total_after);
    const double mem_used = static_cast<double>(free_before - free_after);
    std::printf("GPU usage: %.5f MB / %.5f MB (%.5f%% used)\n",
            mem_used / (1024.0 * 1024.0),
            total_after / (1024.0 * 1024.0),
            mem_used * 100.0 / static_cast<double>(total_after));
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

    ResultList result_all;
    const auto [directory, graph_name] = read_input_paths();
    const std::string config_file_path = directory + graph_name + ".properties";

    LDBC<double> graph(directory, graph_name);
    graph.read_config(config_file_path);

    const double load_ldbc_time = measure_average_seconds(1, [&]() {
        graph.load_graph();
    });
    std::printf("load_ldbc_time cost time: %f s\n", load_ldbc_time);

    std::size_t free_before = 0;
    std::size_t total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);

    CSR_graph<double> csr_graph;
    const double graph_to_csr_time = measure_average_seconds(1, [&]() {
        csr_graph = toCSR(graph, graph.is_directed);
    });
    std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size() - 1 << std::endl;
    std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;
    std::printf("graph_to_csr_time cost time: %f s\n", graph_to_csr_time);
    print_gpu_usage(free_before, total_before);

    constexpr int kIterations = 1;

    if (graph.sup_bfs) {
        run_checked_algorithm(result_all, "BFS", "GPU BFS", kIterations,
                [&]() { return Cuda_BFS(graph, csr_graph, graph.bfs_src_name); },
                [&](auto& output) { return Bfs_checker(graph, output, graph.base_path + "-BFS"); });
    } else {
        append_result(result_all, "BFS", "N/A");
    }

    if (!graph.sup_sssp) {
        graph.sup_sssp = 1;
    }
    if (graph.sup_sssp) {
        run_checked_algorithm(result_all, "SSSP", "GPU SSSP", kIterations,
                [&]() { return Cuda_SSSP(graph, csr_graph, graph.sssp_src_name, std::numeric_limits<double>::max()); },
                [&](auto& output) { return SSSP_checker(graph, output, graph.base_path + "-SSSP"); });

        try {
            std::vector<int> pre_v;
            (void) Cuda_SSSP_pre(graph, csr_graph, graph.sssp_src_name, pre_v, std::numeric_limits<double>::max());
        } catch (...) {
        }
    } else {
        append_result(result_all, "SSSP", "N/A");
    }

    if (graph.sup_wcc) {
        run_checked_algorithm(result_all, "WCC", "GPU WCC", kIterations,
                [&]() { return Cuda_WCC(graph, csr_graph); },
                [&](auto& output) { return WCC_checker(graph, output, graph.base_path + "-WCC"); });
    } else {
        append_result(result_all, "WCC", "N/A");
    }

    if (graph.sup_pr) {
        run_checked_algorithm(result_all, "PR", "GPU PageRank", kIterations,
                [&]() { return Cuda_PR(graph, csr_graph, graph.pr_its, graph.pr_damping); },
                [&](auto& output) { return PR_checker(graph, output, graph.base_path + "-PR"); });
    } else {
        append_result(result_all, "PR", "N/A");
    }

    if (graph.sup_cdlp) {
        run_checked_algorithm(result_all, "CDLP", "GPU Community Detection", kIterations,
                [&]() { return Cuda_CDLP(graph, csr_graph, graph.cdlp_max_its); },
                [&](auto& output) { return CDLP_checker(graph, output, graph.base_path + "-CDLP"); });
    } else {
        append_result(result_all, "CDLP", "N/A");
    }

    print_result_row(result_all);
    graph.save_to_CSV(result_all, "./result-gpu-csr.csv");
    return 0;
}
