#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <GPU_gpma/algorithm/GPU_BFS_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_CDLP_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_PR_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_SSSP_gpma.cuh>
#include <GPU_gpma/algorithm/GPU_WCC_gpma.cuh>

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

inline KEY_TYPE make_edge_key(KEY_TYPE src, KEY_TYPE dst) {
    return (src << 32) | (dst & 0xFFFFFFFFu);
}

void read_edge_file(const std::string& graph_file_path) {
    const std::string add_file_path = graph_file_path + "-add-edge.txt";
    const std::string delete_file_path = graph_file_path + "-delete-edge.txt";

    std::ifstream infile_add(add_file_path);
    if (!infile_add.is_open()) {
        throw std::runtime_error("Failed to open add edge file: " + add_file_path);
    }

    std::ifstream infile_delete(delete_file_path);
    if (!infile_delete.is_open()) {
        throw std::runtime_error("Failed to open delete edge file: " + delete_file_path);
    }

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

void batch_update_edges(const EdgeList& edges, VALUE_TYPE value, GPMA& gpma_in, GPMA& gpma_out, LDBC<double>& graph) {
    constexpr std::size_t kBatchSize = 1000;
    const std::size_t update_count = std::min(kBatchSize, edges.size());
    DEV_VEC_KEY keys_in(update_count);
    DEV_VEC_KEY keys_out(update_count);
    DEV_VEC_VALUE values_in(update_count, value);
    DEV_VEC_VALUE values_out(update_count, value);

    for (std::size_t index = 0; index < update_count; ++index) {
        const auto& edge = edges[index];
        const int src = graph.vertex_str_to_id[edge.first];
        const int dst = graph.vertex_str_to_id[edge.second];
        keys_in[index] = make_edge_key(src, dst);
        keys_out[index] = make_edge_key(dst, src);
    }

    update_gpma(gpma_out, keys_out, values_out);
    if (graph.is_directed) {
        update_gpma(gpma_in, keys_in, values_in);
    }
}

void batch_add_edge(GPMA& gpma_in, GPMA& gpma_out, LDBC<double>& graph) {
    batch_update_edges(g_edges_to_add, 1, gpma_in, gpma_out, graph);
}

void batch_delete_edge(GPMA& gpma_in, GPMA& gpma_out, LDBC<double>& graph) {
    batch_update_edges(g_edges_to_delete, VALUE_NONE, gpma_in, gpma_out, graph);
}

void build_gpma_from_graph(LDBC<double>& graph, GPMA& gpma_graph_out, GPMA& gpma_graph_in) {
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
            cpu_keys_out.push_back(make_edge_key(src, static_cast<KEY_TYPE>(edge.first)));
            cpu_values_out.push_back(static_cast<VALUE_TYPE>(edge.second));
            ++current_pos_out;
        }
        for (const auto& edge : graph.INs[src]) {
            cpu_keys_in.push_back(make_edge_key(src, static_cast<KEY_TYPE>(edge.first)));
            cpu_values_in.push_back(static_cast<VALUE_TYPE>(edge.second));
            ++current_pos_in;
        }
    }
    cpu_row_offset_out[num_nodes] = current_pos_out;
    cpu_row_offset_in[num_nodes] = current_pos_in;

    init_csr_gpma(gpma_graph_out, static_cast<SIZE_TYPE>(num_nodes));
    init_csr_gpma(gpma_graph_in, static_cast<SIZE_TYPE>(num_nodes));
    cudaDeviceSynchronize();

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
    update_gpma(gpma_graph_out, dev_keys_out, dev_values_out);

    if (!graph.is_directed) {
        gpma_graph_in = gpma_graph_out;
        return;
    }
    update_gpma(gpma_graph_in, dev_keys_in, dev_values_in);
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
    std::size_t free_before = 0;
    std::size_t total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);

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

    GPMA gpma_graph_out;
    GPMA gpma_graph_in;
    const double graph_to_gpma_time = measure_average_seconds(1, [&]() {
        build_gpma_from_graph(graph, gpma_graph_out, gpma_graph_in);
    });
    std::printf("LDBC to GPMA cost time: %f s\n", graph_to_gpma_time);
    print_gpu_usage(free_before, total_before);

    constexpr int kIterations = 1;

    if (graph.sup_bfs) {
        run_checked_algorithm(result_all, "BFS", "GPU BFS", kIterations,
                [&]() { return Cuda_BFS_optimized(graph, gpma_graph_out, graph.bfs_src_name); },
                [&](auto& output) { return Bfs_checker(graph, output, graph.base_path + "-BFS"); });
    } else {
        append_result(result_all, "BFS", "N/A");
    }

    if (!graph.sup_sssp) {
        graph.sup_sssp = 1;
    }
    if (graph.sup_sssp) {
        run_checked_algorithm(result_all, "SSSP", "GPU SSSP", kIterations,
                [&]() {
                    return Cuda_SSSP_optimized(graph, gpma_graph_out, graph.sssp_src_name,
                            std::numeric_limits<double>::max());
                },
                [&](auto& output) { return SSSP_checker(graph, output, graph.base_path + "-SSSP"); });
    } else {
        append_result(result_all, "SSSP", "N/A");
    }

    if (graph.sup_wcc) {
        run_checked_algorithm(result_all, "WCC", "GPU WCC", kIterations,
                [&]() { return Cuda_WCC_optimized(graph, gpma_graph_out); },
                [&](auto& output) { return WCC_checker(graph, output, graph.base_path + "-WCC"); });
    } else {
        append_result(result_all, "WCC", "N/A");
    }

    if (graph.sup_pr) {
        run_checked_algorithm(result_all, "PR", "GPU PageRank", kIterations,
                [&]() { return Cuda_PR_optimized(graph, gpma_graph_in, gpma_graph_out, graph.pr_its, graph.pr_damping); },
                [&](auto& output) { return PR_checker(graph, output, graph.base_path + "-PR"); });
    } else {
        append_result(result_all, "PR", "N/A");
    }

    if (graph.sup_cdlp) {
        run_checked_algorithm(result_all, "CDLP", "GPU Community Detection", kIterations,
                [&]() { return Cuda_CDLP_optimized(graph, gpma_graph_in, gpma_graph_out, graph.cdlp_max_its); },
                [&](auto& output) { return CDLP_checker(graph, output, graph.base_path + "-CDLP"); });
    } else {
        append_result(result_all, "CDLP", "N/A");
    }

    print_result_row(result_all);
    graph.save_to_CSV(result_all, "./result-gpu-gpma.csv");
    return 0;
}
