#pragma once

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>

#include <cstdio>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <CPU_adj_list/CPU_adj_list.hpp>

namespace csr_detail {

// Keep CUDA calls centralized so allocation/copy/free failures include the operation name.

inline void report_cuda_error(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    }
}

template <typename T>
inline void cuda_alloc_array(T*& pointer, std::size_t count, const char* operation) {
    pointer = nullptr;
    // CUDA accepts some zero-sized calls inconsistently across APIs; skip them explicitly.
    if (count == 0) {
        return;
    }
    report_cuda_error(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), operation);
}

template <typename T>
inline void cuda_copy_array(T* dst, const T* src, std::size_t count, cudaMemcpyKind kind, const char* operation) {
    if (count == 0) {
        return;
    }
    report_cuda_error(cudaMemcpy(dst, src, count * sizeof(T), kind), operation);
}

inline void cuda_free_array(void*& pointer) {
    if (pointer != nullptr) {
        report_cuda_error(cudaFree(pointer), "cudaFree");
        pointer = nullptr;
    }
}

template <typename T>
inline void cuda_free_typed_array(T*& pointer) {
    void* raw_pointer = pointer;
    cuda_free_array(raw_pointer);
    pointer = nullptr;
}

} // namespace csr_detail

/*for GPU*/
template <typename weight_type>
class CSR_graph {
    // CSR has space efficiency and is easy to use by GPUs.
public:
    CSR_graph() = default;
    CSR_graph(const CSR_graph& other);
    CSR_graph(CSR_graph&& other) noexcept;
    CSR_graph& operator=(CSR_graph other) noexcept;
    ~CSR_graph();

    template <typename other_weight_type>
    friend CSR_graph<other_weight_type> toCSR(graph_structure<other_weight_type>& graph, bool is_directed);

    friend void swap(CSR_graph& lhs, CSR_graph& rhs) noexcept {
        using std::swap;
        swap(lhs.is_directed, rhs.is_directed);
        swap(lhs.INs_Neighbor_start_pointers, rhs.INs_Neighbor_start_pointers);
        swap(lhs.OUTs_Neighbor_start_pointers, rhs.OUTs_Neighbor_start_pointers);
        swap(lhs.ALL_start_pointers, rhs.ALL_start_pointers);
        swap(lhs.INs_Edges, rhs.INs_Edges);
        swap(lhs.OUTs_Edges, rhs.OUTs_Edges);
        swap(lhs.all_Edges, rhs.all_Edges);
        swap(lhs.INs_Edge_weights, rhs.INs_Edge_weights);
        swap(lhs.OUTs_Edge_weights, rhs.OUTs_Edge_weights);
        swap(lhs.in_pointer, rhs.in_pointer);
        swap(lhs.out_pointer, rhs.out_pointer);
        swap(lhs.in_edge, rhs.in_edge);
        swap(lhs.out_edge, rhs.out_edge);
        swap(lhs.all_pointer, rhs.all_pointer);
        swap(lhs.all_edge, rhs.all_edge);
        swap(lhs.in_edge_weight, rhs.in_edge_weight);
        swap(lhs.out_edge_weight, rhs.out_edge_weight);
        swap(lhs.E_all, rhs.E_all);
    }

    bool is_directed = true; // direct graph or undirect graph

    // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights.
    std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers, ALL_start_pointers;
    /*
        Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
        And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    */
    std::vector<int> INs_Edges, OUTs_Edges, all_Edges;
    std::vector<weight_type> INs_Edge_weights, OUTs_Edge_weights;
    int *in_pointer = nullptr, *out_pointer = nullptr, *in_edge = nullptr, *out_edge = nullptr;
    int *all_pointer = nullptr, *all_edge = nullptr; // all_edge merges in_edge and out_edge, mainly used by CDLP.
    double *in_edge_weight = nullptr, *out_edge_weight = nullptr;
    std::size_t E_all = 0;

private:
    void allocate_device_memory(int vertices, std::size_t out_edges, std::size_t in_edges, std::size_t all_edges);
    void copy_host_to_device();
    void copy_device_from(const CSR_graph& other);
    void release_device_memory();
    void reset_moved_from() noexcept;
};

template <typename weight_type>
void CSR_graph<weight_type>::allocate_device_memory(int vertices, std::size_t out_edges, std::size_t in_edges,
        std::size_t all_edges) {
    csr_detail::cuda_alloc_array(out_pointer, static_cast<std::size_t>(vertices + 1), "cudaMalloc(out_pointer)");
    csr_detail::cuda_alloc_array(out_edge, out_edges, "cudaMalloc(out_edge)");
    csr_detail::cuda_alloc_array(out_edge_weight, out_edges, "cudaMalloc(out_edge_weight)");

    if (is_directed) {
        csr_detail::cuda_alloc_array(in_pointer, static_cast<std::size_t>(vertices + 1), "cudaMalloc(in_pointer)");
        csr_detail::cuda_alloc_array(all_pointer, static_cast<std::size_t>(vertices + 1), "cudaMalloc(all_pointer)");
        csr_detail::cuda_alloc_array(in_edge, in_edges, "cudaMalloc(in_edge)");
        csr_detail::cuda_alloc_array(all_edge, all_edges, "cudaMalloc(all_edge)");
        csr_detail::cuda_alloc_array(in_edge_weight, in_edges, "cudaMalloc(in_edge_weight)");
    } else {
        // Undirected CSR reuses outgoing storage; these are aliases and must not be freed twice.
        in_pointer = out_pointer;
        all_pointer = out_pointer;
        in_edge = out_edge;
        all_edge = out_edge;
        in_edge_weight = out_edge_weight;
    }
}

template <typename weight_type>
void CSR_graph<weight_type>::copy_host_to_device() {
    csr_detail::cuda_copy_array(out_pointer, OUTs_Neighbor_start_pointers.data(), OUTs_Neighbor_start_pointers.size(),
            cudaMemcpyHostToDevice, "cudaMemcpy(out_pointer)");
    csr_detail::cuda_copy_array(out_edge, OUTs_Edges.data(), OUTs_Edges.size(), cudaMemcpyHostToDevice,
            "cudaMemcpy(out_edge)");
    csr_detail::cuda_copy_array(out_edge_weight, OUTs_Edge_weights.data(), OUTs_Edge_weights.size(),
            cudaMemcpyHostToDevice, "cudaMemcpy(out_edge_weight)");

    if (is_directed) {
        csr_detail::cuda_copy_array(in_pointer, INs_Neighbor_start_pointers.data(), INs_Neighbor_start_pointers.size(),
                cudaMemcpyHostToDevice, "cudaMemcpy(in_pointer)");
        csr_detail::cuda_copy_array(all_pointer, ALL_start_pointers.data(), ALL_start_pointers.size(),
                cudaMemcpyHostToDevice, "cudaMemcpy(all_pointer)");
        csr_detail::cuda_copy_array(in_edge, INs_Edges.data(), INs_Edges.size(), cudaMemcpyHostToDevice,
                "cudaMemcpy(in_edge)");
        csr_detail::cuda_copy_array(all_edge, all_Edges.data(), all_Edges.size(), cudaMemcpyHostToDevice,
                "cudaMemcpy(all_edge)");
        csr_detail::cuda_copy_array(in_edge_weight, INs_Edge_weights.data(), INs_Edge_weights.size(),
                cudaMemcpyHostToDevice, "cudaMemcpy(in_edge_weight)");
    }
    csr_detail::report_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize(copy_host_to_device)");
}

template <typename weight_type>
// Deep-copy device buffers so copied CSR_graph objects own independent CUDA allocations.
void CSR_graph<weight_type>::copy_device_from(const CSR_graph& other) {
    const int vertices = static_cast<int>(other.OUTs_Neighbor_start_pointers.size()) - 1;
    allocate_device_memory(vertices, other.OUTs_Edges.size(), other.INs_Edges.size(), other.E_all);

    csr_detail::cuda_copy_array(out_pointer, other.out_pointer, other.OUTs_Neighbor_start_pointers.size(),
            cudaMemcpyDeviceToDevice, "cudaMemcpy(copy out_pointer)");
    csr_detail::cuda_copy_array(out_edge, other.out_edge, other.OUTs_Edges.size(), cudaMemcpyDeviceToDevice,
            "cudaMemcpy(copy out_edge)");
    csr_detail::cuda_copy_array(out_edge_weight, other.out_edge_weight, other.OUTs_Edge_weights.size(),
            cudaMemcpyDeviceToDevice, "cudaMemcpy(copy out_edge_weight)");

    if (is_directed) {
        csr_detail::cuda_copy_array(in_pointer, other.in_pointer, other.INs_Neighbor_start_pointers.size(),
                cudaMemcpyDeviceToDevice, "cudaMemcpy(copy in_pointer)");
        csr_detail::cuda_copy_array(all_pointer, other.all_pointer, other.ALL_start_pointers.size(),
                cudaMemcpyDeviceToDevice, "cudaMemcpy(copy all_pointer)");
        csr_detail::cuda_copy_array(in_edge, other.in_edge, other.INs_Edges.size(), cudaMemcpyDeviceToDevice,
                "cudaMemcpy(copy in_edge)");
        csr_detail::cuda_copy_array(all_edge, other.all_edge, other.all_Edges.size(), cudaMemcpyDeviceToDevice,
                "cudaMemcpy(copy all_edge)");
        csr_detail::cuda_copy_array(in_edge_weight, other.in_edge_weight, other.INs_Edge_weights.size(),
                cudaMemcpyDeviceToDevice, "cudaMemcpy(copy in_edge_weight)");
    }
}

template <typename weight_type>
void CSR_graph<weight_type>::release_device_memory() {
    if (is_directed) {
        // Directed graphs own all CSR buffers independently.
        csr_detail::cuda_free_typed_array(in_pointer);
        csr_detail::cuda_free_typed_array(in_edge);
        csr_detail::cuda_free_typed_array(in_edge_weight);
        csr_detail::cuda_free_typed_array(all_pointer);
        csr_detail::cuda_free_typed_array(all_edge);
    } else {
        // Alias pointers were released through the outgoing buffers below.
        in_pointer = nullptr;
        in_edge = nullptr;
        in_edge_weight = nullptr;
        all_pointer = nullptr;
        all_edge = nullptr;
    }

    csr_detail::cuda_free_typed_array(out_pointer);
    csr_detail::cuda_free_typed_array(out_edge);
    csr_detail::cuda_free_typed_array(out_edge_weight);
}

template <typename weight_type>
// Leave moved-from objects destructible without touching transferred device memory.
void CSR_graph<weight_type>::reset_moved_from() noexcept {
    in_pointer = nullptr;
    out_pointer = nullptr;
    in_edge = nullptr;
    out_edge = nullptr;
    all_pointer = nullptr;
    all_edge = nullptr;
    in_edge_weight = nullptr;
    out_edge_weight = nullptr;
    E_all = 0;
}

template <typename weight_type>
CSR_graph<weight_type>::CSR_graph(const CSR_graph& other)
        : is_directed(other.is_directed),
          INs_Neighbor_start_pointers(other.INs_Neighbor_start_pointers),
          OUTs_Neighbor_start_pointers(other.OUTs_Neighbor_start_pointers),
          ALL_start_pointers(other.ALL_start_pointers),
          INs_Edges(other.INs_Edges),
          OUTs_Edges(other.OUTs_Edges),
          all_Edges(other.all_Edges),
          INs_Edge_weights(other.INs_Edge_weights),
          OUTs_Edge_weights(other.OUTs_Edge_weights),
          E_all(other.E_all) {
    if (!OUTs_Neighbor_start_pointers.empty()) {
        copy_device_from(other);
    }
}

template <typename weight_type>
CSR_graph<weight_type>::CSR_graph(CSR_graph&& other) noexcept
        : is_directed(other.is_directed),
          INs_Neighbor_start_pointers(std::move(other.INs_Neighbor_start_pointers)),
          OUTs_Neighbor_start_pointers(std::move(other.OUTs_Neighbor_start_pointers)),
          ALL_start_pointers(std::move(other.ALL_start_pointers)),
          INs_Edges(std::move(other.INs_Edges)),
          OUTs_Edges(std::move(other.OUTs_Edges)),
          all_Edges(std::move(other.all_Edges)),
          INs_Edge_weights(std::move(other.INs_Edge_weights)),
          OUTs_Edge_weights(std::move(other.OUTs_Edge_weights)),
          in_pointer(other.in_pointer),
          out_pointer(other.out_pointer),
          in_edge(other.in_edge),
          out_edge(other.out_edge),
          all_pointer(other.all_pointer),
          all_edge(other.all_edge),
          in_edge_weight(other.in_edge_weight),
          out_edge_weight(other.out_edge_weight),
          E_all(other.E_all) {
    other.reset_moved_from();
}

template <typename weight_type>
CSR_graph<weight_type>& CSR_graph<weight_type>::operator=(CSR_graph other) noexcept {
    swap(*this, other);
    return *this;
}

template <typename weight_type>
CSR_graph<weight_type>::~CSR_graph() {
    release_device_memory();
}

template <typename weight_type>
CSR_graph<weight_type> toCSR(graph_structure<weight_type>& graph, bool is_directed = true) {
    CSR_graph<weight_type> array;
    array.is_directed = is_directed;

    const int vertices = graph.size();
    array.OUTs_Neighbor_start_pointers.resize(static_cast<std::size_t>(vertices) + 1);

    int pointer = 0;
    if (is_directed) {
        array.INs_Neighbor_start_pointers.resize(static_cast<std::size_t>(vertices) + 1);
        for (int i = 0; i < vertices; i++) {
            array.INs_Neighbor_start_pointers[i] = pointer;
            for (const auto& edge : graph.INs[i]) {
                array.INs_Edges.push_back(edge.first);
                array.INs_Edge_weights.push_back(edge.second);
            }
            pointer += static_cast<int>(graph.INs[i].size());
        }
        array.INs_Neighbor_start_pointers[vertices] = pointer;
    }

    pointer = 0;
    for (int i = 0; i < vertices; i++) {
        array.OUTs_Neighbor_start_pointers[i] = pointer;
        for (const auto& edge : graph.OUTs[i]) {
            array.OUTs_Edges.push_back(edge.first);
            array.OUTs_Edge_weights.push_back(edge.second);
        }
        pointer += static_cast<int>(graph.OUTs[i].size());
    }
    array.OUTs_Neighbor_start_pointers[vertices] = pointer;

    if (is_directed) {
        // CDLP needs a merged in/out adjacency stream while other algorithms keep directions separate.
        array.ALL_start_pointers.resize(static_cast<std::size_t>(vertices) + 1);
        pointer = 0;
        for (int i = 0; i < vertices; i++) {
            array.ALL_start_pointers[i] = pointer;
            for (const auto& edge : graph.INs[i]) {
                array.all_Edges.push_back(edge.first);
            }
            for (const auto& edge : graph.OUTs[i]) {
                array.all_Edges.push_back(edge.first);
            }
            pointer += static_cast<int>(graph.INs[i].size() + graph.OUTs[i].size());
        }
        array.ALL_start_pointers[vertices] = pointer;
    }

    const std::size_t out_edges = array.OUTs_Edges.size();
    const std::size_t in_edges = array.INs_Edges.size();
    array.E_all = is_directed ? in_edges + out_edges : out_edges;

    array.allocate_device_memory(vertices, out_edges, in_edges, array.E_all);
    array.copy_host_to_device();
    return array;
}
