#pragma once

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

/*
for a sorted vector<pair<int,T>>, this file provides binary operations on this vector.

The int values are unique and sorted from small to large.

https://blog.csdn.net/EbowTang/article/details/50770315
*/

namespace sorted_vector_detail {

// Shared lookup primitive: all public operations assume the vector is sorted by pair::first.

template <typename T>
auto lower_bound_by_key(std::vector<std::pair<int, T>>& input_vector, int key) {
    return std::lower_bound(input_vector.begin(), input_vector.end(), key,
            [](const std::pair<int, T>& item, int value) {
                return item.first < value;
            });
}

template <typename T>
auto lower_bound_by_key(const std::vector<std::pair<int, T>>& input_vector, int key) {
    return std::lower_bound(input_vector.begin(), input_vector.end(), key,
            [](const std::pair<int, T>& item, int value) {
                return item.first < value;
            });
}

} // namespace sorted_vector_detail

template <typename T>
bool sorted_vector_binary_operations_search(const std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
T sorted_vector_binary_operations_search_weight(const std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
int sorted_vector_binary_operations_search_position(const std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
void sorted_vector_binary_operations_erase(std::vector<std::pair<int, T>>& input_vector, int key);

template <typename T>
int sorted_vector_binary_operations_insert(std::vector<std::pair<int, T>>& input_vector, int key, const T& load);

template <typename T>
bool sorted_vector_binary_operations_search(const std::vector<std::pair<int, T>>& input_vector, int key) {
    // lower_bound keeps the lookup O(log n); equality check distinguishes insertion point from match.
    const auto it = sorted_vector_detail::lower_bound_by_key(input_vector, key);
    return it != input_vector.end() && it->first == key;
}

template <typename T>
T sorted_vector_binary_operations_search_weight(const std::vector<std::pair<int, T>>& input_vector, int key) {
    // Preserve the historical sentinel behavior when no edge weight exists.
    const auto it = sorted_vector_detail::lower_bound_by_key(input_vector, key);
    if (it != input_vector.end() && it->first == key) {
        return it->second;
    }
    return std::numeric_limits<T>::max();
}

template <typename T>
int sorted_vector_binary_operations_search_position(const std::vector<std::pair<int, T>>& input_vector, int key) {
    /*return -1 if key is not in vector; time complexity O(log n)*/
    const auto it = sorted_vector_detail::lower_bound_by_key(input_vector, key);
    if (it != input_vector.end() && it->first == key) {
        return static_cast<int>(std::distance(input_vector.begin(), it));
    }
    return -1;
}

template <typename T>
void sorted_vector_binary_operations_erase(std::vector<std::pair<int, T>>& input_vector, int key) {
    // Lookup is logarithmic, but vector erase still shifts the suffix in O(n).
    const auto it = sorted_vector_detail::lower_bound_by_key(input_vector, key);
    if (it != input_vector.end() && it->first == key) {
        input_vector.erase(it);
    }
}

template <typename T>
int sorted_vector_binary_operations_insert(std::vector<std::pair<int, T>>& input_vector, int key, const T& load) {
    // Update existing keys in place; otherwise insert at the sorted position to keep adjacency lists ordered.
    auto it = sorted_vector_detail::lower_bound_by_key(input_vector, key);
    if (it != input_vector.end() && it->first == key) {
        it->second = load;
        return static_cast<int>(std::distance(input_vector.begin(), it));
    }

    const auto insert_pos = static_cast<int>(std::distance(input_vector.begin(), it));
    input_vector.insert(it, {key, load});
    return insert_pos;
}
