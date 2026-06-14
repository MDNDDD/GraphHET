#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Binary layout: outer size, then for each row its size followed by contiguous T elements.
template<typename T>
void binary_save_vector_of_vectors(const std::string& path, const std::vector<std::vector<T>>& myVector);

template<typename T>
void binary_read_vector_of_vectors(const std::string& path, std::vector<std::vector<T>>& myVector);

template<typename T>
void binary_save_vector_of_vectors(const std::string& path, const std::vector<std::vector<T>>& myVector) {
    std::ofstream file(path, std::ios::out | std::ofstream::binary);
    if (!file) {
        std::cout << "Unable to open file " << path << std::endl
                  << "Please check the file location or file name." << std::endl;
        std::exit(1);
    }

    const int outer_size = static_cast<int>(myVector.size());
    file.write(reinterpret_cast<const char*>(&outer_size), sizeof(outer_size));

    // Write each inner vector as one contiguous block to avoid per-element overhead.
    for (const auto& row : myVector) {
        const int row_size = static_cast<int>(row.size());
        file.write(reinterpret_cast<const char*>(&row_size), sizeof(row_size));
        if (!row.empty()) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(T));
        }
    }
}

template<typename T>
void binary_read_vector_of_vectors(const std::string& path, std::vector<std::vector<T>>& myVector) {
    myVector.clear();

    std::ifstream file(path, std::ios::in | std::ifstream::binary);
    if (!file) {
        std::cout << "Unable to open file " << path << std::endl
                  << "Please check the file location or file name." << std::endl;
        std::exit(1);
    }

    int outer_size = 0;
    file.read(reinterpret_cast<char*>(&outer_size), sizeof(outer_size));
    if (!file || outer_size < 0) {
        std::cout << "Unable to read file " << path << std::endl;
        std::exit(1);
    }

    // Resize first, then read directly into each row buffer for efficient reconstruction.
    myVector.resize(static_cast<std::size_t>(outer_size));
    for (int n = 0; n < outer_size; ++n) {
        int row_size = 0;
        file.read(reinterpret_cast<char*>(&row_size), sizeof(row_size));
        if (!file || row_size < 0) {
            std::cout << "Unable to read file " << path << std::endl;
            std::exit(1);
        }

        auto& row = myVector[static_cast<std::size_t>(n)];
        row.resize(static_cast<std::size_t>(row_size));
        if (!row.empty()) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(T));
            if (!file) {
                std::cout << "Unable to read file " << path << std::endl;
                std::exit(1);
            }
        }
    }
}

/*
---------an example main file-------------
#include <text_mining/binary_save_read_vector_of_vectors.h>

int main()
{
    ;
}
-------------------
*/
