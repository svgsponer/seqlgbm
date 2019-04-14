#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace SEQL {
static inline std::vector<std::string> split(const std::string &in,
                                             const std::string &sep) {
    std::string::size_type b = 0;
    std::vector<std::string> result;
    while ((b = in.find_first_not_of(sep, b)) != std::string::npos) {
        auto e = in.find_first_of(sep, b);
        result.push_back(in.substr(b, e - b));
        b = e;
    }
    return result;
}

// Following two function are used to sort two vector based on one of them

// Calculate permutation vector of a sort operation on a vector of type T
template <typename T, typename Compare = std::less<T>>
std::vector<std::size_t> sort_permutation(const std::vector<T> &vec,
                                          const Compare compare = Compare()) {
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&](std::size_t i, std::size_t j) {
        return compare(vec[i], vec[j]);
    });
    return p;
}

// Apply a permutation vector to a vector of type T
template <typename T>
std::vector<T> apply_permutation(const std::vector<T> &vec,
                                 const std::vector<std::size_t> &p) {
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](std::size_t i) {
        return vec[i];
    });
    return sorted_vec;
}

// Transpose column-matrix to row-matrix
template <typename T>
std::vector<std::vector<T>>
transpose(const std::vector<std::vector<T>> &column_matrix) {
    auto n_rows = column_matrix[0].size();
    auto n_cols = column_matrix.size();

    std::vector<std::vector<T>> transposed;
    for (auto row = 0u; row < n_rows; ++row) {
        std::vector<T> new_row;
        for (auto column = 0u; column < n_cols; ++column) {
            new_row.push_back(column_matrix[column][row]);
        }
        transposed.push_back(new_row);
    }
    return transposed;
}

// Writes a matrix saved as vector of columns to a file
template <typename T>
void save_column_matrix(const std::vector<std::vector<T>> &column_matrix,
                        const std::string filename,
                        const std::string sep = " ") {
    std::ofstream of(filename, std::ios::out);
    for (auto row = 0u; row < column_matrix[0].size(); ++row) {
        for (auto column = 0u; column < column_matrix.size(); ++column) {
            of << column_matrix[column][row] << sep;
        }
        of << '\n';
    }
}

// Find maximum column in each row and returns it as vector
template <typename T>
std::vector<T>
get_max_entry_column_matrix(const std::vector<std::vector<T>> column_matrix) {
    std::vector<T> ret;
    for (auto row = 0u; row < column_matrix[0].size(); ++row) {
        std::pair<int, T> maximum =
            std::make_pair(-1, std::numeric_limits<T>::lowest());
        for (auto column = 0u; column < column_matrix.size(); ++column) {
            if (column_matrix[column][row] > maximum.second) {
                maximum = std::make_pair(column, column_matrix[column][row]);
            }
        }
        ret.push_back(maximum.first);
    }
    return ret;
}

// Writes a matrix saved as vector of rows to a file
template <typename T>
void save_row_matrix(const std::vector<std::vector<T>> row_matrix,
                     const std::string filename, const std::string sep = " ") {
    std::ofstream of(filename, std::ios::out);
    for (const auto &row : row_matrix) {
        for (const auto &ele : row) {
            of << ele << sep;
        }
        of << '\n';
    }
}

// Compares pairs pair based on second element
template <typename T1, typename T2>
struct pair_2nd_cmp : public std::binary_function<bool, T1, T2> {
    bool operator()(const std::pair<T1, T2> &x1, const std::pair<T1, T2> &x2) {
        return x1.second > x2.second;
    }
};

template <typename T> double norm2(std::vector<T> v) {
    return std::sqrt(
        std::inner_product(std::begin(v), std::end(v), std::begin(v), 0.0));
}
} // namespace SEQL
#endif
