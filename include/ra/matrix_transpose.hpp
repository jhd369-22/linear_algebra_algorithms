#include <cstdio>

namespace ra::cache {
    // The matrix_transpose function is to utilize the cache-oblivious algorithm. This algorithm
    // uses a divide and conquer strategy and is based on recursion. Note that, for optimal efficiency, the recursion
    // should not be continued until a 1x1 matrix is encountered. For example, the base case for the recursion might
    // be chosen to correspond to m*n <= 64.
    template <class T>
    void matrix_transpose(const T* a, std::size_t m, std::size_t n, T* b) {

    }

    template <class T>
    void naive_matrix_transpose(const T* a, std::size_t m, std::size_t n, T* b){
        for(std::size_t i = 0; i < m; ++i){
            for(std::size_t j = 0; j < n; ++j){
                b[j][i] = a[i][j];
            }
        }
    }
}