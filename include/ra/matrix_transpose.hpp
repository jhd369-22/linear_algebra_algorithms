#include <algorithm>  // std::copy
#include <cstdio>

namespace ra::cache {
    
    template <typename T>
    void matrix_transpose_helper(const T* a, std::size_t m, std::size_t n, T* b,
                                 std::size_t t_m, std::size_t t_n) {

        if (t_m * t_n <= 4) {
            for (std::size_t i = 0; i < t_m; ++i) {
                for (std::size_t j = 0; j < t_n; ++j) {
                    b[j * m + i] = a[i * n + j];
                }
            }
        } else {
            if (t_m < t_n) {
                matrix_transpose_helper(a, m, n, b, t_m, t_n / 2);
                matrix_transpose_helper(a + (t_n / 2), m, n, b + (t_n / 2) * m, t_m, t_n - t_n / 2);
            } else {
                matrix_transpose_helper(a, m, n, b, t_m / 2, t_n);
                matrix_transpose_helper(a + (t_m / 2) * n, m, n, b + (t_m / 2), t_m - t_m / 2, t_n);
            }
        }

        return;
    }

    // cache-oblivious matrix transpose
    template <class T>
    void matrix_transpose(const T* a, std::size_t m, std::size_t n, T* b) {

        if(a == b) {
            T* tmp = new T[n * m];
            matrix_transpose_helper(a, m, n, tmp, m, n);
            std::copy(tmp, tmp + n * m, b);
            delete[] tmp;
        }
        else{
            matrix_transpose_helper(a, m, n, b, m, n);
        }
    }

    template <class T>
    void naive_matrix_transpose(const T* a, std::size_t m, std::size_t n, T* b) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                b[j * m + i] = a[i * n + j];
            }
        }
    }
}  // namespace ra::cache