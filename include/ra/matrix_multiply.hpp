#include <algorithm>
#include <cstdio>

namespace ra::cache {

    template <typename T>
    void matrix_multiply_helper(const T* a, const T* b, std::size_t m, std::size_t n, std::size_t p, T* c,
                                std::size_t t_m, std::size_t t_n, std::size_t t_p) {
        if (t_m == 1 && t_n == 1 && t_p == 1) {
            for (std::size_t i = 0; i < t_m; ++i) {
                for (std::size_t j = 0; j < t_p; ++j) {
                    for (std::size_t k = 0; k < t_n; ++k) {
                        c[i * p + j] += a[i * n + k] * b[k * p + j];
                    }
                }
            }
        } else {
            if (std::max({t_m, t_n, t_p}) == t_m) {
                matrix_multiply_helper(a, b, m, n, p, c, t_m / 2, t_n, t_p);
                matrix_multiply_helper(a + (t_m / 2) * n, b, m, n, p, c + (t_m / 2) * p, t_m - t_m / 2, t_n, t_p);
            } else if (std::max({t_m, t_n, t_p}) == t_n) {
                matrix_multiply_helper(a, b, m, n, p, c, t_m, t_n / 2, t_p);
                matrix_multiply_helper(a + (t_n / 2), b + (t_n / 2) * p, m, n, p, c, t_m, t_n - t_n / 2, t_p);
            } else {
                matrix_multiply_helper(a, b, m, n, p, c, t_m, t_n, t_p / 2);
                matrix_multiply_helper(a, b + (t_p / 2), m, n, p, c + t_p / 2, t_m, t_n, t_p - t_p / 2);
            }
        }
    }

    template <class T>
    void matrix_multiply(const T* a, const T* b, std::size_t m, std::size_t n, std::size_t p, T* c) {
        std::fill_n(c, m * p, T(0));
        matrix_multiply_helper(a, b, m, n, p, c, m, n, p);
    }

    template <class T>
    void naive_matrix_multiply(const T* a, const T* b, std::size_t m, std::size_t n, std::size_t p, T* c) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                T sum = T(0);
                for (std::size_t k = 0; k < n; ++k) {
                    sum += a[i * n + k] * b[k * p + j];
                }
                c[i * p + j] = sum;
            }
        }
    }
}  // namespace ra::cache