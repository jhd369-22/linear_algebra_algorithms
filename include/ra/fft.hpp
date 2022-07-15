#include <algorithm>  // std::copy
#include <cmath>
#include <cstdio>
#include <numbers>  // Numerics library: std::number::pi_v, std::numbers::e_v
#include <ra/matrix_transpose.hpp>
#include <complex>

namespace ra::cache {

    template <class T>
    void forward_fft(T* x, std::size_t n) {
        if (n <= 1) return;

        // divide
        T* even = new T[n / 2]();
        T* odd = new T[n / 2]();
        for (std::size_t i = 0; i < n / 2; ++i) {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }

        // conquer
        forward_fft(even, n / 2);
        forward_fft(odd, n / 2);

        // combine
        for (size_t k = 0; k < n / 2; ++k) {
            T t = std::exp(T(0, -2 * std::numbers::pi_v<double> * k / n)) * odd[k];
            x[k] = even[k] + t;
            x[k + n / 2] = even[k] - t;
        }
    }

}  // namespace ra::cache