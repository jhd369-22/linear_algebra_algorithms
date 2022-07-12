#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <ra/matrix_transpose.hpp>
#include <complex>

TEMPLATE_TEST_CASE("matrix transpose", "[matrix_transpose]", float, double, std::complex<double>) {
    namespace rc = ra::cache;

    SECTION("1 x 1 matrix") {
        constexpr std::size_t m = 1, n = 1;
        constexpr TestType a[m * n] = {TestType(1)};
        TestType b[m * n];
        TestType c[m * n];

        rc::naive_matrix_transpose(a, m, n, b);

        rc::matrix_transpose(a, m, n, c);

        for(std::size_t i = 0; i < n; ++i) {    // b is n x m matrix
            for(std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }

    SECTION("2 x 2 matrix") {
        constexpr std::size_t m = 2, n = 2;
        constexpr TestType a[m * n] = {TestType(1), TestType(2), TestType(3), TestType(4)};
        TestType b[m * n];
        TestType c[m * n];

        rc::naive_matrix_transpose(a, m, n, b);

        rc::matrix_transpose(a, m, n, c);

        for (std::size_t i = 0; i < n; ++i) {  // b is n x m matrix
            for (std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }

    SECTION("3 x 6 matrix"){
        constexpr std::size_t m = 3, n = 6;
        constexpr TestType a[m * n] = {
            TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9), TestType(10), TestType(11), TestType(12),
            TestType(13), TestType(14), TestType(15), TestType(16), TestType(17), TestType(18)
        };
        TestType b[m * n];
        TestType c[m * n];

        rc::naive_matrix_transpose(a, m, n, b);

        rc::matrix_transpose(a, m, n, c);

        for (std::size_t i = 0; i < n; ++i) {  // b is n x m matrix
            for (std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }

    SECTION("6 x 3 matrix"){
        constexpr std::size_t m = 6, n = 3;
        constexpr TestType a[m * n] = {
            TestType(1), TestType(2), TestType(3),
            TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9),
            TestType(10), TestType(11), TestType(12),
            TestType(13), TestType(14), TestType(15),
            TestType(16), TestType(17), TestType(18)
        };
        TestType b[m * n];
        TestType c[m * n];

        rc::naive_matrix_transpose(a, m, n, b);

        rc::matrix_transpose(a, m, n, c);

        for (std::size_t i = 0; i < n; ++i) {  // b is n x m matrix
            for (std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }

    SECTION("9 x 9 matrix"){
        constexpr std::size_t m = 9, n = 9;
        constexpr TestType a[m * n] = {TestType(0)};
        TestType b[m * n];
        TestType c[m * n];

        rc::naive_matrix_transpose(a, m, n, b);

        rc::matrix_transpose(a, m, n, c);

        for (std::size_t i = 0; i < n; ++i) {  // b is n x m matrix
            for (std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }
}
