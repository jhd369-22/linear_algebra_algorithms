#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <complex>
#include <ra/matrix_multiply.hpp>

TEMPLATE_TEST_CASE("matrix multiply", "[matrix_multiply]", int, double, std::complex<double>){
    namespace rc = ra::cache;

    SECTION("1 x 1 multiplies 1 x 1 matrixs") {
        constexpr std::size_t m = 1, n = 1, p = 1;

        constexpr TestType a[m * n] = {TestType(1)};
        constexpr TestType b[n * p] = {TestType(2)};
        TestType c[m * p];
        TestType d[m * p];

        rc::matrix_multiply(&a[0], &b[0], m, n, p, &c[0]);
        rc::naive_matrix_multiply(&a[0], &b[0], m, n, p, &d[0]);

        for(std::size_t i = 0; i < m; ++i) {    // c, d are m x p matrix
            for(std::size_t j = 0; j < p; ++j) {
                CHECK(c[i * p + j] == d[i * p + j]);
            }
        }
    }

    SECTION("2 x 2 multiplies 2 x 2 matrixes") {
        constexpr std::size_t m = 2, n = 2, p = 2;

        constexpr TestType a[m * n] = {TestType(1), TestType(2), TestType(3), TestType(4)};
        constexpr TestType b[n * p] = {TestType(5), TestType(6), TestType(7), TestType(8)};
        TestType c[m * p];
        TestType d[m * p];

        rc::matrix_multiply(&a[0], &b[0], m, n, p, &c[0]);
        rc::naive_matrix_multiply(&a[0], &b[0], m, n, p, &d[0]);

        for(std::size_t i = 0; i < m; ++i) {    // c, d are m x p matrix
            for(std::size_t j = 0; j < p; ++j) {
                CHECK(c[i * p + j] == d[i * p + j]);
            }
        }
    }

    SECTION("3 x 6 multiplies 6 x 6 matrixes") {
        constexpr std::size_t m = 3, n = 6, p = 6;
        constexpr TestType a[m * n] = {
            TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9), TestType(10), TestType(11), TestType(12),
            TestType(13), TestType(14), TestType(15), TestType(16), TestType(17), TestType(18)
        };
        constexpr TestType b[n * p] = {
            TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9), TestType(10), TestType(11), TestType(12),
            TestType(13), TestType(14), TestType(15), TestType(16), TestType(17), TestType(18),
            TestType(19), TestType(20), TestType(21), TestType(22), TestType(23), TestType(24),
            TestType(25), TestType(26), TestType(27), TestType(28), TestType(29), TestType(30),
            TestType(31), TestType(32), TestType(33), TestType(34), TestType(35), TestType(36)
        };
        TestType c[m * p];
        TestType d[m * p];

        rc::matrix_multiply(&a[0], &b[0], m, n, p, &c[0]);
        rc::naive_matrix_multiply(&a[0], &b[0], m, n, p, &d[0]);

        for (std::size_t i = 0; i < m; ++i) {  // c, d are m x p matrix
            for (std::size_t j = 0; j < p; ++j) {
                CHECK(c[i * p + j] == d[i * p + j]);
            }
        }
    }

    SECTION("6 x 3 multiplies 3 x 3 matrixes") {
        constexpr std::size_t m = 6, n = 3, p = 3;
        constexpr TestType a[m * n] = {
            TestType(1), TestType(2), TestType(3),
            TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9),
            TestType(10), TestType(11), TestType(12),
            TestType(13), TestType(14), TestType(15),
            TestType(16), TestType(17), TestType(18)
        };
        constexpr TestType b[n * p] = {
            TestType(1), TestType(2), TestType(3),
            TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9)
        };
        TestType c[m * p];
        TestType d[m * p];

        rc::matrix_multiply(&a[0], &b[0], m, n, p, &c[0]);
        rc::naive_matrix_multiply(&a[0], &b[0], m, n, p, &d[0]);

        for (std::size_t i = 0; i < m; ++i) {  // c, d are m x p matrix
            for (std::size_t j = 0; j < p; ++j) {
                CHECK(c[i * p + j] == d[i * p + j]);
            }
        }
    }

    SECTION("6 x 9 multiplies 9 x 6 matrixes") {
        constexpr std::size_t m = 6, n = 9, p = 6;
        constexpr TestType a[m * n] = {
            TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6),
            TestType(7), TestType(8), TestType(9), TestType(10), TestType(11), TestType(12),
            TestType(13), TestType(14), TestType(15), TestType(16), TestType(17), TestType(18),
            TestType(19), TestType(20), TestType(21), TestType(22), TestType(23), TestType(24),
            TestType(25), TestType(26), TestType(27), TestType(28), TestType(29), TestType(30),
            TestType(31), TestType(32), TestType(33), TestType(34), TestType(35), TestType(36),
            TestType(37), TestType(38), TestType(39), TestType(40), TestType(41), TestType(42),
            TestType(43), TestType(44), TestType(45), TestType(46), TestType(47), TestType(48),
            TestType(49), TestType(50), TestType(51), TestType(52), TestType(53), TestType(54)
        };
        constexpr TestType b[n * p] = {
            TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6), TestType(7), TestType(8), TestType(9),
            TestType(10), TestType(11), TestType(12), TestType(13), TestType(14), TestType(15), TestType(16), TestType(17), TestType(18),
            TestType(19), TestType(20), TestType(21), TestType(22), TestType(23), TestType(24), TestType(25), TestType(26), TestType(27),
            TestType(28), TestType(29), TestType(30), TestType(31), TestType(32), TestType(33), TestType(34), TestType(35), TestType(36),
            TestType(37), TestType(38), TestType(39), TestType(40), TestType(41), TestType(42), TestType(43), TestType(44), TestType(45),
            TestType(46), TestType(47), TestType(48), TestType(49), TestType(50), TestType(51), TestType(52), TestType(53), TestType(54)
        };
        TestType c[m * p];
        TestType d[m * p];

        rc::matrix_multiply(&a[0], &b[0], m, n, p, &c[0]);
        rc::naive_matrix_multiply(&a[0], &b[0], m, n, p, &d[0]);

        for (std::size_t i = 0; i < m; ++i) {  // c, d are m x p matrix
            for (std::size_t j = 0; j < p; ++j) {
                CHECK(c[i * p + j] == d[i * p + j]);
            }
        }
    }    
}
