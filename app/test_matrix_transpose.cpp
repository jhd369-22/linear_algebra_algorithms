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

        rc::naive_matrix_transpose(&a[0], m, n, &b[0]);

        rc::matrix_transpose(a, m, n, &c[0]);

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

        rc::naive_matrix_transpose(&a[0], m, n, &b[0]);

        rc::matrix_transpose(a, m, n, &c[0]);

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

        rc::naive_matrix_transpose(&a[0], m, n, &b[0]);

        rc::matrix_transpose(a, m, n, &c[0]);

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

        rc::naive_matrix_transpose(&a[0], m, n, &b[0]);

        rc::matrix_transpose(a, m, n, &c[0]);

        for (std::size_t i = 0; i < n; ++i) {  // b is n x m matrix
            for (std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }

    SECTION("9 x 9 matrix"){
        constexpr std::size_t m = 9, n = 9;
        constexpr TestType a[m * n] = {
            TestType(1), TestType(2), TestType(3), TestType(4), TestType(5), TestType(6), TestType(7), TestType(8), TestType(9),
            TestType(10), TestType(11), TestType(12), TestType(13), TestType(14), TestType(15), TestType(16), TestType(17), TestType(18),
            TestType(19), TestType(20), TestType(21), TestType(22), TestType(23), TestType(24), TestType(25), TestType(26), TestType(27),
            TestType(28), TestType(29), TestType(30), TestType(31), TestType(32), TestType(33), TestType(34), TestType(35), TestType(36),
            TestType(37), TestType(38), TestType(39), TestType(40), TestType(41), TestType(42), TestType(43), TestType(44), TestType(45),
            TestType(46), TestType(47), TestType(48), TestType(49), TestType(50), TestType(51), TestType(52), TestType(53), TestType(54),
            TestType(55), TestType(56), TestType(57), TestType(58), TestType(59), TestType(60), TestType(61), TestType(62), TestType(63),
            TestType(64), TestType(65), TestType(66), TestType(67), TestType(68), TestType(69), TestType(70), TestType(71), TestType(72),
            TestType(73), TestType(74), TestType(75), TestType(76), TestType(77), TestType(78), TestType(79), TestType(80), TestType(81)
        };
        TestType b[m * n];
        TestType c[m * n];

        rc::naive_matrix_transpose(&a[0], m, n, &b[0]);

        rc::matrix_transpose(a, m, n, &c[0]);

        for (std::size_t i = 0; i < n; ++i) {  // b is n x m matrix
            for (std::size_t j = 0; j < m; ++j) {
                CHECK(b[i * m + j] == c[i * m + j]);
            }
        }
    }
}
