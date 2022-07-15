#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <ra/fft.hpp>

TEMPLATE_TEST_CASE("Fast Fourier Transform", "[fft]", std::complex<float>, std::complex<double>) {
    namespace rc = ra::cache;

    SECTION("4 complex variables"){
        constexpr std::size_t n = 4;
        TestType a[n] = {TestType(1), TestType(2), TestType(3), TestType(4)};
        TestType b[n] = {TestType(10, 0), TestType(-2, 2), TestType(-2, 0), TestType(-2, -2)};

        rc::forward_fft(&a[0], n);

        for (std::size_t i = 0; i < n; ++i) {
            CHECK(a[i] == b[i]);
        }
    }

}