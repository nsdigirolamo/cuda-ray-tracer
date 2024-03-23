#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "lib/doctest/doctest.hpp"
#include "vector.hpp"

#define CHECK_VECTORS(lhs, rhs) \
    REQUIRE(lhs.height() == rhs.height()); \
    for (int i = 0; i < lhs.height(); ++i) { \
        CAPTURE(i); \
        CHECK(lhs[i] == doctest::Approx(rhs[i])); \
    } \

#endif
