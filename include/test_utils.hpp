#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include "lib/doctest/doctest.hpp"
#include "vector.hpp"

#define CHECK_VECTORS(lhs, rhs) \
    REQUIRE(lhs.height() == rhs.height()); \
    for (int i = 0; i < lhs.height(); ++i) { \
        CAPTURE(i); \
        CHECK(lhs[i] == doctest::Approx(rhs[i])); \
    }

#define CHECK_RAYS(lhs, rhs) \
    CHECK_VECTORS(lhs.origin, rhs.origin); \
    CHECK_VECTORS(lhs.direction, rhs.direction);

#define CHECK_HITS(lhs, rhs) \
    CHECK_RAYS(lhs.incoming, rhs.incoming); \
    CHECK(lhs.distance == doctest::Approx(rhs.distance)); \
    CHECK_VECTORS(lhs.origin, rhs.origin); \
    CHECK_VECTORS(lhs.surface_normal, rhs.surface_normal); \
    CHECK(lhs.is_front == rhs.is_front);

#define CHECK_OPTIONAL_HITS(lhs, rhs) \
    REQUIRE(lhs.exists == rhs.exists); \
    if (lhs.exists) CHECK_HITS(lhs, rhs);

#endif
