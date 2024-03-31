#include "lib/doctest/doctest.hpp"
#include "utils/test_utils.hpp"
#include "interval.hpp"

TEST_SUITE ("Initialization Tests") {

    TEST_CASE ("Default initialization should fill the members properly.") {

        double expected_min = 1.0;
        double expected_max = 2.0;

        Interval interval { expected_min, expected_max };

        double actual_min = interval.min;
        double actual_max = interval.max;

        CHECK(expected_min == doctest::Approx(actual_min));
        CHECK(expected_max == doctest::Approx(actual_max));
    }
}
