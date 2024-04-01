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

TEST_SUITE ("Overlap Tests") {

    TEST_CASE ("The same interval overlaps with itself.") {

        Interval interval { 0.0, 1.0 };
        CHECK(IS_OVERLAPPING(interval, interval));
    }

    TEST_CASE ("Intervals overlap when a min is less than a max.") {

        Interval interval1 { 0.0, 1.0 };
        Interval interval2 { 0.8, 2.0 };
        CHECK(IS_OVERLAPPING(interval1, interval2));
    }

    TEST_CASE ("Intervals overlap when a min is less than a max (reversed).") {

        Interval interval1 { 0.0, 1.0 };
        Interval interval2 { 0.8, 2.0 };
        CHECK(IS_OVERLAPPING(interval2, interval1));
    }

    TEST_CASE ("Intervals overlap when a max is equal to a min.") {

        Interval interval1 { 0.0, 1.0 };
        Interval interval2 { 1.0, 2.0 };
        CHECK(IS_OVERLAPPING(interval1, interval2));
    }

    TEST_CASE ("Intervals overlap when a max is equal to a min (reversed).") {

        Interval interval1 { 0.0, 1.0 };
        Interval interval2 { 1.0, 2.0 };
        CHECK(IS_OVERLAPPING(interval2, interval1));
    }

    TEST_CASE ("Intervals don't overlap when a min is greater than a max.") {

        Interval interval1 { 0.0, 1.0 };
        Interval interval2 { 1.1, 2.0 };
        CHECK_FALSE(IS_OVERLAPPING(interval1, interval2));
    }

    TEST_CASE ("Intervals don't overlap when a min is greater than a max (reversed).") {

        Interval interval1 { 0.0, 1.0 };
        Interval interval2 { 1.1, 2.0 };
        CHECK_FALSE(IS_OVERLAPPING(interval2, interval1));
    }
}
