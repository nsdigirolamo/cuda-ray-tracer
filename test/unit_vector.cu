#include "lib/doctest/doctest.hpp"
#include "utils/test_utils.hpp"
#include "vector.hpp"

TEST_SUITE ("Initialization Tests") {

    TEST_CASE ("Default initialization should make vector have all zeroes and a 1 in the first element.") {

        UnitVector<3> actual;

        CHECK(1.0 == doctest::Approx(actual[0]));
        CHECK(0.0 == doctest::Approx(actual[1]));
        CHECK(0.0 == doctest::Approx(actual[2]));
    }

    TEST_CASE ("Array initialization should convert array to a vector and then make the length 1.") {

        double array[3] = { 1.0, 2.0, 3.0 };

        UnitVector<3> actual { array };
        double expected[3] = { 1.0 / sqrtf(14), 2.0 / sqrtf(14), 3.0 / sqrtf(14) };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }

        CHECK(1.0 == doctest::Approx(actual.length()));
    }

    TEST_CASE ("Vector initialization should return unit vector with length of 1.") {

        Vector<3> vector {{ 1.0, 2.0, 3.0 }};

        UnitVector<3> actual { vector };
        double expected[3] = { 1.0 / sqrtf(14), 2.0 / sqrtf(14), 3.0 / sqrtf(14) };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }

        CHECK(1.0 == doctest::Approx(actual.length()));
    }
}

TEST_SUITE ("Unit Vector Operations") {

    TEST_CASE ("The direction method should return the unit vector of a vector.") {

        Vector<3> vector {{ 1.0, 2.0, 3.0 }};

        UnitVector<3> actual = vector.direction();
        double expected[3] = { 1.0 / sqrtf(14), 2.0 / sqrtf(14), 3.0 / sqrtf(14) };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }

        CHECK(1.0 == doctest::Approx(actual.length()));
    }

    TEST_CASE ("The normalize function should return the unit vector of a vector.") {

        Vector<3> vector {{ 1.0, 2.0, 3.0 }};

        UnitVector<3> actual = normalize(vector);
        double expected[3] = { 1.0 / sqrtf(14), 2.0 / sqrtf(14), 3.0 / sqrtf(14) };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }

        CHECK(1.0 == doctest::Approx(actual.length()));
    }
}
