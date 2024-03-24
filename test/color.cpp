#include "lib/doctest/doctest.hpp"
#include "test_utils.hpp"
#include "color.hpp"

TEST_SUITE ("Initialization Tests") {

    TEST_CASE ("Default initialization should make color values all zeroes.") {

        Color actual;

        for (int i = 0; i < 3; ++i) {
            CHECK(0.0 == doctest::Approx(actual[i]));
        }
    }

    TEST_CASE ("Array initialization should make color have the array's values.") {

        double expected[3] { 1.2, 3.4, 5.6 };
        Color actual { expected };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }
    }

    TEST_CASE ("Vector initialization should make color have the vector's values.") {

        Vector<3> expected {{ 1.2, 3.4, 5.6 }};
        Color actual { expected };

        for (int i = 0; i < 3; ++i) {
            CHECK(expected[i] == doctest::Approx(actual[i]));
        }
    }

    TEST_CASE ("Initialization with three doubles should make color have those three values.") {

        double expected_r = 1.2;
        double expected_g = 3.4;
        double expected_b = 5.6;

        Color actual { expected_r, expected_g, expected_b };

        CHECK(expected_r == doctest::Approx(actual[0]));
        CHECK(expected_g == doctest::Approx(actual[1]));
        CHECK(expected_b == doctest::Approx(actual[2])); 
    }

    TEST_CASE ("Initialization with hex value should make color have appropriate values.") {

        Color expected_beige = {{245.0 / 256, 245.0 / 256, 220.0 / 256}};
        CHECK_VECTORS(expected_beige, BEIGE);
    }
}