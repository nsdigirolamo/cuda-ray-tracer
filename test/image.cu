#include "image.hpp"
#include "lib/doctest/doctest.hpp"
#include "test_utils.hpp"

TEST_SUITE ("Initialization Tests") {

    int expected_height = 1080;
    int expected_width = 1920;

    TEST_CASE ("Dimensions are as expected.") {

        Image image { expected_height, expected_width };

        int actual_height = image.height;
        int actual_width = image.width;

        CHECK(actual_height == expected_height);
        CHECK(actual_width == expected_width);
    }

    TEST_CASE ("Pixels are properly initialized.") {

        Image image { expected_height, expected_width };

        Color actual = image.getPixel(0, 0);
        Color expected {{ 1, 1, 1 }};

        CHECK_VECTORS(actual, expected);
    }

    TEST_CASE ("Setting pixels changes them permanently.") {

        Image image { expected_height, expected_width };

        Color expected = {{ 12, 34, 56 }};

        image.setPixel(0, 0, expected);

        Color actual = image.getPixel(0, 0);

        CHECK_VECTORS(actual, expected);
    }

    TEST_CASE ("Changing a returned pixel does not change the image.") {

        Image image { expected_height, expected_width };

        Color pixel = image.getPixel(0, 0);
        pixel[0] = 10;

        Color actual = image.getPixel(0, 0);
        Color expected {{ 1, 1, 1 }};

        CHECK_VECTORS(actual, expected);
    }

    TEST_CASE ("Changing a set pixel does not change the image.") {

        Image image { expected_height, expected_width };

        Color pixel {{ 12, 34, 56 }};
        image.setPixel(0, 0, pixel);

        pixel[0] = 10;

        Color actual = image.getPixel(0, 0);
        Color expected {{ 12, 34, 56 }};

        CHECK_VECTORS(actual, expected);
    }
}