#include "camera.hpp"
#include "lib/doctest/doctest.hpp"
#include "utils/test_utils.hpp"

TEST_SUITE ("Initialization Tests") {

    Point expected_origin {{ 0, 0, 0 }};
    int expected_image_height = 1080;
    int expected_image_width = 1920;
    double expected_hfov = 90;
    Point focal_point {{ 0, 0, 1 }};
    double focal_angle = 1.0;

    Camera actual {
        expected_origin,
        expected_image_height,
        expected_image_width,
        expected_hfov,
        focal_point,
        focal_angle
    };

    TEST_CASE ("Origin is as expected.") {

        CHECK_VECTORS(actual.getOrigin(), expected_origin);
    }

    TEST_CASE ("Image dimensions are as expected.") {

        CHECK(actual.getImageHeight() == expected_image_height);
        CHECK(actual.getImageWidth() == expected_image_width);
    }

    double expected_vfov = 58.715507;

    TEST_CASE ("Fields of view are as expected.") {

        CHECK(actual.getVfov() == doctest::Approx(expected_vfov));
        CHECK(actual.getHfov() == doctest::Approx(expected_hfov));
    }

    double expected_focal_distance = 1;
    double expected_view_width = 2.0;
    double expected_view_height = expected_view_width * ((double)(expected_image_height) / (double)(expected_image_width));
    double expected_lens_radius = 0.00872;

    TEST_CASE ("View dimensions are as expected.") {

        CHECK(actual.getViewHeight() == doctest::Approx(expected_view_height));
        CHECK(actual.getViewWidth() == doctest::Approx(expected_view_width));
        CHECK(actual.getFocalDistance() == doctest::Approx(expected_focal_distance));
        CHECK(actual.getLensRadius() == doctest::Approx(expected_lens_radius));
    }

    double expected_pixel_width = expected_view_width / expected_image_width;
    double expected_pixel_height = expected_view_height / expected_image_height;

    TEST_CASE ("Pixel dimensions are as expected.") {

        CHECK(actual.getPixelHeight() == doctest::Approx(expected_pixel_height));
        CHECK(actual.getPixelWidth() == doctest::Approx(expected_pixel_width));
    }

    UnitVector<3> expected_up_direction {{0, 1, 0}};
    UnitVector<3> expected_view_direction {{0, 0, 1}};
    UnitVector<3> expected_view_top {{0, 1, 0}};
    UnitVector<3> expected_view_left {{-1, 0, 0}};

    TEST_CASE ("View directions are as expected") {

        CHECK_VECTORS(actual.getUpDirection(), expected_up_direction);
        CHECK_VECTORS(actual.getViewDirection(), expected_view_direction);
        CHECK_VECTORS(actual.getViewTop(), expected_view_top);
        CHECK_VECTORS(actual.getViewLeft(), expected_view_left);
    }
}
