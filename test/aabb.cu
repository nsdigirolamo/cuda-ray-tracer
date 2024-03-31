#include "lib/doctest/doctest.hpp"
#include "utils/test_utils.hpp"
#include "aabb.hpp"
#include "hittables/sphere.hpp"
#include "materials/diffuse.hpp"

TEST_SUITE ("Initialization Tests") {

    TEST_CASE ("Default initialization sets up members properly.") {

        Sphere* expected_bounded = new Sphere(
            {{ 0, 0, 0 }},
            1.0,
            new Diffuse(RED)
        );

        Interval expected_x { 1.0, 2.0 };
        Interval expected_y { 3.0, 4.0 };
        Interval expected_z { 5.0, 6.0 };

        AABB actual = {
            expected_bounded,
            expected_x,
            expected_y,
            expected_z
        };

        CHECK(expected_bounded == actual.bounded);
        CHECK_INTERVALS(expected_x, actual.x_interval);
        CHECK_INTERVALS(expected_y, actual.y_interval);
        CHECK_INTERVALS(expected_z, actual.z_interval);
    }
}

TEST_SUITE ("Method Tests") {

    TEST_CASE("getInterval returns intervals properly.") {

        Sphere* expected_bounded = new Sphere(
            {{ 0, 0, 0 }},
            1.0,
            new Diffuse(RED)
        );

        Interval expected_x { 1.0, 2.0 };
        Interval expected_y { 3.0, 4.0 };
        Interval expected_z { 5.0, 6.0 };

        AABB actual = {
            expected_bounded,
            expected_x,
            expected_y,
            expected_z
        };

        CHECK_INTERVALS(expected_x, actual.getInterval(0));
        CHECK_INTERVALS(expected_y, actual.getInterval(1));
        CHECK_INTERVALS(expected_z, actual.getInterval(2));
        CHECK_INTERVALS(expected_z, actual.getInterval(3));
        CHECK_INTERVALS(expected_z, actual.getInterval(4));
        CHECK_INTERVALS(expected_z, actual.getInterval(-1));
    }
}
