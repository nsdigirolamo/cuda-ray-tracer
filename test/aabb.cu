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

TEST_SUITE ("Hit Tests") {

    TEST_CASE ("Direct hit should hit.") {

        AABB aabb {
            NULL,
            { 0.0, 3.0 },
            { 0.0, 3.0 },
            { 2.0, 3.0 }
        };

        Ray ray {
            {{ 1.0, 1.0, 1.0 }},
            {{ 0.0, 0.0, 1.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("Direct hit with some minor offsets should hit.") {

        AABB aabb {
            NULL,
            { 0.0, 3.0 },
            { 0.0, 3.0 },
            { 2.0, 3.0 }
        };

        Ray ray {
            {{ 1.0, 1.0, 1.0 }},
            {{ 0.01, 0.01, 0.8}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("Direct hit on the z axis should hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { -1.0, 1.0 },
            { 1.0, 2.0 }
        };

        Ray ray {
            {{ 0.0, 0.0, 0.0 }},
            {{ 0.0, 0.0, 1.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("Direct hit on the x axis should hit.") {

        AABB aabb {
            NULL,
            { 1.0, 2.0 },
            { -1.0, 1.0 },
            { -1.0, 1.0 }
        };

        Ray ray {
            {{ 0.0, 0.0, 0.0 }},
            {{ 1.0, 0.0, 0.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("Direct hit on the y axis should hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { 1.0, 2.0 },
            { -1.0, 1.0 }
        };

        Ray ray {
            {{ 0.0, 0.0, 0.0 }},
            {{ 0.0, 1.0, 0.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("Direct hit on a corner should hit.") {

        AABB aabb {
            NULL,
            { 1.0, 2.0 },
            { 1.0, 2.0 },
            { 1.0, 2.0 }
        };

        Ray ray {
            {{ 0.0, 0.0, 0.0 }},
            {{ 1.0, 1.0, 1.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("A ray within the AABB should hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { -1.0, 1.0 },
            { -1.0, 1.0 },
        };

        Ray ray {
            {{ 0.0, 0.0, 0.0 }},
            {{ 1.0, 1.0, 1.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("A ray within its origin on a border should hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { -1.0, 1.0 },
            { -1.0, 1.0 },
        };

        Ray ray {
            {{ 0.0, 0.0, -1.0 }},
            {{ 1.0, 1.0, 1.0}}
        };

        CHECK(aabb.isHit(ray));
    }

    TEST_CASE ("A ray parallel to a face should not hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { -1.0, 1.0 },
            { -1.0, 1.0 },
        };

        Ray ray {
            {{ 0.0, 0.0, -1.1 }},
            {{ 0.0, 1.0, 0.0}}
        };

        CHECK_FALSE(aabb.isHit(ray));
    }

    TEST_CASE ("A ray that is directed away from a face should not hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { -1.0, 1.0 },
            { -1.0, 1.0 },
        };

        Ray ray {
            {{ 0.0, 0.0, -1.1 }},
            {{ 0.0, 0.0, -1.0}}
        };

        CHECK_FALSE(aabb.isHit(ray));
    }

    TEST_CASE ("A ray that misses an AABB should not hit.") {

        AABB aabb {
            NULL,
            { -1.0, 1.0 },
            { -1.0, 1.0 },
            { -1.0, 1.0 },
        };

        Ray ray {
            {{ 0.0, 0.0, -5.0 }},
            {{ 0.0, 1.0, 1.0}}
        };

        CHECK_FALSE(aabb.isHit(ray));
    }
}
