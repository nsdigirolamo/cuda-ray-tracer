#include "hittables/sphere.hpp"
#include "lib/doctest/doctest.hpp"
#include "materials/diffuse.hpp"
#include "test_utils.hpp"

TEST_SUITE ("Sphere Hit Tests") {

    Sphere sphere {
        {{ 0, 0, 10 }},
        1.0,
        new Diffuse(RED)
    };

    TEST_CASE ("No hit when the sphere is behind the ray.") {

        Ray ray {
            {{ 0, 0, 0 }},
            {{ 0, 0, -1 }}
        };

        OptionalHit opt = sphere.checkHit(ray);

        CHECK_FALSE(opt.exists);
    }

    TEST_CASE ("No hit when the ray misses the sphere completely.") {

        Ray ray {
            {{ 10, 0, 0 }},
            {{ 0, 0, 1 }}
        };

        OptionalHit opt = sphere.checkHit(ray);

        CHECK_FALSE(opt.exists);
    }

    TEST_CASE ("No hit when the hit occurs below the minimum hit distance.") {

        Ray ray {
            {{ 0, 0, 9.00001 }},
            {{ 0, 0, -1 }}
        };

        OptionalHit opt = sphere.checkHit(ray);

        CHECK_FALSE(opt.exists);
    }

    TEST_CASE ("One hit when the ray is tangent to the sphere.") {

        Ray ray {
            {{ 1, 0, 0 }},
            {{ 0, 0, 1 }}
        };

        OptionalHit opt = sphere.checkHit(ray);

        REQUIRE(opt.exists);

        Hit actual = opt.hit;
        Hit expected {
            ray,
            10,
            {{ 1, 0, 10 }},
            {{ 1, 0, 0 }},
            true
        };

        CHECK_HITS(actual, expected);
    }

    TEST_CASE ("One hit when the ray is within the sphere.") {

        Ray ray {
            {{ 0, 0, 10 }},
            {{ 0, 0, 1 }}
        };

        OptionalHit opt = sphere.checkHit(ray);

        REQUIRE(opt.exists);

        Hit actual = opt.hit;
        Hit expected {
            ray,
            1,
            {{ 0, 0, 11 }},
            {{ 0, 0, -1 }},
            false
        };

        CHECK_HITS(actual, expected);
    }

    TEST_CASE ("Closest hit is returned when the ray could hit the sphere twice.") {

        Ray ray {
            {{ 0, 0, 0 }},
            {{ 0, 0, 1 }}
        };

        OptionalHit opt = sphere.checkHit(ray);

        REQUIRE(opt.exists);

        Hit actual = opt.hit;
        Hit expected {
            ray,
            9,
            {{ 0, 0, 9 }},
            {{ 0, 0, -1 }},
            true
        };

        CHECK_HITS(actual, expected);
    }
}