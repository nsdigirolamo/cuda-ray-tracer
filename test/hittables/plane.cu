#include "hittables/plane.hpp"
#include "lib/doctest/doctest.hpp"
#include "materials/diffuse.hpp"
#include "utils/test_utils.hpp"

TEST_SUITE ("Plane Hit Tests") {

    Plane plane {
        {{ 0, 0, 0 }},
        {{ 0, 1, 0 }},
        new Diffuse(RED)
    };

    TEST_CASE ("No hit when the ray is orthogonal to the plane's normal.") {

        Ray ray {
            {{ 0, 1, 0 }},
            {{ 0, 0, 1 }}
        };

        Optional<Hit> opt = plane.checkHit(ray);

        CHECK_FALSE(opt.exists);
    }

    TEST_CASE ("No hit when the plane is behind the ray.") {

        Ray ray {
            {{ 0, 1, 0 }},
            {{ 0, 1, 0 }}
        };

        Optional<Hit> opt = plane.checkHit(ray);

        CHECK_FALSE(opt.exists);
    }

    TEST_CASE ("One hit when the ray is directed towards plane's front.") {

        Ray ray {
            {{ 0, 1, 0 }},
            {{ 0, -1, 0 }}
        };

        Optional<Hit> opt = plane.checkHit(ray);

        REQUIRE(opt.exists);

        Hit actual = opt.value;
        Hit expected = {
            &plane,
            ray,
            1,
            {{ 0, 0, 0 }},
            {{ 0, 1, 0 }},
            true
        };

        CHECK_HITS(actual, expected);
    }

    TEST_CASE ("One hit when the ray is directed towards the plane's rear.") {

        Ray ray {
            {{ 0, -1, 0 }},
            {{ 0, 1, 0 }}
        };

        Optional<Hit> opt = plane.checkHit(ray);

        REQUIRE(opt.exists);

        Hit actual = opt.value;
        Hit expected {
            &plane,
            ray,
            1,
            {{ 0, 0, 0 }},
            {{ 0, -1, 0 }},
            false
        };

        CHECK_HITS(actual, expected);
    }
}