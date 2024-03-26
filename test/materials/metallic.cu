#include "hittables/plane.hpp"
#include "lib/doctest/doctest.hpp"
#include "materials/metallic.hpp"
#include "test_utils.hpp"

TEST_SUITE ("Metallic Scatter Tests") {

    TEST_CASE ("Scattering on a plane's front face should reflect properly.") {

        Plane plane {
            {{ 0, 0, 0 }},
            {{ 0, 1, 0 }},
            new Metallic(RED, 0.0)
        };

        Ray incoming {
            {{ -1, 1, 0 }},
            {{ 1, -1, 0 }}
        };

        Hit expected {
            incoming,
            sqrt(2),
            {{ 0, 0, 0 }},
            {{ 0, 1, 0 }},
            true
        };

        OptionalHit opt = plane.checkHit(incoming);

        REQUIRE(opt.exists);

        Hit actual = opt.hit;

        CHECK_HITS(expected, actual);
    }

    TEST_CASE ("Scattering on a plane's rear face should reflect properly.") {

        Plane plane {
            {{ 0, 0, 0 }},
            {{ 0, 1, 0 }},
            new Metallic(RED, 0.0)
        };

        Ray incoming {
            {{ -1, -1, 0 }},
            {{ 1, 1, 0 }}
        };

        Hit expected {
            incoming,
            sqrt(2),
            {{ 0, 0, 0 }},
            {{ 0, -1, 0 }},
            false
        };

        OptionalHit opt = plane.checkHit(incoming);

        REQUIRE(opt.exists);

        Hit actual = opt.hit;

        CHECK_HITS(expected, actual);
    }
}
