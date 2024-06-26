#include "hittables/plane.hpp"
#include "lib/doctest/doctest.hpp"
#include "materials/metallic.hpp"
#include "utils/test_utils.hpp"

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
            &plane,
            incoming,
            sqrt(2),
            {{ 0, 0, 0 }},
            {{ 0, 1, 0 }},
            true
        };

        Optional<Hit> opt = plane.checkHit(incoming);

        REQUIRE(opt.exists);

        Hit actual = opt.value;

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
            &plane,
            incoming,
            sqrt(2),
            {{ 0, 0, 0 }},
            {{ 0, -1, 0 }},
            false
        };

        Optional<Hit> opt = plane.checkHit(incoming);

        REQUIRE(opt.exists);

        Hit actual = opt.value;

        CHECK_HITS(expected, actual);
    }
}
