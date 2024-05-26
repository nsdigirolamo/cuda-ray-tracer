#include "lib/doctest/doctest.hpp"
#include "utils/test_utils.hpp"
#include "aabb.hpp"
#include "hittables/sphere.hpp"
#include "materials/diffuse.hpp"

TEST_SUITE ("findAxisToSplit Tests") {

    TEST_CASE ("Two objects on the x axis are split on the x axis.") {

        Sphere* sphere1 = new Sphere(
            {{ 0, 0, 0 }},
            1.0,
            new Diffuse({ FIREBRICK })
        );

        Sphere* sphere2 = new Sphere(
            {{ 0, 0, 0 }},
            1.0,
            new Diffuse({ FIREBRICK })
        );

    }

}