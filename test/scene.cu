#include "lib/doctest/doctest.hpp"
#include "scene.hpp"
#include "hittables/sphere.hpp"
#include "materials/diffuse.hpp"
#include "utils/test_utils.hpp"

TEST_SUITE ("Initialization Tests") {

    ListNode* expected_node = NULL;
    Hittable* expected_hittable = NULL;

    TEST_CASE ("Scene members should be NULL.") {

        Scene scene;

        CHECK(scene.head == expected_node);
        CHECK(scene.tail == expected_node);
    }

    TEST_CASE ("ListNode members should be NULL.") {

        ListNode node;

        CHECK(node.next == expected_node);
        CHECK(node.hittable == expected_hittable);
    }
}

TEST_SUITE ("Push Tests") {

    TEST_CASE ("Pushing a single element makes it head and tail.") {

        Scene scene;

        Sphere* sphere = new Sphere(
            {{ 0, 0, 0 }},
            1,
            new Diffuse(RED)
        );

        scene.push(sphere);

        CHECK(scene.head->hittable == sphere);
        CHECK(scene.tail->hittable == sphere);
    }

    TEST_CASE ("Pushing two elements makes the first head and second tail.") {

        Scene scene;

        Sphere* sphere1 = new Sphere(
            {{ 0, 0, 0 }},
            1,
            new Diffuse(RED)
        );

        Sphere* sphere2 = new Sphere(
            {{ 0, 0, 0 }},
            1,
            new Diffuse(RED)
        );

        scene.push(sphere1);
        scene.push(sphere2);

        CHECK(scene.head->hittable == sphere1);
        CHECK(scene.head->next->hittable == sphere2);
        CHECK(scene.tail->hittable == sphere2);
    }

    TEST_CASE ("Pushing three elements makes the first head and third tail.") {

        Scene scene;

        Sphere* sphere1 = new Sphere(
            {{ 0, 0, 0 }},
            1,
            new Diffuse(RED)
        );

        Sphere* sphere2 = new Sphere(
            {{ 0, 0, 0 }},
            1,
            new Diffuse(RED)
        );

        Sphere* sphere3 = new Sphere(
            {{ 0, 0, 0 }},
            1,
            new Diffuse(RED)
        );

        scene.push(sphere1);
        scene.push(sphere2);
        scene.push(sphere3);

        CHECK(scene.head->hittable == sphere1);
        CHECK(scene.head->next->hittable == sphere2);
        CHECK(scene.tail->hittable == sphere3);
    }
}