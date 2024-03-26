#include "hittables/hittable.hpp"
#include "hittables/plane.hpp"
#include "hittables/sphere.hpp"

#include "materials/diffuse.hpp"

#include "camera.hpp"
#include "image.hpp"

int main () {

    /**

    Sphere* host_sphere = new Sphere(
        {{ 0, 0, 0 }},
        1,
        new Diffuse(ANTIQUEWHITE)
    );

    Sphere* device_sphere;
    cudaMalloc(&device_sphere, sizeof(Sphere));
    cudaMemcpy(device_sphere, host_sphere, sizeof(Sphere), cudaMemcpyHostToDevice);

    Plane* host_plane = new Plane(
        {{ 0, -1, 0 }},
        {{ 0, 1, 0 }},
        new Diffuse(ANTIQUEWHITE)
    );

    Plane* device_plane;
    cudaMalloc(&device_plane, sizeof(Plane));
    cudaMemcpy(device_plane, host_plane, sizeof(Plane), cudaMemcpyHostToDevice);

    int hittable_count = 2;
    Hittable** hittables = (Hittable**)(malloc(sizeof(Hittable*) * hittable_count));
    cudaMalloc(&hittables, sizeof(Hittable*) * hittable_count);

    hittables[0] = device_sphere;
    hittables[1] = device_plane;

    Camera camera {
        {{ 0, 0, -5 }},
        1080,
        1920,
        90,
        {{ 0, 0, 0 }}
    };

    Image image = camera.render(hittables, hittable_count, 50, 50);

    */
}