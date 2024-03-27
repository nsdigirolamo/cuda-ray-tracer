#include "curand_kernel.h"

#include "camera.hpp"
#include "hittables/hittable.hpp"
#include "hittables/plane.hpp"
#include "hittables/sphere.hpp"
#include "materials/diffuse.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/rand_utils.hpp"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ Hittable** hittables;
__device__ int hittables_count;

Camera::Camera(
    const Point& origin,
    const int image_height,
    const int image_width,
    const double hfov,
    const Point& focal_point
) {

    this->origin = origin;

    this->image_height = image_height;
    this->image_width = image_width;

    this->focal_distance = (focal_point - origin).length();
    this->view_width = 2.0 * this->focal_distance * tan(degsToRads(hfov) / 2.0);
    this->view_height = this->view_width * image_height / image_width;

    this->vfov = radsToDegs(atan(this->view_height / 2.0 / this->focal_distance) * 2.0);
    this->hfov = hfov;

    this->pixel_height = this->view_height / image_height;
    this->pixel_width = this->view_width / image_width;

    this->up_direction = {{ 0, 1, 0 }};
    this->view_direction = focal_point - origin;
    this->view_left = CROSS(this->view_direction, up_direction);
    this->view_top = CROSS(this->view_left, view_direction);
}

Point Camera::getOrigin () const {
    return this->origin;
}

int Camera::getImageHeight () const {
    return this->image_height;
}

int Camera::getImageWidth () const {
    return this->image_width;
}

double Camera::getVfov () const {
    return this->vfov;
}

double Camera::getHfov () const {
    return this->hfov;
}

double Camera::getViewHeight () const {
    return this->view_height;
}

double Camera::getViewWidth () const {
    return this->view_width;
}

double Camera::getFocalDistance () const {
    return this->focal_distance;
}

double Camera::getPixelHeight () const {
    return this->pixel_height;
}

double Camera::getPixelWidth () const {
    return this->pixel_width;
}

UnitVector<3> Camera::getUpDirection () const {
    return this->up_direction;
}

UnitVector<3> Camera::getViewDirection () const {
    return this->view_direction;
}

UnitVector<3> Camera::getViewTop () const {
    return this->view_top;
}

UnitVector<3> Camera::getViewLeft () const {
    return this->view_left;
}

__device__ Point Camera::generateRayOrigin () const {

    /**
     * TODO: Implement random offset
     *
     * return this->location +
     *  (offset[0] * this->focal_radius * this->view_horizontal) +
     *  (offset[1] * this->focal_radius * this->view_vertical);
     *
     */

    return this->origin;
}

__device__ Point Camera::calculatePixelLocation () const {

    int row = blockIdx.y;
    int col = blockIdx.x;

    double x = -0.5 * this->view_width + col * this->pixel_width + 0.5 * this->pixel_width;
    double y = 0.5 * this->view_height - row * this->pixel_height - 0.5 * this->pixel_height;

    /**
     * TODO: Implement random offset.
     *
     * Vector<2> offset = randomInUnitCircle();
     * x += offset[0] * this->pixel_width;
     * y += offset[1] * this->pixel_height;
     *
     */

    return
        this->view_left * x +
        this->view_top * y +
        this->view_direction * this->focal_distance;
}

__device__ Ray Camera::getInitialRay () const {

    Point origin = this->generateRayOrigin();
    UnitVector<3> direction = this->calculatePixelLocation() - this->origin;
    return { origin, direction };
}

Image Camera::render (
    const int samples_per_pixel,
    const int bounces_per_sample
) const {

    int pixel_count = this->image_width * this->image_height;
    int sample_count = pixel_count * samples_per_pixel;

    // Allocate space for hittables and create them as needed.

    setupHittables<<<1, 1>>>();
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Allocate space for pixel samples.

    Color* samples;
    size_t samples_size = sizeof(Color) * sample_count;
    CUDA_ERROR_CHECK( cudaMalloc(&samples, samples_size) );

    // Allocate space for and setup the device's random states.

    curandState* device_states;

    dim3 rand_grid_dims(this->image_width, this->image_height);
    dim3 rand_block_dims(1);

    CUDA_ERROR_CHECK( cudaMalloc(&device_states, sizeof(curandState) * pixel_count) );
    setupRandoms<<<rand_grid_dims, rand_block_dims>>>(device_states, time(NULL));
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Trace each sample.

    dim3 trace_grid_dims(this->image_width, this->image_height);
    dim3 trace_block_dims(samples_per_pixel);

    traceSample<<<trace_grid_dims, trace_block_dims>>>(*this, samples, bounces_per_sample, device_states);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Reduce the samples down to a single pixel color.

    int reduce_size = 32;
    dim3 reduce_grid_dims(
        (this->image_width + reduce_size - 1) / reduce_size,
        (this->image_height + reduce_size - 1) / reduce_size
    );
    dim3 reduce_block_dims(reduce_size, reduce_size);

    Color* reduced_samples;
    size_t reduced_samples_size = sizeof(Color) * this->image_width * this->image_height;
    CUDA_ERROR_CHECK( cudaMalloc(&reduced_samples, reduced_samples_size) );

    reduceSamples<<<reduce_grid_dims, reduce_block_dims>>>(samples, this->image_height, this->image_width, reduced_samples, samples_per_pixel);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Copy the reduced pixels to host memory and return.

    Image image { this->image_height, this->image_width };
    CUDA_ERROR_CHECK( cudaMemcpy(image.pixels, reduced_samples, reduced_samples_size, cudaMemcpyDeviceToHost) );

    return image;
}

__global__ void setupHittables () {

    Sphere* sphere = new Sphere(
        {{ 0, 0, 0 }},
        1,
        new Diffuse(FIREBRICK)
    );

    Plane* plane = new Plane(
        {{ 0, -1, 0 }},
        {{ 0, 1, 0 }},
        new Diffuse(SLATEGRAY)
    );

    hittables_count = 2;
    hittables = (Hittable**)(malloc(sizeof(Hittable*) * hittables_count));

    hittables[0] = sphere;
    hittables[1] = plane;
}

__global__ void setupRandoms (curandState* device_states, uint64_t seed) {

    int width = gridDim.x;

    int row = blockIdx.y;
    int col = blockIdx.x;

    int thread_id = (row * width) + col;

    curand_init(seed, thread_id, 0, &(device_states[0]));
}

__global__ void traceSample (
    Camera camera,
    Color* samples,
    const int bounces_per_sample,
    curandState* device_states
) {

    int height = gridDim.y;
    int width = gridDim.x;

    int row = blockIdx.y;
    int col = blockIdx.x;
    int depth = threadIdx.x;

    curandState local_state = device_states[(row * width) + col];

    Ray ray = camera.getInitialRay();
    Color traced { 1.0, 1.0, 1.0 };

    int bounce = 0;

    while (bounce < bounces_per_sample) {

        OptionalHit closest;
        Hittable* hittable = NULL;

        for (int i = 0; i < hittables_count; ++i) {

            OptionalHit opt = hittables[i]->checkHit(ray);

            bool hit_is_closest =
                opt.exists &&
                closest.exists &&
                opt.hit.distance < closest.hit.distance;

            if (opt.exists && (!closest.exists || hit_is_closest)) {
                closest = opt;
                hittable = hittables[i];
            }
        }

        if (closest.exists) {

            Hit hit = closest.hit;
            Material* material = hittable->getMaterial();

            traced = hadamard(traced, material->getColor());
            ray = material->scatter(hit, &local_state);

        } else {

            traced = hadamard(traced, (Color)(SKY));
            break;
        }

        ++bounce;
    }

    samples[(depth * width * height) + (row * width) + col] = traced;
}

__global__ void reduceSamples (
    Color* samples,
    const int image_height,
    const int image_width,
    Color* reduced_samples,
    int samples_per_pixel
) {

    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (image_height <= row || image_width <= col) return;

    reduced_samples[(row * image_width) + col] = { 0, 0, 0 };

    for (int i = 0; i < samples_per_pixel; ++i) {
        Color sample = samples[(i * image_width * image_height) + (row * image_width) + col];
        reduced_samples[(row * image_width) + col] += sample;
    }

    reduced_samples[(row * image_width) + col] /= samples_per_pixel;
}
