#include "curand_kernel.h"

#include "camera.hpp"
#include "hittables/hittable.hpp"
#include "hittables/plane.hpp"
#include "hittables/sphere.hpp"
#include "materials/diffuse.hpp"
#include "materials/metallic.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/rand_utils.hpp"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16

__device__ Hittable** hittables;
__device__ int hittables_count;

Camera::Camera(
    const Point& origin,
    const int image_height,
    const int image_width,
    const double hfov,
    const Point& focal_point,
    const double focal_angle
) {

    this->origin = origin;

    this->image_height = image_height;
    this->image_width = image_width;

    this->focal_distance = ((Vector<3>)(focal_point - origin)).length();

    this->lens_radius = this->focal_distance * tan(degsToRads(focal_angle) / 2.0);

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

double Camera::getLensRadius () const {
    return this->lens_radius;
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

__device__ Point Camera::generateRayOrigin (curandState* state) const {

       Vector<2> offset = randomInUnitCircle(state);
       return this->origin +
           offset[0] * this->lens_radius * this->view_left +
           offset[1] * this->lens_radius * this->view_top;
}

__device__ Point Camera::calculatePixelLocation (curandState* state) const {

    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    double x = -0.5 * this->view_width + col * this->pixel_width + 0.5 * this->pixel_width;
    double y = 0.5 * this->view_height - row * this->pixel_height - 0.5 * this->pixel_height;

    Vector<2> offset = randomInUnitCircle(state);
    x += offset[0] * this->pixel_width;
    y += offset[1] * this->pixel_height;

    return
        this->view_left * x +
        this->view_top * y +
        this->view_direction * this->focal_distance;
}

__device__ Ray Camera::getInitialRay (curandState* state) const {

    Point ray_origin = generateRayOrigin(state);
    Point ray_direction = calculatePixelLocation(state) - ray_origin;

    return { ray_origin, ray_direction };
}

Image Camera::render (const int sample_count, const int max_bounces) const {

    cudaError_t e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);

    int dim = BLOCK_DIM;

    dim3 block_dims(dim, dim);

    dim3 grid_dims(
        (this->image_width + dim - 1) / dim,
        (this->image_height + dim - 1) / dim
    );

    // Allocate space for hittables.

    setupHittables<<<1, 1>>>();
    e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Allocate space for image and initialize its pixels.

    Color* device_image;
    size_t device_image_size = sizeof(Color) * this->image_width * this->image_height;
    CUDA_ERROR_CHECK( cudaMalloc(&device_image, device_image_size) );

    setupImage<<<grid_dims, block_dims>>>(*this, device_image);
    e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Allocate space for random states and then initialize them.

    curandState* states;
    size_t states_size = sizeof(curandState) * this->image_width * this->image_height;
    CUDA_ERROR_CHECK( cudaMalloc(&states, states_size) );

    setupRandStates<<<grid_dims, block_dims>>>(*this, states, time(NULL));
    e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Trace samples.

    for (int sample = 0; sample < sample_count; ++sample) {

        traceSamples<<<grid_dims, block_dims>>>(*this, device_image, states, max_bounces);
        e = cudaGetLastError();
        CUDA_ERROR_CHECK(e);
        CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

        std::cout << "Completed sample #" << sample + 1 << "\n";
    }

    // Divide the traced samples to average them out.

    divideSamples<<<grid_dims, block_dims>>>(*this, device_image, sample_count);
    e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    // Copy device image over to a host memory.

    Image host_image { this->image_height, this->image_width };
    CUDA_ERROR_CHECK( cudaMemcpy(host_image.pixels, device_image, device_image_size, cudaMemcpyDeviceToHost) );

    return host_image;
}

__global__ void setupHittables () {

    Sphere* sphere1 = new Sphere(
        {{ 0, 0, 20 }},
        1,
        new Diffuse(FIREBRICK)
    );

    Sphere* sphere2 = new Sphere(
        {{ 5, 0, 15 }},
        1,
        new Diffuse(TURQUOISE)
    );

    Sphere* sphere3 = new Sphere(
        {{ -5, 0, 25 }},
        1,
        new Diffuse(REBECCAPURPLE)
    );

    Sphere* sphere4 = new Sphere(
        {{ -10, 0, 30 }},
        1,
        new Diffuse(MEDIUMSPRINGGREEN)
    );

    Plane* plane = new Plane(
        {{ 0, -1, 0 }},
        {{ 0, 1, 0 }},
        new Diffuse(SLATEGRAY)
    );

    hittables_count = 5;
    hittables = (Hittable**)(malloc(sizeof(Hittable*) * hittables_count));

    hittables[0] = sphere1;
    hittables[1] = sphere2;
    hittables[2] = sphere3;
    hittables[3] = sphere4;
    hittables[4] = plane;
}

__global__ void setupRandStates (Camera camera, curandState* states, uint64_t seed) {

    int height = camera.getImageHeight();
    int width = camera.getImageWidth();
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    int thread = (row * width) + col;

    if (row < height && col < width) {
        curand_init(seed, thread, 0, &states[thread]);
    }
}

__global__ void setupImage (Camera camera, Color* image) {

    int height = camera.getImageHeight();
    int width = camera.getImageWidth();
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    int thread = (row * width) + col;

    if (row < height && col < width) {
        image[thread] = {{ 0, 0, 0 }};
    }
}

__global__ void traceSamples (Camera camera, Color* image, curandState* states, int max_bounces) {

    int height = camera.getImageHeight();
    int width = camera.getImageWidth();
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    int thread = (row * width) + col;

    if (height <= row || width <= col) return;

    curandState local_state = states[thread];

    Ray ray = camera.getInitialRay(&local_state);
    Color traced { 1.0, 1.0, 1.0 };

    for (int bounce = 0; bounce < max_bounces; ++bounce) {

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
    }

    states[thread] = local_state;
    image[thread] += traced;
}

__global__ void divideSamples (Camera camera, Color* image, int sample_count) {

    int height = camera.getImageHeight();
    int width = camera.getImageWidth();
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;

    int thread = (row * width) + col;

    if (height <= row || width <= col) return;

    image[thread] /= sample_count;
}
