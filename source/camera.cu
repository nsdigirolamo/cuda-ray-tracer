#include "camera.hpp"

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

    double y = 0.5 * this->view_height - col * this->pixel_height - 0.5 * this->pixel_height;
    double x = -0.5 * this->view_width + row * this->pixel_width + 0.5 * this->pixel_width;

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
    Hittable** hittables,
    const int hittable_count,
    const int samples_per_pixel,
    const int bounces_per_sample
) const {

    // Allocate space to record samples.

    Color* samples;
    size_t samples_size = sizeof(Color) * this->image_width * this->image_height * samples_per_pixel;
    cudaMalloc(&samples, samples_size);

    dim3 trace_grid_dims(this->image_width, this->image_height);
    dim3 trace_block_dims(samples_per_pixel);

    // Trace each sample.

    traceSample<<<trace_grid_dims, trace_block_dims>>>(*this, samples, hittables, hittable_count, bounces_per_sample);
    cudaDeviceSynchronize();

    // Reduce the samples down to a single pixel color.

    int reduce_size = 32;
    dim3 reduce_grid_dims(
        (this->image_width + reduce_size - 1) / reduce_size,
        (this->image_height + reduce_size - 1) / reduce_size
    );
    dim3 reduce_block_dims(reduce_size, reduce_size);

    Color* reduced_samples;
    size_t reduced_samples_size = sizeof(Color) * this->image_width * this->image_height;
    cudaMalloc(&reduced_samples, reduced_samples_size);

    reduceSamples<<<reduce_grid_dims, reduce_block_dims>>>(samples, this->image_height, this->image_width, reduced_samples, samples_per_pixel);
    cudaDeviceSynchronize();

    // Copy the reduced pixels to host memory and return.

    Image image { this->image_height, this->image_width };
    cudaMemcpy(image.pixels, reduced_samples, reduced_samples_size, cudaMemcpyDeviceToHost);

    return image;
}

__global__ void traceSample (
    Camera camera,
    Color* samples,
    Hittable** hittables,
    const int hittable_count,
    const int bounces_per_sample
) {

    int height = gridDim.y;
    int width = gridDim.x;
    int samples_per_pixel = blockDim.x;

    int row = blockIdx.y;
    int col = blockIdx.x;
    int depth = threadIdx.x;

    Ray ray = camera.getInitialRay();
    Color traced = { 1.0, 1.0, 1.0 };

    int bounce = 0;

    while (bounce < bounces_per_sample) {

        OptionalHit closest;
        Hittable* hittable = NULL;

        for (int i = 0; i < hittable_count; ++i) {

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
            ray = material->scatter(hit);

        } else {

            traced = hadamard(traced, SKY);
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
