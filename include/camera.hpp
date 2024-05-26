#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "curand_kernel.h"
#include "stdint.h"

#include "image.hpp"
#include "ray.hpp"

#define degsToRads(degrees) ((double)(degrees) * M_PI / 180.0)
#define radsToDegs(radians) ((double)(radians) * 180.0 / M_PI)

class Camera {

    private:

        Point origin; // The camera's location in the scene.

        int image_height; // The height of the rendered image in pixels.
        int image_width; // The width of the rendered image in pixels.

        double vfov; // The vertical field of view of the camera in degrees.
        double hfov; // The horizontal field of view of the camera in degrees.

        double view_height; // The height of the view plane.
        double view_width; // The width of the view plane.
        double focal_distance; // The distance between the view plane and the camera's origin.
        double lens_radius; // The radius of the simulated lens.

        double pixel_height; // The height of a pixel projected on the view plane.
        double pixel_width; // The width of a pixel projected on the view plane.

        UnitVector<3> up_direction; // The up direction relative to the camera.
        UnitVector<3> view_direction; // The direction in which the camera is looking.
        UnitVector<3> view_top; // A vector pointing from the camera's origin to the view plane's topmost edge.
        UnitVector<3> view_left; // A vector pointing from the camera's origin to the view plane's leftmost edge.

    public:

        /**
         * @brief Constructs a new Camera object.
         *
         * @param origin The camera's position in the scene.
         * @param image_height The height of the rendered image in pixels.
         * @param image_width The height of the rendered image in pixels.
         * @param hfov The horizontal field of view of the camera in degrees.
         * @param focal_point The point of focus for the camera.
         * @param focal_angle The maximum angle a ray can arrive at the focal point.
        */
        __host__ Camera(
            const Point& origin,
            const int image_height,
            const int image_width,
            const double hfov,
            const Point& focal_point,
            const double focal_angle
        );

        __host__ __device__ Point getOrigin () const;

        __host__ __device__ int getImageHeight () const;
        __host__ __device__ int getImageWidth () const;

        __host__ __device__ double getVfov () const;
        __host__ __device__ double getHfov () const;

        __host__ __device__ double getViewHeight () const;
        __host__ __device__ double getViewWidth () const;
        __host__ __device__ double getFocalDistance () const;
        __host__ __device__ double getLensRadius () const;

        __host__ __device__ double getPixelHeight () const;
        __host__ __device__ double getPixelWidth () const;

        __host__ __device__ UnitVector<3> getUpDirection () const;
        __host__ __device__ UnitVector<3> getViewDirection () const;
        __host__ __device__ UnitVector<3> getViewTop () const;
        __host__ __device__ UnitVector<3> getViewLeft () const;

        /**
         * @brief Generates a ray origin.
         *
         * @param state The random state of the calling thread.
         * @return Point
         */
        __device__ Point generateRayOrigin (curandState* state) const;

        /**
         * @brief Calculates a pixel's location in the scene based on its index.
         *
         * @param state The random state of the calling thread.
         * @return Point
         */
        __device__ Point calculatePixelLocation (curandState* state) const;

        /**
         * @brief Get the initial ray for a pixel.
         *
         * @param state The random state of the calling thread.
         * @return Ray A ray from the camera's origin to the pixel in the scene.
         */
        __device__ Ray getInitialRay (curandState* state) const;

        Image render (const int sample_count, const int max_bounces) const;
};

/**
 * @brief Sets up the curandState for each thread and stores them in device memory.
 *
 * @param camera The camera being used to render the image.
 * @param states A device pointer with enough allocated memory to store all states.
 * @param seed The seed to be used to generate all states.
*/
__global__ void setupRandStates (Camera camera, curandState* states, uint64_t seed = 1234);

/**
 * @brief Sets all pixel values in the image to initially be zero.
 *
 * @param camera The camera being used to render the image.
 * @param image A device pointer with enough allocated memory to store all pixels.
*/
__global__ void setupImage (Camera camera, Color* image);

/**
 * @brief Traces a sample for each pixel in the image, and adds that sample color
 * value to the respective pixel.
 *
 * @param camera The camera being used to render the image.
 * @param image A device pointer to the pixel data.
 * @param states A device pointer to curandState data.
 * @param max_bounces The maximum number of bounces to trace for each sample.
*/
__global__ void traceSamples (Camera camera, Color* image, curandState* states, int max_bounces);

/**
 * @brief Averages out the pixel color data in the image.
 *
 * @param camera The camera being used to render the image.
 * @param image A device pointer to the pixel data.
 * @param sample_count The number of samples used to render the image.
*/
__global__ void divideSamples (Camera camera, Color* image, int sample_count);

#endif
