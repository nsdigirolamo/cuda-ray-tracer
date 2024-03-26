#ifndef CAMERA_HPP
#define CAMERA_HPP

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

        double pixel_height; // The height of a pixel projected on the view plane.
        double pixel_width; // The width of a pixel projected on the view plane.

        UnitVector<3> up_direction; // The up direction relative to the camera.
        UnitVector<3> view_direction; // The direction in which the camera is looking.
        UnitVector<3> view_top; // A vector pointing from the camera's origin to the view plane's topmost edge.
        UnitVector<3> view_left; // A vector pointing from the camera's origin to the view plane's leftmost edge.

    public:

        Camera(
            const Point& origin,
            const int image_height,
            const int image_width,
            const double hfov,
            const Point& focal_point
        );

        Point getOrigin () const;

        int getImageHeight () const;
        int getImageWidth () const;

        double getVfov () const;
        double getHfov () const;

        double getViewHeight () const;
        double getViewWidth () const;
        double getFocalDistance () const;

        double getPixelHeight () const;
        double getPixelWidth () const;

        UnitVector<3> getUpDirection () const;
        UnitVector<3> getViewDirection () const;
        UnitVector<3> getViewTop () const;
        UnitVector<3> getViewLeft () const;

        /**
         * @brief Generates a ray origin.
         *
         * @return Point
         */
        __device__ Point generateRayOrigin () const;

        /**
         * @brief Calculates a pixel's location in the scene based on its index.
         *
         * @return Point
         */
        __device__ Point calculatePixelLocation () const;

        /**
         * @brief Get the initial ray for a pixel.
         *
         * @return Ray A ray from the camera's origin to the pixel in the scene.
         */
        __device__ Ray getInitialRay () const;

        /**
         * @brief Renders the image.
         *
         * @param samples_per_pixel Number of samples per pixel in the image.
         * @param bounces_per_sample The maximum number of bounces a sample can take.
         * @return Image The rendered image.
         */
        Image render (
            const int samples_per_pixel,
            const int bounces_per_sample
        ) const;
};

/**
 * @brief Allocates space for and generates the scene's hittables.
 * This kernel should be launched by a single thread in a single block.
 *
 * @return __global__
 */
__global__ void generateHittables ();

/**
 * @brief Traces a sample in the scene. Each sample should have its own thread.
 * The gridDims for this kernel should be as follows: (image_width, image_height).
 * The blockDims for this kernel should be as follows: (samples_per_pixel).
 *
 * @param camera The camera that is viewing the scene.
 * @param samples A 3D array of samples allocated to device memory.
 * @param bounces_per_sample The maximum number of bounces per sample.
 */
__global__ void traceSample (
    Camera camera,
    Color* samples,
    const int bounces_per_sample
);

/**
 * @brief Reduces a 3D array of samples down to a 2D array of pixels. Each pixel
 * in the 2D array should have a single thread assigned to it.
 *
 * @param samples A 3D array of samples allocated to device memory.
 * @param image_height The height of samples.
 * @param image_width The width of samples.
 * @param reduced_samples A 2D array of pixels allocated to device memory.
 * @param samples_per_pixel The depth of samples.
 */
__global__ void reduceSamples (
    Color* samples,
    const int image_height,
    const int image_width,
    Color* reduced_samples,
    int samples_per_pixel
);


#endif