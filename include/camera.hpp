#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "cuda.h"
#include "math.h"

#include "color.hpp"
#include "image.hpp"
#include "ray.hpp"
#include "scene.hpp"
#include "vector.hpp"

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

        __device__ Point generateRayOrigin () const;
        __device__ Point calculatePixelLocation () const;
        __device__ Ray getInitialRay () const;

        Image render (
            const int samples_per_pixel,
            const int bounces_per_sample
        ) const;
};

__global__ void generateHittables ();

__global__ void traceSample (
    Camera camera,
    Color* samples,
    const int bounces_per_sample
);

__global__ void reduceSamples (
    Color* samples,
    const int image_height,
    const int image_width,
    Color* reduced_samples,
    int samples_per_pixel
);


#endif