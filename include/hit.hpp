#ifndef HIT_HPP
#define HIT_HPP

#include "ray.hpp"
#include "vector.hpp"

class Hittable;

class Hit {

    public:

        const Hittable* hittable; // The hittable that was hit.

        Ray incoming; // The ray that caused the hit.
        double distance; // The distance along the incoming ray where the hit occurred.

        Point origin; // The location of the hit in the scene.
        UnitVector<3> surface_normal; // The normal of the surface that was hit.
        bool is_front; // True if the hit occurred on the front of the surface.

        __host__ __device__ Hit ();
        __host__ __device__ Hit (
            const Hittable* hittable,
            const Ray& incoming,
            const double distance,
            const Point& origin,
            const UnitVector<3>& surface_normal,
            const bool is_front
        );
};

#endif