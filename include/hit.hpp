#ifndef HIT_HPP
#define HIT_HPP

#include "ray.hpp"
#include "vector.hpp"

class Hit {

    public:

        const Ray incoming; // The ray that caused the hit.
        const double distance; // The distance along the incoming ray where the hit occurred.

        const Point origin; // The location of the hit in the scene.
        const UnitVector<3> surface_normal; // The normal of the surface that was hit.
        const bool is_front; // True if the hit occurred on the front of the surface.

        Hit ();
        Hit (
            const Ray& incoming,
            const double distance,
            const Point& origin,
            const UnitVector<3>& surface_normal,
            const bool is_front
        );
};

#endif