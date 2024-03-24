#ifndef HIT_HPP
#define HIT_HPP

#include "vector.hpp"

class Hit {

    public:

        const UnitVector<3> incoming; // The ray that caused the hit.
        const double distance; // The distance along the incoming ray where the hit occurred.

        const Point origin; // The location of the hit in the scene.
        const UnitVector<3> surface_normal; // The normal of the surface that was hit.
        const bool is_front; // True if the hit occurred on the front of the surface.
};

#endif