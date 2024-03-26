#include "hit.hpp"

Hit::Hit ()
    : distance(0)
    , is_front(true)
{ }

Hit::Hit (
    const Ray& incoming,
    const double distance,
    const Point& origin,
    const UnitVector<3>& surface_normal,
    const bool is_front
)
    : incoming(incoming)
    , distance(distance)
    , origin(origin)
    , surface_normal(surface_normal)
    , is_front(is_front)
{ }
