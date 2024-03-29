#include "hit.hpp"

__host__ __device__ Hit::Hit ()
    : distance(0)
    , is_front(true)
{ }

__host__ __device__ Hit::Hit (
    const Hittable* hittable,
    const Ray& incoming,
    const double distance,
    const Point& origin,
    const UnitVector<3>& surface_normal,
    const bool is_front
)
    : hittable(hittable)
    , incoming(incoming)
    , distance(distance)
    , origin(origin)
    , surface_normal(surface_normal)
    , is_front(is_front)
{ }
