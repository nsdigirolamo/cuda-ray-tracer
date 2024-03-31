#include "aabb.hpp"

__host__ __device__ AABB::AABB (
    const Hittable* bounded,
    const Interval& x_interval,
    const Interval& y_interval,
    const Interval& z_interval
)
    : bounded(bounded)
    , x_interval(x_interval)
    , y_interval(y_interval)
    , z_interval(z_interval)
{ }

__host__ __device__ Interval AABB::getInterval (const int axis) const {
    return
        axis == 0 ? this->x_interval :
        axis == 1 ? this->y_interval :
        this->z_interval;
}

__host__ __device__ bool AABB::isHit (const Ray& ray) const {

    Interval intervals[3];

    for (int i = 0; i < 3; ++i) {

        double min_calc = ray.origin[i] - this->getInterval(i).min / ray.direction[i];
        double max_calc = ray.origin[i] - this->getInterval(i).max / ray.direction[i];

        intervals[i] = { min_calc, max_calc };
    }

    return
        IS_OVERLAPPING(intervals[0], intervals[1]) &&
        IS_OVERLAPPING(intervals[1], intervals[2]) &&
        IS_OVERLAPPING(intervals[2], intervals[0]);
}
