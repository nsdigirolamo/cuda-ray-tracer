#include "aabb.hpp"

#include "stdio.h"

__host__ __device__ AABB::AABB (
    const Hittable* bounded,
    Interval x_interval,
    Interval y_interval,
    Interval z_interval
)
    : bounded(bounded)
    , x_interval(x_interval)
    , y_interval(y_interval)
    , z_interval(z_interval)
{ }

__host__ __device__ const Interval AABB::getInterval (const int axis) const {
    return
        axis == 0 ? this->x_interval :
        axis == 1 ? this->y_interval :
        this->z_interval;
}

__host__ __device__ bool AABB::isHit (const Ray& ray) const {

    Interval intervals[3];

    for (int i = 0; i < 3; ++i) {

        double min_calc = (this->getInterval(i).min - ray.origin[i]) / ray.direction[i];
        double max_calc = (this->getInterval(i).max - ray.origin[i]) / ray.direction[i];

        if (isnan(min_calc)) min_calc = -1.0 * max_calc;
        if (isnan(max_calc)) max_calc = -1.0 * min_calc;

        intervals[i] = { min_calc, max_calc };
    }

    bool interval_behind_ray =
        (intervals[0].min < 0 && intervals[0].max < 0) ||
        (intervals[1].min < 0 && intervals[1].max < 0) ||
        (intervals[2].min < 0 && intervals[2].max < 0);

    bool all_overlap =
        IS_OVERLAPPING(intervals[0], intervals[1]) &&
        IS_OVERLAPPING(intervals[1], intervals[2]) &&
        IS_OVERLAPPING(intervals[2], intervals[0]);

    return all_overlap && !interval_behind_ray;
}
