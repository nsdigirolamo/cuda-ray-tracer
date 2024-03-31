#ifndef AABB_HPP
#define AABB_HPP

#include "hit.hpp"
#include "interval.hpp"
#include "optional.hpp"

class Hittable;

class AABB {

    public:

        const Hittable* bounded = NULL;

        const Interval x_interval;
        const Interval y_interval;
        const Interval z_interval;

        __host__ __device__ AABB (
            const Hittable* bounded,
            const Interval& x_interval,
            const Interval& y_interval,
            const Interval& z_interval
        );

        __host__ __device__ Interval getInterval (const int axis) const;

        __host__ __device__ bool isHit (const Ray& ray) const;
};

#endif