#ifndef AABB_HPP
#define AABB_HPP

#include "hit.hpp"
#include "interval.hpp"
#include "optional.hpp"

class Hittable;

class AABB {

    private:

        const Hittable* bounded;

        Interval x_interval;
        Interval y_interval;
        Interval z_interval;

    public:

        __host__ __device__ AABB (
            const Hittable* bounded,
            Interval x_interval,
            Interval y_interval,
            Interval z_interval
        );

        __host__ __device__ const Hittable* getBounded () const;

        __host__ __device__ const Interval getInterval (const int axis) const;

        __host__ __device__ bool isHit (const Ray& ray) const;
};

#endif