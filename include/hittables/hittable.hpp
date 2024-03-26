#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#define MIN_HIT_DISTANCE 0.00001

#include "materials/material.hpp"
#include "optional_hit.hpp"

class Hittable {

    public:

        __host__ __device__ virtual ~Hittable () { };

        __host__ __device__ virtual OptionalHit checkHit (const Ray& ray) const = 0;
        __host__ __device__ virtual Material* getMaterial () = 0;
};

#endif