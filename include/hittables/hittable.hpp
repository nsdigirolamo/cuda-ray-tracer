#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#define MIN_HIT_DISTANCE 0.0001

#include "aabb.hpp"
#include "materials/material.hpp"
#include "optional.hpp"

class Hittable {

    public:

        __host__ __device__ virtual ~Hittable () { };

        __host__ __device__ virtual Optional<Hit> checkHit (const Ray& ray) const = 0;
        __host__ __device__ virtual Optional<const Material*> getMaterial () const = 0;
        __host__ __device__ virtual AABB getSurroundingAABB () const = 0;
};

#endif