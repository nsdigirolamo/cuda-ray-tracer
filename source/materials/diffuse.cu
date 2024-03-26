#include "assert.h"
#include "materials/diffuse.hpp"

__host__ __device__ Diffuse::Diffuse (const Color& color)
    : color(color)
{ }

__host__ __device__ Color Diffuse::getColor () const {
    return this->color;
}

__host__ __device__ Ray Diffuse::scatter (const Hit& hit) const {

    /**
     * TODO: Implement random offset
     *
     *   return {
     *       hit.origin,
     *       hit.surface_normal + random_offset
     *   };
     *
     */

    return {
        hit.origin,
        hit.surface_normal
    };
}
