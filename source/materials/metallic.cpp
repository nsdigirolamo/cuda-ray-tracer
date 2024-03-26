#include "assert.h"
#include "materials/metallic.hpp"

Metallic::Metallic (const Color& color, const double blur)
    : color(color)
    , blur(blur)
{ }

__host__ __device__ Color Metallic::getColor () const {
    return this->color;
}

__host__ __device__ double Metallic::getBlur () const {
    return this->blur;
}

__host__ __device__ Ray Metallic::scatter (const Hit& hit) const {

    /**
     * TODO: Implement random offset with blur
     *
     *   return {
     *       hit.origin,
     *       reflect(hit.incoming.direction, hit.surface_normal)
     *   };
     *
     */

    return {
        hit.origin,
        reflect(hit.incoming.direction, hit.surface_normal)
    };
}
