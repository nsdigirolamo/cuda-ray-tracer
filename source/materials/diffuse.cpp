#include "assert.h"
#include "materials/diffuse.hpp"

Diffuse::Diffuse (const Color& color)
    : color(color)
{ }

Color Diffuse::getColor () const {
    return this->color;
}

Ray Diffuse::scatter (const Hit& hit) const {

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
