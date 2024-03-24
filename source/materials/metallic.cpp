#include "assert.h"
#include "materials/metallic.hpp"

Metallic::Metallic (const Color& color, const double blur)
    : color(color)
    , blur(blur)
{ }

Color Metallic::getColor () const {
    return this->color;
}

double Metallic::getBlur () const {
    return this->blur;
}

Ray Metallic::scatter (const Hit& hit, const UnitVector<3>& random_offset) const {
    assert(hit.exists);
    return {
        hit.origin,
        reflect(hit.incoming.direction, hit.surface_normal) + (blur * random_offset)
    };
}
