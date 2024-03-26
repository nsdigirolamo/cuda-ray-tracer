#include "materials/refractive.hpp"

__host__ __device__ Refractive::Refractive (const Color& color, const double refractive_index)
    : color(color)
    , refractive_index(refractive_index)
{ }

__host__ __device__ Color Refractive::getColor () const {
    return this->color;
}

__host__ __device__ double Refractive::getRefIdx () const {
    return this->refractive_index;
}

__host__ __device__ Ray Refractive::scatter (const Hit& hit) const {

    double ref_idx = this->refractive_index;

    double ref_ratio = hit.is_front ? 1.0 / ref_idx : ref_idx;

    double cos_theta = dot(-1.0 * hit.incoming.direction, hit.surface_normal);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    bool cant_refract = 1.0 < ref_ratio * sin_theta;

    Vector<3> direction;

    if (cant_refract) {
        direction = reflect(hit.incoming.direction, hit.surface_normal);
    } else {
        direction = refract(hit.incoming.direction, hit.surface_normal, ref_ratio);
    }

    return { hit.origin, direction };
}
