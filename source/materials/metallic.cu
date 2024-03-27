#include "materials/metallic.hpp"
#include "utils/rand_utils.hpp"

__host__ __device__ Metallic::Metallic (const Color& color, const double blur)
    : color(color)
    , blur(blur)
{ }

__host__ __device__ Color Metallic::getColor () const {
    return this->color;
}

__host__ __device__ double Metallic::getBlur () const {
    return this->blur;
}

__device__ Ray Metallic::scatter (const Hit& hit, curandState* state) const {

    return {
        hit.origin,
        reflect(hit.incoming.direction, hit.surface_normal) + (this->blur * randomOnUnitSphere(state))
    };
}
