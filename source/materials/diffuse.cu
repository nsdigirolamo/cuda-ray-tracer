#include "materials/diffuse.hpp"
#include "utils/rand_utils.hpp"

__host__ __device__ Diffuse::Diffuse (const Color& color)
    : color(color)
{ }

__host__ __device__ Color Diffuse::getColor () const {
    return this->color;
}

__device__ Ray Diffuse::scatter (const Hit& hit, curandState* state) const {

    return {
        hit.origin,
        hit.surface_normal + randomOnUnitSphere(state)
    };
}
