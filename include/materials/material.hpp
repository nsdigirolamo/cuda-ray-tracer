#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "curand_kernel.h"

#include "color.hpp"
#include "hit.hpp"
#include "ray.hpp"

class Material {

    public:

        __host__ __device__ virtual ~Material () { };

        __host__ __device__ virtual Color getColor () const = 0;
        __device__ virtual Ray scatter (const Hit& hit, curandState* state) const = 0;
};

#endif