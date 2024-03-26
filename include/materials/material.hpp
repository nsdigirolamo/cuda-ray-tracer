#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "color.hpp"
#include "hit.hpp"
#include "ray.hpp"

class Material {

    public:

        __host__ __device__ virtual ~Material () { };

        __host__ __device__ virtual Color getColor () const = 0;
        __host__ __device__ virtual Ray scatter (const Hit& hit) const = 0;
};

#endif