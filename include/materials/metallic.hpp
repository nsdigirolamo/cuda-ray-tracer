#ifndef METALLIC_HPP
#define METALLIC_HPP

#include "materials/material.hpp"

class Metallic : public Material {

    private:

        const Color color;
        const double blur;

    public:

        __host__ __device__ Metallic (const Color& color, const double blur);

        __host__ __device__ Color getColor () const;
        __host__ __device__ double getBlur () const;
        __host__ __device__ Ray scatter (const Hit& hit) const;
};

#endif