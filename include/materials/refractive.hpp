#ifndef REFRACTIVE_HPP
#define REFRACTIVE_HPP

#include "materials/material.hpp"

class Refractive : public Material {

    private:

        const Color color;
        const double refractive_index;

    public:

        Refractive (const Color& color, const double refractive_index);

        __host__ __device__ Color getColor () const;
        __host__ __device__ double getRefIdx () const;
        __host__ __device__ Ray scatter (const Hit& hit) const;
};

#endif