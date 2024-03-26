#ifndef DIFFUSE_HPP
#define DIFFUSE_HPP

#include "materials/material.hpp"

class Diffuse : public Material {

    private:

        const Color color;

    public:

        __host__ __device__ Diffuse (const Color& color);

        __host__ __device__ Color getColor () const;
        __host__ __device__ Ray scatter (const Hit& hit) const;
};

#endif