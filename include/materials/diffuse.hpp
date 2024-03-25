#ifndef DIFFUSE_HPP
#define DIFFUSE_HPP

#include "materials/material.hpp"

class Diffuse : public Material {

    private:

        const Color color;

    public:

        Diffuse (const Color& color);

        Color getColor () const;
        Ray scatter (const Hit& hit) const;
};

#endif