#ifndef REFRACTIVE_HPP
#define REFRACTIVE_HPP

#include "materials/material.hpp"

class Refractive : public Material {

    private:

        const Color color;
        const double refractive_index;

    public:

        Refractive (const Color& color, const double refractive_index);

        Color getColor () const;
        double getRefIdx () const;
        Ray scatter (const Hit& hit) const;
};

#endif