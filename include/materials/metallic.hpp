#ifndef METALLIC_HPP
#define METALLIC_HPP

#include "materials/material.hpp"

class Metallic : public Material {

    private:

        const Color color;
        const double blur;

    public:

        Metallic (const Color& color, const double blur);

        Color getColor () const;
        double getBlur () const;
        Ray scatter (const Hit& hit) const;
};

#endif