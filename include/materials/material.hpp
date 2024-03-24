#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "color.hpp"
#include "hit.hpp"
#include "ray.hpp"

class Material {

    public:

        virtual ~Material () { };

        virtual Color getColor () const = 0;
        virtual Ray scatter (const Hit& hit) const = 0;
};

#endif