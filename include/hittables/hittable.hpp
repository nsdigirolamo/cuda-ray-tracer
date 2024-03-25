#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#define MIN_HIT_DISTANCE 0.0001

#include "materials/material.hpp"
#include "optional_hit.hpp"

class Hittable {

    public:

        virtual ~Hittable () { };

        virtual OptionalHit checkHit (const Ray& ray) const = 0;
        virtual Material* getMaterial () = 0;
};

#endif