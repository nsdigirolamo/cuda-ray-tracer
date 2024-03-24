#ifndef RAY_HPP
#define RAY_HPP

#include "vector.hpp"

class Ray {

    public:

        const Point origin;
        const UnitVector<3> direction;
};

#endif