#ifndef RAY_HPP
#define RAY_HPP

#include "vector.hpp"

class Ray {

    public:

        Point origin;
        UnitVector<3> direction;
};

#endif