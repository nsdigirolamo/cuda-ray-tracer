#ifndef PLANE_HPP
#define PLANE_HPP

#include "hittables/hittable.hpp"

class Plane : public Hittable {

    private:

        const Point origin;
        const UnitVector<3> normal;
        Material* material;

    public:

        Plane (const Point& origin, const UnitVector<3> normal, Material* material);

        OptionalHit checkHit (const Ray& ray) const;
        Material* getMaterial ();
};

#endif