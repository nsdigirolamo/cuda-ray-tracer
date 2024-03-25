#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hittables/hittable.hpp"

class Sphere : public Hittable {

    private:

        const Point origin;
        const double radius;
        Material* material;

    public:

        Sphere (const Point& origin, double radius, Material* material);

        OptionalHit checkHit (const Ray& ray) const;
        Material* getMaterial ();
};

#endif