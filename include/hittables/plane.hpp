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
        ~Plane ();

        __host__ __device__ OptionalHit checkHit (const Ray& ray) const;
        __host__ __device__ Material* getMaterial ();
};

#endif