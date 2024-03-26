#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hittables/hittable.hpp"

class Sphere : public Hittable {

    private:

        const Point origin;
        const double radius;
        Material* material;

    public:

        __host__ __device__ Sphere(const Point& origin, const double radius, Material* material);
        __host__ __device__ ~Sphere ();

        __host__ __device__ OptionalHit checkHit (const Ray& ray) const;
        __host__ __device__ Material* getMaterial ();
};

#endif