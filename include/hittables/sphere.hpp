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
        __host__ __device__ ~Sphere () override;

        __host__ __device__ Optional<Hit> checkHit (const Ray& ray) const override;
        __host__ __device__ Optional<const Material*> getMaterial () const override;
        __host__ __device__ AABB getSurroundingAABB () const override;
};

#endif