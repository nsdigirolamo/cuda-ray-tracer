#ifndef PLANE_HPP
#define PLANE_HPP

#include "hittables/hittable.hpp"

class Plane : public Hittable {

    private:

        const Point origin;
        const UnitVector<3> normal;
        Material* material;

    public:

        __host__ __device__ Plane (const Point& origin, const UnitVector<3> normal, Material* material);
        __host__ __device__ ~Plane () override;

        __host__ __device__ Optional<Hit> checkHit (const Ray& ray) const override;
        __host__ __device__ Optional<const Material*> getMaterial () const override;
        __host__ __device__ AABB getSurroundingAABB () const override;
};

#endif