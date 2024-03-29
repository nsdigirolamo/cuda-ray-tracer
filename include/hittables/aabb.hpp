#ifndef AABB_HPP
#define AABB_HPP

#include "hittables/hittable.hpp"

class AABB : public Hittable {

    private:



    public:

        __host__ __device__ AABB ();
        __host__ __device__ ~AABB () override;

        __host__ __device__ Optional<Hit> checkHit (const Ray& ray) const override;
        __host__ __device__ Optional<const Material*> getMaterial () const override;
};

#endif