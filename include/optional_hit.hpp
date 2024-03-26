#ifndef OPTIONAL_HIT_HPP
#define OPTIONAL_HIT_HPP

#include "hit.hpp"

class OptionalHit {

    public:

        bool exists;
        Hit hit;

        __host__ __device__ OptionalHit ();
        __host__ __device__ OptionalHit (const Hit& hit);
};

#endif