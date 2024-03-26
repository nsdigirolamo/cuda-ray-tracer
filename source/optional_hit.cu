#include "optional_hit.hpp"

__host__ __device__ OptionalHit::OptionalHit (const Hit& hit)
    : exists(true)
    , hit(hit)
{ }

__host__ __device__ OptionalHit::OptionalHit ()
    : exists(false)
{ }
