#ifndef RAND_UTILS_HPP
#define RAND_UTILS_HPP

#include "curand_kernel.h"

#include "vector.hpp"

__device__ Point randomInUnitSphere (curandState* state);
__device__ Point randomOnUnitSphere (curandState* state);

#endif