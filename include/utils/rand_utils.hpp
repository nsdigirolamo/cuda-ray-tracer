#ifndef RAND_UTILS_HPP
#define RAND_UTILS_HPP

#include "curand_kernel.h"

#include "vector.hpp"

#define RANDOM_DOUBLE(state) 2 * curand_uniform_double(state) - 1

__device__ Point randomInUnitSphere (curandState* state);

__device__ Point randomOnUnitSphere (curandState* state);

__device__ Vector<3> randomInUnitHemisphere (const UnitVector<3>& normal, curandState* state);

__device__ UnitVector<3> randomOnUnitHemisphere (const UnitVector<3>& normal, curandState* state);

__device__ Vector<2> randomInUnitCircle (curandState* state);

__device__ UnitVector<2> randomOnUnitCircle (curandState* state);

#endif