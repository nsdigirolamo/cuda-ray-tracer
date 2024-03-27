#include "utils/rand_utils.hpp"

__device__ Point randomInUnitSphere (curandState* state) {

    double x, y, z;
    Point p;

    do {

        x = curand_uniform_double(state);
        y = curand_uniform_double(state);
        z = curand_uniform_double(state);

        p = {{ x, y, z }};

    } while (1 < p.length());

    return p;
}

__device__ Point randomOnUnitSphere (curandState* state) {

    return (UnitVector<3>)(randomInUnitSphere(state));
}
