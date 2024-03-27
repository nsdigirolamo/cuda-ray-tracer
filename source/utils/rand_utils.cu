#include "utils/rand_utils.hpp"

__device__ Point randomInUnitSphere (curandState* state) {

    double x, y, z;
    Point p;

    do {

        x = RANDOM_DOUBLE(state);
        y = RANDOM_DOUBLE(state);
        z = RANDOM_DOUBLE(state);

        p = {{ x, y, z }};

    } while (1 < p.length());

    return p;
}

__device__ Point randomOnUnitSphere (curandState* state) {

    return (UnitVector<3>)(randomInUnitSphere(state));
}

__device__ Vector<3> randomInUnitHemisphere (const UnitVector<3>& normal, curandState* state) {

    Point in_sphere = randomInUnitSphere(state);
    return 0 < dot(normal, in_sphere) ? in_sphere : -1.0 * in_sphere;
}

__device__ UnitVector<3> randomOnUnitHemisphere (const UnitVector<3>& normal, curandState* state) {

    return (UnitVector<3>)(randomInUnitHemisphere(normal, state));
}

__device__ Vector<2> randomInUnitCircle (curandState* state) {

    double x, y;
    Vector<2> p;

    do {

        x = RANDOM_DOUBLE(state);
        y = RANDOM_DOUBLE(state);

        p = {{ x, y }};

    } while (1 < p.length());

    return p;
}

__device__ UnitVector<2> randomOnUnitCircle (curandState* state) {

    return (UnitVector<2>)(randomInUnitCircle(state));
}
