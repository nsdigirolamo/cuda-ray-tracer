#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "math.h"

template <unsigned int N>
class UnitVector;

template <unsigned int N>
class Vector {

    private:

        double values[N];

    public:

        Vector () {
            for (int i = 0; i < N; ++i) {
                this->values[i] = 0;
            }
        }

        Vector (const double (&values)[N]) {
            for (int i = 0; i < N; ++i) {
                this->values[i] = values[i];
            }
        }

        Vector<N>& operator+= (const Vector<N>& rhs) {
            for (int i = 0; i < N; ++i) {
                (*this)[i] += rhs[i];
            }
            return *this;
        }

        Vector<N>& operator-= (const Vector<N>& rhs) {
            for (int i = 0; i < N; ++i) {
                (*this)[i] -= rhs[i];
            }
            return *this;
        }

        Vector<N>& operator*= (const double factor) {
            for (int i = 0; i < N; ++i) {
                (*this)[i] *= factor;
            }
            return *this;
        }

        Vector<N>& operator/= (const double divisor) {
            for (int i = 0; i < N; ++i) {
                (*this)[i] /= divisor;
            }
            return *this;
        }

        double& operator[] (const unsigned int index) {
            return this->values[index];
        }

        const double& operator[] (const unsigned int index) const {
            return this->values[index];
        }

        /**
         * @return The number of elements in the vector.
         */
        unsigned int height () const {
            return N;
        }

        /**
         * @return The vector's length squared.
         */
        double length_squared () const {
            return dot(*this, *this);
        }

        /**
         * @return The vector's length.
         */
        double length () const {
            return sqrtf(this->length_squared());
        }

        /**
         * @return The vector's direction as a unit vector.
         */
        UnitVector<N> direction () const {
            return (UnitVector<3>)(*this);
        }
};

template <unsigned int N>
Vector<N> operator+ (const Vector<N>& lhs, const Vector<N>& rhs) {
    Vector<N> result = lhs;
    result += rhs;
    return result;
}

template <unsigned int N>
Vector<N> operator- (const Vector<N>& lhs, const Vector<N>& rhs) {
    Vector<N> result = lhs;
    result -= rhs;
    return result;
}

template <unsigned int N>
Vector<N> operator* (const Vector<N>& vector, const double factor) {
    Vector<N> result = vector;
    result *= factor;
    return result;
}

template <unsigned int N>
Vector<N> operator* (const double factor, const Vector<N>& vector) {
    Vector<N> result = vector;
    result *= factor;
    return result;
}

/**
 * @brief Performs the hadamard product on two vectors, where the vectors are
 * multipled together in an element-wise fashion.
 *
 * @tparam N The number of elements in the vectors.
 * @param lhs The vector on the left hand side of the operation.
 * @param rhs The vector on the right hand side of the operation.
 * @return The hadamard product vector.
 */
template <unsigned int N>
Vector<N> hadamard (const Vector<N>& lhs, const Vector<N>& rhs) {
    Vector<N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

template <unsigned int N>
Vector<N> operator/ (const Vector<N>& vector, const double divisor) {
    Vector<N> result = vector;
    result /= divisor;
    return result;
}

template <unsigned int N>
bool operator== (const Vector<N>& lhs, const Vector<N>& rhs) {
    for (int i = 0; i < N; ++i) {
        if (lhs[i] != rhs[i]) return false;
    }
    return true;
}

/**
 * @brief Checks if two vectors are approximately equal.
 *
 * @tparam N The number of elements in the vectors.
 * @param lhs The vector on the left hand side of the operation.
 * @param rhs The vector on the right hand side of the operation.
 * @param epsilon The maximum difference between any two elements.
 * @return True if the difference between all elements in the vectors is less than or equal to epsilon.
 * @return False if the difference between any two elements in the vectors is greater than epsilon.
 */
template <unsigned int N>
bool isApprox (const Vector<N>& lhs, const Vector<N>& rhs, const double epsilon = 0.001) {
    for (int i = 0; i < N; ++i) {
        if (epsilon < abs(lhs[i] - rhs[i])) return false;
    }
    return true;
}

template <unsigned int N>
bool operator!= (const Vector<N>& lhs, const Vector<N>& rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Normalizes a vector.
 *
 * @tparam N The number of elements in the vector.
 * @param vector The vector to be normalized.
 * @return The normalized vector.
 */
template <unsigned int N>
UnitVector<N> normalize (Vector<N> vector) {
    return (UnitVector<N>)(vector);
}

/**
 * @brief Calculates the dot product of two vectors.
 *
 * @tparam N The number of elements in the vectors.
 * @param lhs The vector on the left hand side of the operation.
 * @param rhs The vector on the right hand side of the operation.
 * @return The dot product of the two vectors.
 */
template <unsigned int N>
double dot (const Vector<N>& lhs, const Vector<N>& rhs) {
    double dot = 0.0;
    for (int i = 0; i < N; ++i) {
        dot += lhs[i] * rhs[i];
    }
    return dot;
}

/**
 * @brief Expands to a calculation for the cross product of two 3D vectors.
 *
 * @param lhs The vector on the left hand side of the operation. Must be a 3D vector.
 * @param rhs The vector on the right hand side of the operation. Must be a 3D vector.
 */
#define CROSS(lhs, rhs) \
    (Vector<3>){{ \
        lhs[1] * rhs[2] - lhs[2] * rhs[1], \
        lhs[2] * rhs[0] - lhs[0] * rhs[2], \
        lhs[0] * rhs[1] - lhs[1] * rhs[0], \
    }};

/**
 * @brief Reflects a vector off some surface normal.
 *
 * @tparam N The number of elements in the vector and surface normal.
 * @param vector The vector to be reflected.
 * @param surface_normal The normal vector of the surface the vector is reflecting off of.
 * @return The reflected vector.
 */
template <unsigned int N>
Vector<N> reflect (const Vector<N>& vector, const UnitVector<N>& surface_normal) {
    return vector - 2 * dot(vector, surface_normal) * surface_normal;
}

/**
 * @brief Refracts a vector through some object's surface.
 *
 * @tparam N The number of elements in the vector and surface normal.
 * @param vector The vector to be refracted.
 * @param surface_normal The normal vector of the surface the vector is refracting through.
 * @param refractive_index The refractive index of the refracting material.
 * @return The refracted vector.
 */
template <unsigned int N>
Vector<N> refract (const UnitVector<N>& vector, const UnitVector<N>& surface_normal, const double refractive_index) {
    double cos_theta = dot(-1 * vector, surface_normal);
    Vector<3> normal_orthogonal = refractive_index * (vector + cos_theta * surface_normal);
    Vector<3> normal_parallel = -1 * sqrt(abs(1 - normal_orthogonal.length_squared())) * surface_normal;
    return normal_orthogonal + normal_parallel;
}

template <unsigned int N>
class UnitVector: public Vector<N> {

    public:

        UnitVector (const Vector<N>& vector) {
            Vector<N> unit = vector / vector.length();
            for (int i = 0; i < N; ++i) {
                (*this)[i] = unit[i];
            }
        }

        UnitVector (const double (&values)[N]) {
            Vector<N> vector { values };
            *this = { vector };
        }

        UnitVector () {
            *this = {{ 1.0, 0.0, 0.0 }};
        }
};

typedef Vector<3> Point;

#endif
