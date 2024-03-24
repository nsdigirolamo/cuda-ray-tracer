#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cmath>

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

        unsigned int height () const {
            return N;
        }

        double length_squared () const {
            return dot(*this, *this);
        }

        double length () const {
            return sqrtf(this->length_squared());
        }

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

template <unsigned int N>
bool isApprox (const Vector<N>& lhs, const Vector<N>& rhs, const double epsilon = 0.001) {
    for (int i = 0; i < N; ++i) {
        double diff = lhs[i] - rhs[i];
        if (diff < 0) diff *= -1;
        if (epsilon < diff) return false;
    }
    return true;
}

template <unsigned int N>
bool operator!= (const Vector<N>& lhs, const Vector<N>& rhs) {
    return !(lhs == rhs);
}

template <unsigned int N>
UnitVector<N> normalize (Vector<N> vector) {
    return (UnitVector<N>)(vector);
}

template <unsigned int N>
double dot (const Vector<N>& lhs, const Vector<N>& rhs) {
    double dot = 0.0;
    for (int i = 0; i < N; ++i) {
        dot += lhs[i] * rhs[i];
    }
    return dot;
}

#define CROSS(lhs, rhs) \
    (Vector<3>){{ \
        lhs[1] * rhs[2] - lhs[2] * rhs[1], \
        lhs[2] * rhs[0] - lhs[0] * rhs[2], \
        lhs[0] * rhs[1] - lhs[1] * rhs[0], \
    }};

template <unsigned int N>
Vector<N> reflect (const Vector<N>& vector, const UnitVector<N>& surface_normal) {
    return vector - 2 * dot(vector, surface_normal) * surface_normal;
}

template <unsigned int N>
Vector<N> refract (const UnitVector<N>& vector, const UnitVector<N>& surface_normal, const double refractive_index) {
    double cos_theta = dot(-1 * vector, surface_normal);
    Vector<3> normal_orthogonal = refractive_index * (vector + cos_theta * surface_normal);
    Vector<3> normal_parallel = -1 * sqrt(abs(1 - normal_orthogonal.length_squared()));
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
