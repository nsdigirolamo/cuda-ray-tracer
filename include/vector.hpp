#ifndef VECTOR_HPP
#define VECTOR_HPP

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

#endif
