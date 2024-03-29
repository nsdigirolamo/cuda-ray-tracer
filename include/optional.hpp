#ifndef OPTIONAL_HPP
#define OPTIONAL_HPP

template <class T>
class Optional {

    public:

        bool exists;
        T value;

        __host__ __device__ Optional ()
            : exists(false)
        { }

        __host__ __device__ Optional (const T& value)
            : exists(true)
            , value(value)
        { }
};

#endif