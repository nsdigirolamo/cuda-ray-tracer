#ifndef OPTIONAL_HIT_HPP
#define OPTIONAL_HIT_HPP

#include "hit.hpp"

class OptionalHit {

    public:

        const bool exists;
        const Hit hit;

        OptionalHit ();
        OptionalHit (const Hit& hit);
};

#endif