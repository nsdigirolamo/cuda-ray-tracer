#ifndef OPTIONAL_HIT_HPP
#define OPTIONAL_HIT_HPP

#include "hit.hpp"

class OptionalHit {

    public:

        bool exists;
        Hit hit;

        OptionalHit ();
        OptionalHit (const Hit& hit);
};

#endif