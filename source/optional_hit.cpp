#include "optional_hit.hpp"

OptionalHit::OptionalHit (const Hit& hit)
    : exists(true)
    , hit(hit)
{ }

OptionalHit::OptionalHit ()
    : exists(false)
{ }
