#include "hittables/sphere.hpp"

__host__ __device__ Sphere::Sphere (const Point& origin, double radius, Material* material)
    : origin(origin)
    , radius(radius)
    , material(material)
{ }

__host__ __device__ Sphere::~Sphere () {
    delete this->material;
    this->material = NULL;
}

__host__ __device__ Optional<Hit> Sphere::checkHit (const Ray& ray) const {

    Optional<Hit> opt;

    UnitVector<3> ud = ray.direction;
    Vector<3> oc = ray.origin - this->origin;

    double discriminant = pow(dot(ud, oc), 2) - (oc.length_squared() - pow(this->radius, 2));

    if (0 <= discriminant) {

        double uoc = -1.0 * dot(ray.direction, ray.origin - this->origin);

        double distance;

        double smaller_distance = uoc - sqrt(discriminant);
        double larger_distance = uoc + sqrt(discriminant);

        if (MIN_HIT_DISTANCE < smaller_distance) {
            distance = smaller_distance;
        } else if (MIN_HIT_DISTANCE < larger_distance) {
            distance = larger_distance;
        } else {
            return opt;
        }

        Point intersection = ray.direction * distance + ray.origin;
        UnitVector<3> surface_normal = intersection - this->origin;
        bool is_front = dot(ray.direction, surface_normal) <= 0;

        surface_normal = surface_normal * (is_front ? 1.0 : -1.0);

        Hit hit {
            this,
            ray,
            distance,
            intersection,
            surface_normal,
            is_front
        };

        return { hit };
    }

    return opt;
}

__host__ __device__ Optional<const Material*> Sphere::getMaterial () const {

    return this->material;
}

__host__ __device__ AABB Sphere::getSurroundingAABB () const {

    Vector<3> radius_vector {{ this->radius, this->radius, this->radius }};

    Point max = origin + radius_vector;
    Point min = origin - radius_vector;

    return {
        this,
        { min[0], max[0] },
        { min[1], max[1] },
        { min[2], max[2] },
    };
};
