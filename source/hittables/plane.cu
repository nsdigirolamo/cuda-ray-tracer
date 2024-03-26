#include "hittables/plane.hpp"

__host__ __device__ Plane::Plane (const Point& origin, const UnitVector<3> normal, Material* material)
    : origin(origin)
    , normal(normal)
    , material(material)
{ }

__host__ __device__ Plane::~Plane () {
    delete this->material;
    this->material = NULL;
}

__host__ __device__ OptionalHit Plane::checkHit (const Ray& ray) const {

    OptionalHit opt;

    double denominator = dot(ray.direction, this->normal);

    if (denominator != 0) {

        double distance = dot(this->origin - ray.origin, this->normal) / denominator;

        if (distance < MIN_HIT_DISTANCE) { return opt; }

        Point intersection = ray.direction * distance + ray.origin;
        bool is_front = dot(ray.direction, this->normal) <= 0;
        UnitVector<3> surface_normal = this->normal * (is_front ? 1.0 : -1.0);

        Hit hit {
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

__host__ __device__ Material* Plane::getMaterial () {
    return this->material;
}
