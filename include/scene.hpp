#ifndef SCENE_HPP
#define SCENE_HPP

#include "aabb.hpp"
#include "hittable.hpp"

class SortedListNode {

    public:

        const AABB aabb;
        SortedListNode* next;

        __host__ __device__ SortedListNode (AABB aabb);
};

class SortedList {

    private:

        int sort_axis;

    public:

        SortedListNode* head;

        __host__ __device__ SortedList (int sort_axis);
        __host__ __device__ int getSortAxis ();
        __host__ __device__ void setSortAxis (int sort_axis);

        __host__ __device__ void insert (AABB aabb);
};

class BVHNode {

    public:

        AABB aabb;
        BVHNode* left;
        BVHNode* right;

        __host__ __device__ BVHNode (AABB aabb);
};

class BVH {

    private:

        const int sort_axis;

    public:

        BVHNode* root;

        __host__ __device__ BVH (SortedList list);

        __host__ __device__ int getSortAxis ();
        __host__ __device__ void setSortAxis (int sort_axis);

        __host__ __device__ void insert (AABB aabb);
};

extern __device__ Hittable** boundables;
extern __device__ int boundables_count;

extern __device__ Hittable** unboundables;
extern __device__ int unboundables_count;

extern __device__ BVHNode* root;

void setupScene ();

#endif