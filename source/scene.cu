#include "assert.h"
#include "curand_kernel.h"

#include "scene.hpp"
#include "hittables/plane.hpp"
#include "hittables/sphere.hpp"
#include "materials/diffuse.hpp"
#include "materials/metallic.hpp"
#include "materials/refractive.hpp"
#include "utils/cuda_utils.hpp"

__device__ Hittable** boundables;
__device__ int boundables_count;

__device__ Hittable** unboundables;
__device__ int unboundables_count;

__device__ BVHNode* root;

SortedListNode::SortedListNode (AABB aabb)
    : aabb(aabb)
    , next(NULL)
{ }

SortedList::SortedList (int sort_axis)
    : sort_axis(sort_axis)
    , head(NULL)
{ }

int SortedList::getSortAxis () {
    return this->sort_axis;
}

void SortedList::setSortAxis (int sort_axis) {

    SortedList temp_list { sort_axis };
    SortedListNode* current = this->head;

    while (current) {
        temp_list.insert(current->aabb);
        current = current->next;
    }

    this->sort_axis = sort_axis;
    this->head = temp_list.head;
}

void SortedList::insert (AABB aabb) {

    SortedListNode* new_node = new SortedListNode(aabb);

    if (!this->head) {
        this->head = new_node;
        return;
    }

    SortedListNode* current_node = this->head;

    int new_pos = CALC_MIDPOINT(aabb.getInterval(this->sort_axis));
    int current_pos = CALC_MIDPOINT(current_node->aabb.getInterval(this->sort_axis));

    if (new_pos < current_pos) {
        new_node->next = current_node;
        this->head = new_node;
        return;
    }

    SortedListNode* prev_node = current_node;
    current_node = current_node->next;

    while (current_node) {

        current_pos = CALC_MIDPOINT(current_node->aabb.getInterval(this->sort_axis));

        if (new_pos < current_pos) {
            new_node->next = current_node;
            prev_node->next = new_node;
            return;
        }

        prev_node = current_node;
        current_node = current_node->next;
    }
}

BVHNode::BVHNode (AABB aabb)
    : aabb(aabb)
    , left(NULL)
    , right(NULL)
{ }

BVH::BVH (SortedList list) {

}

int BVH::getSortAxis () {
    return this->sort_axis;
}

void BVH::setSortAxis (int sort_axis) {

}

void BVH::insert (AABB aabb) {

    BVHNode* new_node = new BVHNode(aabb);

    if (!this->root) {
        this->root = new_node;
        return;
    }

    BVHNode* current_node = this->root;

    while (current_node) {



    }

}


int findAxisToSplit (SortedList list) {

    double widest_range = 0;
    int split_axis = 0;

    for (int axis = 0; axis < 3; ++axis) {

        double max = list.head->value->getSurroundingAABB().getInterval(axis).max;
        double min = list.head->value->getSurroundingAABB().getInterval(axis).min;

        ListNode* current = list.head->next;

        while (current) {

            double new_max = current->value->getSurroundingAABB().getInterval(axis).max;
            double new_min = current->value->getSurroundingAABB().getInterval(axis).min;

            max = max < new_max ? new_max : max;
            min = new_min < min ? new_min : min;

            current = current->next;
        }

        int new_range = max - min;

        if (widest_range < new_range) {
            widest_range = new_range;
            split_axis = axis;
        }
    }

    return split_axis;
}

__global__ void setupHittables () {

    curandState state;
    curand_init(1234, 0, 0, &state);

    Plane* ground = new Plane(
        {{ 0, 0, 0 }},
        {{ 0, 1, 0 }},
        new Diffuse({ CORAL })
    );

    unboundables_count = 1;
    unboundables = (Hittable**)(malloc(sizeof(Hittable*) * unboundables_count));

    unboundables[0] = ground;

    Sphere* sphere1 = new Sphere(
        {{ 0, 1, 0 }},
        1.0,
        new Refractive({ WHITESMOKE }, 1.5)
    );

    Sphere* sphere2 = new Sphere(
        {{ -4, 1, 0 }},
        1.0,
        new Diffuse({ FIREBRICK })
    );

    Sphere* sphere3 = new Sphere(
        {{ 4, 1, 0 }},
        1.0,
        new Metallic({ STEELBLUE }, 0.0)
    );

    boundables_count = 403;
    boundables = (Hittable**)(malloc(sizeof(Hittable*) * boundables_count));

    int idx = 0;

    for (int i = -10; i < 10; ++i) {
        for (int j = -10; j < 10; ++j) {

            double choose_mat = curand_uniform_double(&state);
            Color color { curand_uniform_double(&state), curand_uniform_double(&state), curand_uniform_double(&state) };
            Material* mat;

            if (choose_mat < 0.33) {
                mat = new Diffuse(color);
            } else if (choose_mat < 0.66) {
                mat = new Metallic(color, curand_uniform_double(&state));
            } else {
                mat = new Refractive(color, 2.0 * curand_uniform_double(&state));
            }

            Sphere* s = new Sphere(
                {{ (double)(2 * i), 0.2, (double)(2 * j) }},
                0.2,
                mat
            );

            boundables[idx] = s;

            ++idx;
        }
    }

    boundables[400] = sphere1;
    boundables[401] = sphere2;
    boundables[402] = sphere3;
}

__device__ BVHNode* generateTree (SortedList list) {

    if (head->next == NULL) {
        return new BVHNode(
            head->value->getSurroundingAABB(),
            NULL,
            NULL
        );
    }

    int axis_to_split = findAxisToSplit(head);
    DLLNode* sorted = sortList(head, axis_to_split);

    DLLNode* left_head = head;
    DLLNode* right_head = NULL;

    int count = 0;
    DLLNode* node = head;
    while (node) {
        ++count;
        node = node->next;
    }

    node = head;
    int index = 0;
    while (node) {
        if (index == count / 2) {
            right_head = node;
            break;
        }
        node = node->next;
        ++index;
    }

    right_head->prev->next = NULL;
    right_head->prev = NULL;

    BVHNode* left_tree = generateTree(left_head);
    BVHNode* right_tree = generateTree(right_head);

    Interval x {
        left_tree->aabb.getInterval(0).min < right_tree->aabb.getInterval(0).min
            ? left_tree->aabb.getInterval(0).min
            : right_tree->aabb.getInterval(0).min,
        left_tree->aabb.getInterval(0).max < right_tree->aabb.getInterval(0).max
            ? left_tree->aabb.getInterval(0).max
            : right_tree->aabb.getInterval(0).max,
    };

    Interval y {
        left_tree->aabb.getInterval(1).min < right_tree->aabb.getInterval(1).min
            ? left_tree->aabb.getInterval(1).min
            : right_tree->aabb.getInterval(1).min,
        left_tree->aabb.getInterval(1).max < right_tree->aabb.getInterval(1).max
            ? left_tree->aabb.getInterval(1).max
            : right_tree->aabb.getInterval(1).max,
    };

    Interval z {
        left_tree->aabb.getInterval(2).min < right_tree->aabb.getInterval(2).min
            ? left_tree->aabb.getInterval(2).min
            : right_tree->aabb.getInterval(2).min,
        left_tree->aabb.getInterval(2).max < right_tree->aabb.getInterval(2).max
            ? left_tree->aabb.getInterval(2).max
            : right_tree->aabb.getInterval(2).max,
    };

    return new BVHNode(
        { NULL, x, y, z },
        left_tree,
        right_tree
    );
}

__global__ void generateRoot () {

    SortedList list { NULL, 0 };

    for (int i = 0; i < boundables_count; ++i) {
        list.insert(boundables[i]);
    }

    root = generateTree(list);
}

void setupScene () {

    setupHittables<<<1, 1>>>();
    cudaError_t e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );

    generateRoot<<<1, 1>>>();
    e = cudaGetLastError();
    CUDA_ERROR_CHECK(e);
    CUDA_ERROR_CHECK( cudaDeviceSynchronize() );
}
