#ifndef SCENE_HPP
#define SCENE_HPP

#include "hittables/hittable.hpp"

struct ListNode {

    ListNode* next = NULL;
    Hittable* hittable = NULL;
};

class Scene {

    public:

        ListNode* head = NULL;
        ListNode* tail = NULL;

        ~Scene ();

        void push (Hittable* hittable);
        void pop ();

};

#endif