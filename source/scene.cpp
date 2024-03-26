#include "scene.hpp"

Scene::~Scene () {

    while (this->head != NULL) {

        ListNode* old = this->head;
        this->head = this->head->next;

        delete old->hittable;
        delete old;
    }
}

void Scene::push (Hittable* hittable) {

    ListNode* newest = new ListNode();
    newest->hittable = hittable;

    this->tail ? this->tail->next = newest : this->head = newest;
    this->tail = newest;
}

void Scene::pop () {

    ListNode* old = this->tail;
    this->tail->next = NULL;

    delete old->hittable;
    delete old;
}
