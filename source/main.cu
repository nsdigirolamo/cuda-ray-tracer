#include "time.h"
#include <iostream>

#include "camera.hpp"
#include "image.hpp"

int main () {

    clock_t start = clock();

    Camera camera {
        {{ 0, 0, -20 }},
        1080,
        1920,
        90,
        {{ 0, 0, 0 }},
        5.0
    };

    Image image = camera.render(50, 50);

    image.writeToFile("render");

    clock_t end = clock();

    std::cout << "Completed in " << (end - start) / 1000000.0 << " seconds.\n";
}