#include "time.h"
#include <iostream>

#include "camera.hpp"
#include "image.hpp"

int main () {

    clock_t start = clock();

    Camera camera {
        {{13, 0.75, 3}},
        1080,
        1920,
        35,
        {{ 4, 1, 0 }},
        0.4
    };

    Image image = camera.render(100, 100);

    image.writeToFile("render");

    clock_t end = clock();

    std::cout << "Completed in " << (end - start) / 1000000.0 << " seconds.\n";
}