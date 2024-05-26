#include "time.h"
#include <iostream>

#include "camera.hpp"
#include "image.hpp"
#include "scene.hpp"

int main () {

    clock_t start = clock();

    std::cout << "Setting up the scene... ";

    Camera camera {
        {{13, 0.75, 3}},
        1080,
        1920,
        35,
        {{ 4, 1, 0 }},
        0.4
    };

    setupScene();

    std::cout << "Setup complete. Beginning render.\n";

    Image image = camera.render(10, 10);

    image.writeToFile("render");

    clock_t end = clock();

    std::cout << "Completed in " << (end - start) / 1000000.0 << " seconds.\n";
}