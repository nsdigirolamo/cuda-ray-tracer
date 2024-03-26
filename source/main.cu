#include "camera.hpp"
#include "image.hpp"

int main () {

    Camera camera {
        {{ 0, 0, -5 }},
        1080,
        1920,
        90,
        {{ 0, 0, 0 }}
    };

    Image image = camera.render(50, 50);

    image.writeToFile("render");
}