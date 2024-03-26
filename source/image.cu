#include <fstream>
#include <iostream>

#include "image.hpp"

Image::Image (const int height, const int width)
    : height(height)
    , width(width)
{

    int pixel_count = width * height;
    this->pixels = (Color*)(malloc(sizeof(Color) * pixel_count));

    for (int i = 0; i < pixel_count; ++i) {
        this->pixels[i] = {{ 1, 1, 1 }};
    }
}

Image::~Image () {

    free(this->pixels);
    this->pixels = NULL;
}

Color Image::getPixel (const int row, const int col) const {
    return this->pixels[(row * width) + col];
}

void Image::setPixel(const int row, const int col, const Color& pixel) {
    this->pixels[(row * width) + col] = pixel;
}

Color correctGamma (const Color& color) {

    double gamma = 2.2;
    double exp = 1.0 / gamma;

    return {
        pow(color[0], exp),
        pow(color[1], exp),
        pow(color[2], exp)
    };
}

void writePixel (std::ofstream& file, const Color& pixel) {

    int r = pixel[0] * 255.0;
    int g = pixel[1] * 255.0;
    int b = pixel[2] * 255.0;

    file << r << " " << g << " " << b << "\n";
}

void Image::writeToFile (const std::string file_name) const {

    std::cout << "Printing image to file...\n";

    std::ofstream file(file_name + ".ppm");

    file << "P3\n\n" << this->width << " " << this->height << "\n" << "255\n";

    for (int row = 0; row < this->height; ++row) {
        for (int col = 0; col < this->width; ++col) {
            writePixel(
                file,
                correctGamma(pixels[(row * this->width) + col])
            );
        }
    }

    std::cout << "Done printing to file!\n";
}