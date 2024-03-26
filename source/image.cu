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