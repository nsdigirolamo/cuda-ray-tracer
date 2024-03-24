#include "color.hpp"

Color::Color (const Vector<3>& vector) {
    for (int i = 0; i < 3; ++i) {
        (*this)[i] = vector[i];
    }
}

Color::Color () {
    Vector<3> vector;
    *this = { vector };
}

Color::Color (const double (&values)[3]) {
    Vector<3> vector { values };
    *this = { vector };
}

Color::Color (const double red, const double green, const double blue) {
    Vector<3> vector {{ red, green, blue }};
    *this = { vector };
}

Color::Color (const int rgb_hex_value) {

    int val = rgb_hex_value < MIN_HEX ? 
        MIN_HEX : MAX_HEX < rgb_hex_value ? 
            MAX_HEX : rgb_hex_value;

    double b = val % 0x100;
    double g = (val >> 8) % 0x100;
    double r = (val >> 16);

    *this = { r / 256, g / 256, b / 256 };
}
