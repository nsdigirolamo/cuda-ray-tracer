#ifndef COLOR_HPP
#define COLOR_HPP

#include "vector.hpp"

#define MIN_HEX 0x000000
#define MAX_HEX 0xFFFFFF

class Color : public Vector<3> {

    public:

        Color ();
        Color (const double (&values)[3]);
        Color (const Vector<3>& vector);
        Color (const double red, const double green, const double blue);
        Color (const int rgb_hex_value);
};

#define SKY SKYBLUE

#define BLACK                (Color){ 0x000000 }
#define SILVER               (Color){ 0xC0C0C0 }
#define GRAY                 (Color){ 0x808080 }
#define WHITE                (Color){ 0xFFFFFF }
#define MAROON               (Color){ 0x800000 }
#define RED                  (Color){ 0xFF0000 }
#define PURPLE               (Color){ 0x800080 }
#define FUCHSIA              (Color){ 0xFF00FF }
#define GREEN                (Color){ 0x008000 }
#define LIME                 (Color){ 0x00FF00 }
#define OLIVE                (Color){ 0x808000 }
#define YELLOW               (Color){ 0xFFFF00 }
#define NAVY                 (Color){ 0x000080 }
#define BLUE                 (Color){ 0x0000FF }
#define TEAL                 (Color){ 0x008080 }
#define AQUA                 (Color){ 0x00FFFF }
#define ALICEBLUE            (Color){ 0xF0F8FF }
#define ANTIQUEWHITE         (Color){ 0xFAEBD7 }
#define AQUA                 (Color){ 0x00FFFF }
#define AQUAMARINE           (Color){ 0x7FFFD4 }
#define AZURE                (Color){ 0xF0FFFF }
#define BEIGE                (Color){ 0xF5F5DC }
#define BISQUE               (Color){ 0xFFE4C4 }
#define BLACK                (Color){ 0x000000 }
#define BLANCHEDALMOND       (Color){ 0xFFEBCD }
#define BLUE                 (Color){ 0x0000FF }
#define BLUEVIOLET           (Color){ 0x8A2BE2 }
#define BROWN                (Color){ 0xA52A2A }
#define BURLYWOOD            (Color){ 0xDEB887 }
#define CADETBLUE            (Color){ 0x5F9EA0 }
#define CHARTREUSE           (Color){ 0x7FFF00 }
#define CHOCOLATE            (Color){ 0xD2691E }
#define CORAL                (Color){ 0xFF7F50 }
#define CORNFLOWERBLUE       (Color){ 0x6495ED }
#define CORNSILK             (Color){ 0xFFF8DC }
#define CRIMSON              (Color){ 0xDC143C }
#define CYAN                 (Color){ 0x00FFFF }
#define DARKBLUE             (Color){ 0x00008B }
#define DARKCYAN             (Color){ 0x008B8B }
#define DARKGOLDENROD        (Color){ 0xB8860B }
#define DARKGRAY             (Color){ 0xA9A9A9 }
#define DARKGREEN            (Color){ 0x006400 }
#define DARKGREY             (Color){ 0xA9A9A9 }
#define DARKKHAKI            (Color){ 0xBDB76B }
#define DARKMAGENTA          (Color){ 0x8B008B }
#define DARKOLIVEGREEN       (Color){ 0x556B2F }
#define DARKORANGE           (Color){ 0xFF8C00 }
#define DARKORCHID           (Color){ 0x9932CC }
#define DARKRED              (Color){ 0x8B0000 }
#define DARKSALMON           (Color){ 0xE9967A }
#define DARKSEAGREEN         (Color){ 0x8FBC8F }
#define DARKSLATEBLUE        (Color){ 0x483D8B }
#define DARKSLATEGRAY        (Color){ 0x2F4F4F }
#define DARKSLATEGREY        (Color){ 0x2F4F4F }
#define DARKTURQUOISE        (Color){ 0x00CED1 }
#define DARKVIOLET           (Color){ 0x9400D3 }
#define DEEPPINK             (Color){ 0xFF1493 }
#define DEEPSKYBLUE          (Color){ 0x00BFFF }
#define DIMGRAY              (Color){ 0x696969 }
#define DIMGREY              (Color){ 0x696969 }
#define DODGERBLUE           (Color){ 0x1E90FF }
#define FIREBRICK            (Color){ 0xB22222 }
#define FLORALWHITE          (Color){ 0xFFFAF0 }
#define FORESTGREEN          (Color){ 0x228B22 }
#define FUCHSIA              (Color){ 0xFF00FF }
#define GAINSBORO            (Color){ 0xDCDCDC }
#define GHOSTWHITE           (Color){ 0xF8F8FF }
#define GOLD                 (Color){ 0xFFD700 }
#define GOLDENROD            (Color){ 0xDAA520 }
#define GRAY                 (Color){ 0x808080 }
#define GREEN                (Color){ 0x008000 }
#define GREENYELLOW          (Color){ 0xADFF2F }
#define GREY                 (Color){ 0x808080 }
#define HONEYDEW             (Color){ 0xF0FFF0 }
#define HOTPINK              (Color){ 0xFF69B4 }
#define INDIANRED            (Color){ 0xCD5C5C }
#define INDIGO               (Color){ 0x4B0082 }
#define IVORY                (Color){ 0xFFFFF0 }
#define KHAKI                (Color){ 0xF0E68C }
#define LAVENDER             (Color){ 0xE6E6FA }
#define LAVENDERBLUSH        (Color){ 0xFFF0F5 }
#define LAWNGREEN            (Color){ 0x7CFC00 }
#define LEMONCHIFFON         (Color){ 0xFFFACD }
#define LIGHTBLUE            (Color){ 0xADD8E6 }
#define LIGHTCORAL           (Color){ 0xF08080 }
#define LIGHTCYAN            (Color){ 0xE0FFFF }
#define LIGHTGOLDENRODYELLOW (Color){ 0xFAFAD2 }
#define LIGHTGRAY            (Color){ 0xD3D3D3 }
#define LIGHTGREEN           (Color){ 0x90EE90 }
#define LIGHTGREY            (Color){ 0xD3D3D3 }
#define LIGHTPINK            (Color){ 0xFFB6C1 }
#define LIGHTSALMON          (Color){ 0xFFA07A }
#define LIGHTSEAGREEN        (Color){ 0x20B2AA }
#define LIGHTSKYBLUE         (Color){ 0x87CEFA }
#define LIGHTSLATEGRAY       (Color){ 0x778899 }
#define LIGHTSLATEGREY       (Color){ 0x778899 }
#define LIGHTSTEELBLUE       (Color){ 0xB0C4DE }
#define LIGHTYELLOW          (Color){ 0xFFFFE0 }
#define LIME                 (Color){ 0x00FF00 }
#define LIMEGREEN            (Color){ 0x32CD32 }
#define LINEN                (Color){ 0xFAF0E6 }
#define MAGENTA              (Color){ 0xFF00FF }
#define MAROON               (Color){ 0x800000 }
#define MEDIUMAQUAMARINE     (Color){ 0x66CDAA }
#define MEDIUMBLUE           (Color){ 0x0000CD }
#define MEDIUMORCHID         (Color){ 0xBA55D3 }
#define MEDIUMPURPLE         (Color){ 0x9370DB }
#define MEDIUMSEAGREEN       (Color){ 0x3CB371 }
#define MEDIUMSLATEBLUE      (Color){ 0x7B68EE }
#define MEDIUMSPRINGGREEN    (Color){ 0x00FA9A }
#define MEDIUMTURQUOISE      (Color){ 0x48D1CC }
#define MEDIUMVIOLETRED      (Color){ 0xC71585 }
#define MIDNIGHTBLUE         (Color){ 0x191970 }
#define MINTCREAM            (Color){ 0xF5FFFA }
#define MISTYROSE            (Color){ 0xFFE4E1 }
#define MOCCASIN             (Color){ 0xFFE4B5 }
#define NAVAJOWHITE          (Color){ 0xFFDEAD }
#define NAVY                 (Color){ 0x000080 }
#define OLDLACE              (Color){ 0xFDF5E6 }
#define OLIVE                (Color){ 0x808000 }
#define OLIVEDRAB            (Color){ 0x6B8E23 }
#define ORANGE               (Color){ 0xFFA500 }
#define ORANGERED            (Color){ 0xFF4500 }
#define ORCHID               (Color){ 0xDA70D6 }
#define PALEGOLDENROD        (Color){ 0xEEE8AA }
#define PALEGREEN            (Color){ 0x98FB98 }
#define PALETURQUOISE        (Color){ 0xAFEEEE }
#define PALEVIOLETRED        (Color){ 0xDB7093 }
#define PAPAYAWHIP           (Color){ 0xFFEFD5 }
#define PEACHPUFF            (Color){ 0xFFDAB9 }
#define PERU                 (Color){ 0xCD853F }
#define PINK                 (Color){ 0xFFC0CB }
#define PLUM                 (Color){ 0xDDA0DD }
#define POWDERBLUE           (Color){ 0xB0E0E6 }
#define PURPLE               (Color){ 0x800080 }
#define REBECCAPURPLE        (Color){ 0x663399 }
#define RED                  (Color){ 0xFF0000 }
#define ROSYBROWN            (Color){ 0xBC8F8F }
#define ROYALBLUE            (Color){ 0x4169E1 }
#define SADDLEBROWN          (Color){ 0x8B4513 }
#define SALMON               (Color){ 0xFA8072 }
#define SANDYBROWN           (Color){ 0xF4A460 }
#define SEAGREEN             (Color){ 0x2E8B57 }
#define SEASHELL             (Color){ 0xFFF5EE }
#define SIENNA               (Color){ 0xA0522D }
#define SILVER               (Color){ 0xC0C0C0 }
#define SKYBLUE              (Color){ 0x87CEEB }
#define SLATEBLUE            (Color){ 0x6A5ACD }
#define SLATEGRAY            (Color){ 0x708090 }
#define SLATEGREY            (Color){ 0x708090 }
#define SNOW                 (Color){ 0xFFFAFA }
#define SPRINGGREEN          (Color){ 0x00FF7F }
#define STEELBLUE            (Color){ 0x4682B4 }
#define TAN                  (Color){ 0xD2B48C }
#define TEAL                 (Color){ 0x008080 }
#define THISTLE              (Color){ 0xD8BFD8 }
#define TOMATO               (Color){ 0xFF6347 }
#define TURQUOISE            (Color){ 0x40E0D0 }
#define VIOLET               (Color){ 0xEE82EE }
#define WHEAT                (Color){ 0xF5DEB3 }
#define WHITE                (Color){ 0xFFFFFF }
#define WHITESMOKE           (Color){ 0xF5F5F5 }
#define YELLOW               (Color){ 0xFFFF00 }
#define YELLOWGREEN          (Color){ 0x9ACD32 }

#endif