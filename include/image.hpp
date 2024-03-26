#ifndef IMAGE_HPP
#define IMAGE_HPP

#include "color.hpp"

class Image {

    public:

        Color* pixels;

        const int height;
        const int width;

        Image (const int height, const int width);
        ~Image ();

        /**
         * @brief Get a pixel from the image.
         *
         * @param row The pixel's row.
         * @param col The pixel's column.
         * @return Color
         */
        Color getPixel (const int row, const int col) const;

        /**
         * @brief Set a pixel in the image.
         *
         * @param row The pixel's row.
         * @param col The pixel's column.
         * @param pixel The pixel's new color.
         */
        void setPixel(const int row, const int col, const Color& pixel);
};

#endif