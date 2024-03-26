#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <string>

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

        /**
         * @brief Write the contents of the image to a ppm file.
         *
         * @param file_name The file's name.
         */
        void writeToFile (const std::string file_name) const;
};

#endif