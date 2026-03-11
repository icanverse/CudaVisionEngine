#include "../../include/io/StbImageTarget.h"
#include "stb_image_write.h" // STB Write sadece burada çalışacak!
#include <iostream>

StbImageTarget::StbImageTarget(const char* filepath) : filename(filepath) {}

void StbImageTarget::present(unsigned char* data, int width, int height, int channels) {
    std::cout << "[StbImageTarget] Islenmis gorsel diske kaydediliyor: " << filename << std::endl;
    stbi_write_png(filename, width, height, channels, data, width * channels);
}