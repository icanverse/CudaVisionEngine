#include <cstddef>
#include <cstdlib>
#include <iostream>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif

    #include <windows.h>
#endif

#include <GLFW/glfw3.h>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

#include "TextureUtility/TextureUtility.h"

#include "stb_image.h"
#include "stb_image_resize.h"

namespace {

void configureLinearTexture() {
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_S,
        GL_CLAMP_TO_EDGE
    );
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_WRAP_T,
        GL_CLAMP_TO_EDGE
    );
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_LINEAR
    );
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_LINEAR
    );
}

} // namespace

unsigned int TextureUtility::LoadTextureFromFile(
    const char* filename
) {
    int width = 0;
    int height = 0;
    int channels = 0;

    stbi_set_flip_vertically_on_load(true);

    unsigned char* data = stbi_load(
        filename,
        &width,
        &height,
        &channels,
        4
    );

    if (!data) {
        std::cerr
            << "[TextureUtility] Gorsel yuklenemedi: "
            << filename
            << std::endl;
        return 0;
    }

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    configureLinearTexture();

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data
    );

    stbi_image_free(data);
    return textureID;
}

unsigned int TextureUtility::LoadThumbnailFromFile(
    const std::string& path,
    int targetWidth,
    int targetHeight,
    int& outOriginalWidth,
    int& outOriginalHeight
) {
    unsigned char* resizedPixels = LoadResizedPixels(
        path,
        targetWidth,
        targetHeight,
        outOriginalWidth,
        outOriginalHeight
    );

    if (!resizedPixels) {
        return 0;
    }

    const unsigned int textureID = CreateTextureFromPixels(
        resizedPixels,
        targetWidth,
        targetHeight
    );

    std::free(resizedPixels);
    return textureID;
}

unsigned char* TextureUtility::LoadResizedPixels(
    const std::string& path,
    int targetWidth,
    int targetHeight,
    int& outOriginalWidth,
    int& outOriginalHeight
) {
    outOriginalWidth = 0;
    outOriginalHeight = 0;

    if (targetWidth <= 0 || targetHeight <= 0) {
        std::cerr
            << "[TextureUtility] Gecersiz yeniden boyutlandirma: "
            << targetWidth
            << "x"
            << targetHeight
            << std::endl;
        return nullptr;
    }

    int width = 0;
    int height = 0;
    int channels = 0;

    stbi_set_flip_vertically_on_load(true);

    unsigned char* sourcePixels = stbi_load(
        path.c_str(),
        &width,
        &height,
        &channels,
        4
    );

    if (!sourcePixels) {
        std::cerr
            << "[TextureUtility] Gorsel yuklenemedi: "
            << path
            << std::endl;
        return nullptr;
    }

    const std::size_t resizedByteCount =
        static_cast<std::size_t>(targetWidth)
        * static_cast<std::size_t>(targetHeight)
        * 4;

    auto* resizedPixels = static_cast<unsigned char*>(
        std::malloc(resizedByteCount)
    );

    if (!resizedPixels) {
        std::cerr
            << "[TextureUtility] Piksel bellegi ayrilamadi: "
            << path
            << std::endl;

        stbi_image_free(sourcePixels);
        return nullptr;
    }

    const int resizeSucceeded = stbir_resize_uint8(
        sourcePixels,
        width,
        height,
        0,
        resizedPixels,
        targetWidth,
        targetHeight,
        0,
        4
    );

    stbi_image_free(sourcePixels);

    if (!resizeSucceeded) {
        std::cerr
            << "[TextureUtility] Yeniden boyutlandirma basarisiz: "
            << path
            << std::endl;

        std::free(resizedPixels);
        return nullptr;
    }

    outOriginalWidth = width;
    outOriginalHeight = height;
    return resizedPixels;
}

unsigned int TextureUtility::CreateTextureFromPixels(
    const unsigned char* pixels,
    int width,
    int height
) {
    if (!pixels || width <= 0 || height <= 0) {
        std::cerr
            << "[TextureUtility] Gecersiz texture piksel verisi."
            << std::endl;
        return 0;
    }

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    configureLinearTexture();

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        pixels
    );

    return textureID;
}

unsigned int TextureUtility::CreateSolidColor(
    float red,
    float green,
    float blue
) {
    const unsigned char pixel[4] = {
        static_cast<unsigned char>(red * 255.0f),
        static_cast<unsigned char>(green * 255.0f),
        static_cast<unsigned char>(blue * 255.0f),
        255
    };

    GLuint textureID = 0;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        GL_NEAREST
    );
    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_NEAREST
    );

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        1,
        1,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        pixel
    );

    return textureID;
}