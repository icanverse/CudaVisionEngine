#pragma once

#include <string>

class TextureUtility {
public:
    // Dosyadaki gorseli OpenGL texture olarak yukler.
    static unsigned int LoadTextureFromFile(const char* filename);

    // Dosyadaki gorseli istenen boyuta getirip OpenGL texture olarak yukler.
    // outOriginalWidth ve outOriginalHeight kaynak gorselin gercek boyutudur.
    static unsigned int LoadThumbnailFromFile(
        const std::string& path,
        int targetWidth,
        int targetHeight,
        int& outOriginalWidth,
        int& outOriginalHeight
    );

    // Gorseli CPU tarafinda okuyup yeniden boyutlandirir.
    // Donen bellek std::free ile serbest birakilmalidir.
    static unsigned char* LoadResizedPixels(
        const std::string& path,
        int targetWidth,
        int targetHeight,
        int& outOriginalWidth,
        int& outOriginalHeight
    );

    // Hazir RGBA piksel verisini OpenGL texture'a yukler.
    // OpenGL context'inin aktif oldugu render thread'inde cagrilmalidir.
    static unsigned int CreateTextureFromPixels(
        const unsigned char* pixels,
        int width,
        int height
    );

    // Tek piksellik, duz renkli bir OpenGL texture olusturur.
    static unsigned int CreateSolidColor(
        float red,
        float green,
        float blue
    );
};