#pragma once
#include <string>

class TextureUtility {
public:
    // Dosya yolunu alıp OpenGL Texture ID'si döndüren statik fonksiyon
    static unsigned int LoadTextureFromFile(const char* filename);
};