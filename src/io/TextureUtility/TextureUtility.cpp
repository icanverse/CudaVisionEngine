#include <iostream>

// 1. ÖNCE WINDOWS BAŞLIKLARI
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

// 2. SONRA OPENGL/GLFW BAŞLIKLARI
#include <GLFW/glfw3.h>

// --- WINDOWS OPENGL EKSİK MAKRO YAMASI ---
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

// 3. BAŞLIKLAR
#include "io/TextureUtility/TextureUtility.h"

// DİKKAT: "#define STB_IMAGE_IMPLEMENTATION" BURADAN SİLİNDİ!
#include "stb_image.h"

unsigned int TextureUtility::LoadTextureFromFile(const char* filename) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Doku sarmalama ve filtreleme parametreleri
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Görüntüyü yükle
    int width, height, nrChannels;

    // OpenGL resimleri aşağıdan yukarı okur
    stbi_set_flip_vertically_on_load(true);

    unsigned char* data = stbi_load(filename, &width, &height, &nrChannels, 4);

    if (data) {
        // Artık if-else ile 3 mü 4 mü diye kontrol etmemize gerek yok.
        // Veriyi zorla 4 kanala çektiğimiz için format her zaman GL_RGBA'dır.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    } else {
        std::cout << "[HATA] Logo/Ikon yuklenemedi: " << filename << "\n";
        stbi_image_free(data);
        return 0;
    }

    return textureID;
}