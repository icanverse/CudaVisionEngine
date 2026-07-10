#pragma once
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

// YENİ: Windows / Eski GL Sürümleri için Güvenlik Kalkanı
#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

class CudaDynamicTexture {
public:
    CudaDynamicTexture(int width, int height);
    ~CudaDynamicTexture();

    // CUDA Kernel'inin içine piksel yazabilmesi için yüzeyi (Surface) açar
    cudaSurfaceObject_t map();

    // Çizim bitince yüzeyi kapatıp kontrolü OpenGL/ImGui'ye geri verir
    void unmap();

    // ImGui'nin ekrana basması gereken Doku ID'sini verir
    GLuint getTextureID() const { return textureID; }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    int width, height;
    GLuint textureID;

    // CUDA-GL Etkileşim (Interop) Kaynakları
    cudaGraphicsResource_t cudaResource;
    cudaSurfaceObject_t surfaceObject;
};