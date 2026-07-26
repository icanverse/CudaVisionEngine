#pragma once
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

// YENİ: Layers / Eski GL Sürümleri için Güvenlik Kalkanı
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

    // ProjectData::d_imageData icindeki interleaved float goruntuyu dogrudan
    // CUDA-OpenGL texture'ina aktarir. 1, 3 ve 4 kanal desteklenir.
    bool updateFromDeviceData(
        const float* deviceImageData,
        int channels,
        float displayValueScale = 1.0f / 255.0f
    );

    int getWidth() const { return width; }
    int getHeight() const { return height; }

    CudaDynamicTexture(const CudaDynamicTexture&) = delete;
    CudaDynamicTexture& operator=(const CudaDynamicTexture&) = delete;

private:
    int width, height;
    GLuint textureID;

    // CUDA-GL Etkileşim (Interop) Kaynakları
    cudaGraphicsResource_t cudaResource;
    cudaSurfaceObject_t surfaceObject;
};
