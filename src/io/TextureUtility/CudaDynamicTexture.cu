#include "io/TextureUtility/CudaDynamicTexture.cuh"

#include <cuda_gl_interop.h>
#include <iostream>

CudaDynamicTexture::CudaDynamicTexture(int w, int h) : width(w), height(h), surfaceObject(0) {
    // 1. OpenGL tarafında boş bir doku (Texture) oluştur
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Dokunun filtreleme ayarları (Lineer = Pürüzsüz)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // CUDA'nın üzerine yazabilmesi için boş VRAM ayır (RGBA32F kullanıyoruz ki renk hesaplamaları hassas olsun)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // 2. Bu OpenGL dokusunu CUDA'ya "Yazılabilir Kaynak" olarak kaydet
    cudaError_t err = cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (err != cudaSuccess) {
        std::cerr << "[CudaDynamicTexture] HATA: Doku CUDA'ya kaydedilemedi! Kod: " << err << std::endl;
    }
}

CudaDynamicTexture::~CudaDynamicTexture() {
    if (surfaceObject) {
        cudaDestroySurfaceObject(surfaceObject);
    }
    if (cudaResource) {
        cudaGraphicsUnregisterResource(cudaResource);
    }
    if (textureID) {
        glDeleteTextures(1, &textureID);
    }
}

cudaSurfaceObject_t CudaDynamicTexture::map() {
    // Kaynağı CUDA kullanımına aç
    cudaGraphicsMapResources(1, &cudaResource, 0);

    // Mapped array'i (diziyi) al
    cudaArray_t mappedArray;
    cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResource, 0, 0);

    // CUDA Kernel'inin içine yazabileceği bir Surface Object (Yüzey Nesnesi) oluştur
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = mappedArray;

    cudaCreateSurfaceObject(&surfaceObject, &resDesc);

    return surfaceObject;
}

void CudaDynamicTexture::unmap() {
    if (surfaceObject) {
        cudaDestroySurfaceObject(surfaceObject);
        surfaceObject = 0;
    }
    // İşimiz bitti, ImGui'nin okuması için kaynağı serbest bırak
    cudaGraphicsUnmapResources(1, &cudaResource, 0);
}