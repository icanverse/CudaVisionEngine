#include "TextureUtility/CudaDynamicTexture.cuh"

#include <cuda_gl_interop.h>
#include <iostream>

namespace {

__global__ void copyInterleavedFloatToSurface(
    cudaSurfaceObject_t surface,
    const float* source,
    int width,
    int height,
    int channels,
    float valueScale
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int pixelIndex = y * width + x;
    const int sourceIndex = pixelIndex * channels;

    float4 pixel;
    if (channels == 1) {
        const float value = fminf(fmaxf(source[sourceIndex] * valueScale, 0.0f), 1.0f);
        pixel = make_float4(value, value, value, 1.0f);
    } else if (channels == 3) {
        pixel = make_float4(
            fminf(fmaxf(source[sourceIndex] * valueScale, 0.0f), 1.0f),
            fminf(fmaxf(source[sourceIndex + 1] * valueScale, 0.0f), 1.0f),
            fminf(fmaxf(source[sourceIndex + 2] * valueScale, 0.0f), 1.0f),
            1.0f
        );
    } else {
        pixel = make_float4(
            fminf(fmaxf(source[sourceIndex] * valueScale, 0.0f), 1.0f),
            fminf(fmaxf(source[sourceIndex + 1] * valueScale, 0.0f), 1.0f),
            fminf(fmaxf(source[sourceIndex + 2] * valueScale, 0.0f), 1.0f),
            fminf(fmaxf(source[sourceIndex + 3] * valueScale, 0.0f), 1.0f)
        );
    }

    surf2Dwrite(pixel, surface, x * static_cast<int>(sizeof(float4)), y);
}

} // namespace

CudaDynamicTexture::CudaDynamicTexture(int w, int h)
    : width(w),
      height(h),
      textureID(0),
      cudaResource(nullptr),
      surfaceObject(0) {
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
    if (cudaResource == nullptr) return 0;

    // Kaynağı CUDA kullanımına aç
    const cudaError_t mapError = cudaGraphicsMapResources(1, &cudaResource, 0);
    if (mapError != cudaSuccess) {
        std::cerr << "[CudaDynamicTexture] Map hatasi: "
                  << cudaGetErrorString(mapError) << std::endl;
        return 0;
    }

    // Mapped array'i (diziyi) al
    cudaArray_t mappedArray = nullptr;
    const cudaError_t arrayError = cudaGraphicsSubResourceGetMappedArray(
        &mappedArray,
        cudaResource,
        0,
        0
    );
    if (arrayError != cudaSuccess) {
        std::cerr << "[CudaDynamicTexture] Mapped array alinamadi: "
                  << cudaGetErrorString(arrayError) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        return 0;
    }

    // CUDA Kernel'inin içine yazabileceği bir Surface Object (Yüzey Nesnesi) oluştur
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = mappedArray;

    const cudaError_t surfaceError = cudaCreateSurfaceObject(
        &surfaceObject,
        &resDesc
    );
    if (surfaceError != cudaSuccess) {
        std::cerr << "[CudaDynamicTexture] Surface olusturulamadi: "
                  << cudaGetErrorString(surfaceError) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
        return 0;
    }

    return surfaceObject;
}

void CudaDynamicTexture::unmap() {
    if (surfaceObject) {
        cudaDestroySurfaceObject(surfaceObject);
        surfaceObject = 0;
    }
    // İşimiz bitti, ImGui'nin okuması için kaynağı serbest bırak
    if (cudaResource != nullptr) {
        cudaGraphicsUnmapResources(1, &cudaResource, 0);
    }
}

bool CudaDynamicTexture::updateFromDeviceData(
    const float* deviceImageData,
    int channels,
    float displayValueScale
) {
    if (deviceImageData == nullptr || cudaResource == nullptr) return false;
    if (channels != 1 && channels != 3 && channels != 4) {
        std::cerr << "[CudaDynamicTexture] Desteklenmeyen kanal sayisi: "
                  << channels << std::endl;
        return false;
    }

    const cudaSurfaceObject_t surface = map();
    if (surface == 0) return false;

    const dim3 block(16, 16);
    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    copyInterleavedFloatToSurface<<<grid, block>>>(
        surface,
        deviceImageData,
        width,
        height,
        channels,
        displayValueScale
    );

    const cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        std::cerr << "[CudaDynamicTexture] Kopyalama kernel hatasi: "
                  << cudaGetErrorString(launchError) << std::endl;
        unmap();
        return false;
    }

    // OpenGL texture'i okumadan once CUDA yaziminin tamamlandigini garanti et.
    const cudaError_t syncError = cudaDeviceSynchronize();
    unmap();

    if (syncError != cudaSuccess) {
        std::cerr << "[CudaDynamicTexture] Senkronizasyon hatasi: "
                  << cudaGetErrorString(syncError) << std::endl;
        return false;
    }

    return true;
}
