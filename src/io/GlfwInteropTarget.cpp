#include "../../include/io/GlfwInteropTarget.h"

#include <iostream>

// Windows'ta PBO kullanmak için gerekli OpenGL C++ Köprüleri (Sıfır Bağımlılık)
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#define GL_STREAM_DRAW 0x88E0

typedef void (*PFNGLGENBUFFERSPROC)(int, unsigned int*);
typedef void (*PFNGLBINDBUFFERPROC)(unsigned int, unsigned int);
typedef void (*PFNGLBUFFERDATAPROC)(unsigned int, ptrdiff_t, const void*, unsigned int);

PFNGLGENBUFFERSPROC glGenBuffersExt = nullptr;
PFNGLBINDBUFFERPROC glBindBufferExt = nullptr;
PFNGLBUFFERDATAPROC glBufferDataExt = nullptr;

// 1. DİKKAT: Kurucu fonksiyona "cuda_pbo_resource(nullptr)" eklendi!
GlfwInteropTarget::GlfwInteropTarget(int w, int h, int c, const std::string& title)
    : width(w), height(h), channels(c), cuda_pbo_resource(nullptr) {

    if (!glfwInit()) exit(1);

    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!window) exit(1);
    glfwMakeContextCurrent(window);

    glfwSwapInterval(0);

    glGenBuffersExt = (PFNGLGENBUFFERSPROC)glfwGetProcAddress("glGenBuffers");
    glBindBufferExt = (PFNGLBINDBUFFERPROC)glfwGetProcAddress("glBindBuffer");
    glBufferDataExt = (PFNGLBUFFERDATAPROC)glfwGetProcAddress("glBufferData");

    glGenBuffersExt(1, &pbo);
    glBindBufferExt(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferDataExt(GL_PIXEL_UNPACK_BUFFER, width * height * channels, NULL, GL_STREAM_DRAW);
    glBindBufferExt(GL_PIXEL_UNPACK_BUFFER, 0);

    // ==========================================
    // HATA YAKALAYICI EKLENDİ
    // ==========================================
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cout << "[FATAL] CUDA-GL Koprusu Kurulamadi! Hata: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[Basari] CUDA-GL Koprusu VRAM'e baglandi." << std::endl;
    }
}

GlfwInteropTarget::~GlfwInteropTarget() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glfwDestroyWindow(window);
    glfwTerminate();
}

unsigned char* GlfwInteropTarget::mapVRAM() {
    // Güvenlik Kilidi 1: Pointer boşsa çökme, null döndür!
    if (cuda_pbo_resource == nullptr) {
        return nullptr;
    }

    // Güvenlik Kilidi 2: Eşleme sırasında hata çıkarsa çökme!
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (err != cudaSuccess) {
        std::cout << "[HATA] mapResources Patladi: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    unsigned char* d_pbo_data = nullptr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_data, &num_bytes, cuda_pbo_resource);

    return d_pbo_data;
}

void GlfwInteropTarget::unmapAndRender() {
    // CUDA'ya "İşim bitti, kilidi aç" de
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Çizim Hazırlığı (Daha önce yaptığımız Ters Takla işlemi dahil)
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(-1.0f, 1.0f);
    glPixelZoom(1.0f, -1.0f);

    // OpenGL'e "Pikselleri RAM'den değil, şu PBO havuzundan oku" diyoruz
    glBindBufferExt(GL_PIXEL_UNPACK_BUFFER, pbo);

    // DİKKAT: En sondaki 0 (veya nullptr) CPU'da veri yok, VRAM'i kullan demek!
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);

    glBindBufferExt(GL_PIXEL_UNPACK_BUFFER, 0);

    // Çifte Yansıtmayı (Double Buffering) iptal ettik, main.cpp hallediyor.
}

bool GlfwInteropTarget::shouldClose() {
    return glfwWindowShouldClose(window);
}