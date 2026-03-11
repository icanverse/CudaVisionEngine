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

GlfwInteropTarget::GlfwInteropTarget(int w, int h, int c, const std::string& title)
    : width(w), height(h), channels(c) {

    if (!glfwInit()) exit(1);
    window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
    if (!window) exit(1);
    glfwMakeContextCurrent(window);

    // V-Sync'i kapatan (FPS limitini kaldıran) o sihirli satır:
    glfwSwapInterval(0);

    // OpenGL Uzantılarını GLFW üzerinden manuel yükle
    glGenBuffersExt = (PFNGLGENBUFFERSPROC)glfwGetProcAddress("glGenBuffers");
    glBindBufferExt = (PFNGLBINDBUFFERPROC)glfwGetProcAddress("glBindBuffer");
    glBufferDataExt = (PFNGLBUFFERDATAPROC)glfwGetProcAddress("glBufferData");

    // ==========================================================
    // İŞTE SİHİR BURADA: PBO (Pixel Buffer Object) OLUŞTURMA
    // ==========================================================
    glGenBuffersExt(1, &pbo);
    glBindBufferExt(GL_PIXEL_UNPACK_BUFFER, pbo);

    // VRAM'de boş bir alan ayır (Stream Draw = Sürekli güncellenecek demek)
    glBufferDataExt(GL_PIXEL_UNPACK_BUFFER, width * height * channels, NULL, GL_STREAM_DRAW);
    glBindBufferExt(GL_PIXEL_UNPACK_BUFFER, 0);

    // Bu OpenGL havuzunu CUDA'ya kaydet (Köprüyü kur)
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

GlfwInteropTarget::~GlfwInteropTarget() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glfwDestroyWindow(window);
    glfwTerminate();
}

unsigned char* GlfwInteropTarget::mapVRAM() {
    // CUDA'ya "Bu belleği ben kullanacağım, OpenGL dokunmasın" de
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);

    unsigned char* d_pbo_data = nullptr;
    size_t num_bytes;
    // O havuzun VRAM'deki GERÇEK bellek adresini al
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_data, &num_bytes, cuda_pbo_resource);

    return d_pbo_data; // Bu adresi EngineFactory'ye göndereceğiz!
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

    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool GlfwInteropTarget::shouldClose() {
    return glfwWindowShouldClose(window);
}