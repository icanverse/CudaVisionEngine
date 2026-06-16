#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

// MOTOR BAŞLIK DOSYALARI
#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Scene.cuh"
#include "Graphics/Renderer3D.cuh"

// CANLI AKIŞ VE EKRAN ÇIKTISI BAŞLIKLARI
#include "io/NetworkStream.h"
#include "io/Video/NvDecoder.h"
#include "io/GlfwInteropTarget.h"

// NVIDIA Optimus sürücüsünü bu EXE için ayrık GPU'yu (RTX 4070) kullanmaya zorlar
#ifdef _WIN32
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}
#endif

int main() {
    std::cout << "[Main] Kivilcim OOP Asenkron Motor Modu Baslatiliyor..." << std::endl;

    // 0. CUDA BAĞLAMI BAŞLATMA
    cudaSetDevice(0);
    cudaFree(0);

    // 1. CANLI AKIŞ VE MOTOR KURULUMU
    std::string rtspUrl = "rtsp://192.168.1.102:8080/h264_pcm.sdp";

    NetworkStream phoneInput(rtspUrl);
    NvDecoder decoder;

    int width = 1920;
    int height = 1080;
    int channels = 3;

    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Asenkron AR Motoru");

    // ==============================================================================
    // 2. SAHNE VE GEOMETRİ TANIMLAMALARI
    // ==============================================================================
    Scene myScene;

    float3 starVertices[10] = {
        { 0.0f,  0.0f,  0.25f}, { 0.0f,  0.0f, -0.25f}, { 0.0f,  0.8f,  0.0f},
        { 0.2f,  0.2f,  0.0f},  { 0.8f,  0.0f,  0.0f},  { 0.2f, -0.2f,  0.0f},
        { 0.0f, -0.8f,  0.0f},  {-0.2f, -0.2f,  0.0f},  {-0.8f,  0.0f,  0.0f},
        {-0.2f,  0.2f,  0.0f}
    };
    int3 starIndices[16] = {
        {0, 2, 9}, {0, 9, 8}, {0, 8, 7}, {0, 7, 6}, {0, 6, 5}, {0, 5, 4}, {0, 4, 3}, {0, 3, 2},
        {1, 2, 3}, {1, 3, 4}, {1, 4, 5}, {1, 5, 6}, {1, 6, 7}, {1, 7, 8}, {1, 8, 9}, {1, 9, 2}
    };

    float3 stickVertices[8] = {
        {-0.02f, -0.02f, 0.0f}, { 0.02f, -0.02f, 0.0f}, { 0.02f,  0.02f, 0.0f}, {-0.02f,  0.02f, 0.0f},
        {-0.02f, -0.02f, 3.0f}, { 0.02f, -0.02f, 3.0f}, { 0.02f,  0.02f, 3.0f}, {-0.02f,  0.02f, 3.0f}
    };
    int3 stickIndices[12] = {
        {0,1,2}, {0,2,3}, {4,6,5}, {4,7,6}, {0,4,5}, {0,5,1},
        {3,2,6}, {3,6,7}, {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
    };

    Material redPlastic = {{0.9f, 0.1f, 0.1f}, 0.1f, 0.8f, 0.3f, 16.0f};
    Material blueMetal  = {{0.2f, 0.4f, 0.9f}, 0.1f, 0.4f, 1.0f, 128.0f};
    Material axisRed    = {{1.0f, 0.0f, 0.0f}, 0.8f, 0.2f, 0.0f, 1.0f};
    Material axisGreen  = {{0.0f, 1.0f, 0.0f}, 0.8f, 0.2f, 0.0f, 1.0f};
    Material axisBlue   = {{0.0f, 0.0f, 1.0f}, 0.8f, 0.2f, 0.0f, 1.0f};

    myScene.setCamera({0.0f, 1.5f, -6.0f}, {-0.22f, 0.0f, 0.0f})
           .addObject(starVertices, 10, starIndices, 16, {-1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, redPlastic)
           .addObject(starVertices, 10, starIndices, 16, { 1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, blueMetal)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {0.0f, -1.5708f, 0.0f}, axisRed)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {-1.5708f, 0.0f, 0.0f}, axisGreen)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, axisBlue)
           .addLight({0.0f, 5.0f, -4.0f}, {1.0f, 1.0f, 1.0f}, 1.2f)
           .addLight({-3.0f, -2.0f, -3.0f}, {0.2f, 0.2f, 0.9f}, 0.5f);

    // ==============================================================================
    // 3. ASENKRON AĞI BAŞLAT
    // ==============================================================================
    phoneInput.startStream(&decoder);

    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    std::cout << "[Main] Cizim dongusu basladi! Guc kilidi acildi." << std::endl;

    // ==============================================================================
    // 4. ANA ÇİZİM DÖNGÜSÜ (Tüketici)
    // ==============================================================================
    while (!target.shouldClose()) {
        timeTracker += 0.02f;

        CUdeviceptr renderFrame = 0;
        unsigned int renderPitch = 0;

        // A) VİZYON KATI: Nesne yönelimli, tertemiz bir okuma yapıyoruz.
        // Yeni kare varsa tuvale bas, yoksa eski tuvalle yoluna devam et.
        if (phoneInput.getLatestFrame(renderFrame, renderPitch)) {
            visionEngine.loadNV12DevicePointer(renderFrame, renderPitch);
        }

        // B) SİMÜLASYON KATI: Yıldızları kendi ekseninde döndür
        auto& mutableObjects = const_cast<std::vector<Object3D>&>(myScene.getObjects());
        if (mutableObjects.size() >= 2) {
            mutableObjects[0].rotation.y = timeTracker;
            mutableObjects[0].rotation.x = timeTracker * 0.3f;
            mutableObjects[1].rotation.y = -timeTracker;
            mutableObjects[1].rotation.x = timeTracker * 0.3f;
        }

        // C) GRAFİK KATI: Görüntü üzerine 3D'leri çiz
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // D) EKRANA YANSIT
        unsigned char* d_pbo_vram_address = target.mapVRAM();
        visionEngine.copyToDeviceUchar(d_pbo_vram_address);
        target.unmapAndRender();

        // FPS SAYACI
        frameCount++;
        if (frameCount % 100 == 0) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double fps = 1000.0 / (std::chrono::duration<double, std::milli>(t_end - t_start).count() / 100.0);
            std::cout << "Kivilcim Core FPS: " << std::fixed << std::setprecision(1) << fps << "    \r" << std::flush;
            t_start = std::chrono::high_resolution_clock::now();
        }

        glfwPollEvents();
    }

    // ==============================================================================
    // 5. GÜVENLİ KAPANIŞ
    // ==============================================================================
    phoneInput.stopStream();

    return 0;
}