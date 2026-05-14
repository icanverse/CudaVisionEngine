#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Scene.cuh"
#include "Graphics/Renderer3D.cuh"

#include "io/GlfwInteropTarget.h"

int main() {
    std::cout << "[Main] Kivilcim Saf 3D Motor Modu Baslatiliyor..." << std::endl;

    // 0. CUDA ZORLA BAŞLATMA
    cudaSetDevice(0);
    cudaFree(0);

    // 1. EKRAN VE MOTOR BOYUTLARI
    int width = 1280;
    int height = 720;
    int channels = 3;

    // 2. MOTORLARI BAŞLAT
    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sabit Kamera & Eksen Testi");

    // ==============================================================================
    // 3. SAHNEYİ KUR VE GEOMETRİLERİ TANIMLA
    // ==============================================================================
    Scene myScene;

    // A) Gemini Yıldızı Geometrisi
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

    // B) EKSEN GİZMO GEOMETRİSİ (Z eksenine doğru uzanan ince uzun bir çubuk)
    float3 stickVertices[8] = {
        {-0.02f, -0.02f, 0.0f}, { 0.02f, -0.02f, 0.0f}, { 0.02f,  0.02f, 0.0f}, {-0.02f,  0.02f, 0.0f},
        {-0.02f, -0.02f, 3.0f}, { 0.02f, -0.02f, 3.0f}, { 0.02f,  0.02f, 3.0f}, {-0.02f,  0.02f, 3.0f}
    };
    int3 stickIndices[12] = {
        {0,1,2}, {0,2,3}, {4,6,5}, {4,7,6}, {0,4,5}, {0,5,1},
        {3,2,6}, {3,6,7}, {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
    };

    // C) MATERYALLER
    Material redPlastic = {{0.9f, 0.1f, 0.1f}, 0.1f, 0.8f, 0.3f, 16.0f};
    Material blueMetal  = {{0.2f, 0.4f, 0.9f}, 0.1f, 0.4f, 1.0f, 128.0f};

    Material axisRed   = {{1.0f, 0.0f, 0.0f}, 0.8f, 0.2f, 0.0f, 1.0f}; // X Ekseni Rengi
    Material axisGreen = {{0.0f, 1.0f, 0.0f}, 0.8f, 0.2f, 0.0f, 1.0f}; // Y Ekseni Rengi
    Material axisBlue  = {{0.0f, 0.0f, 1.0f}, 0.8f, 0.2f, 0.0f, 1.0f}; // Z Ekseni Rengi

    // D) SAHNEYE DİZİLİM
    // Kamerayı tamamen STATİK hale getirdik: Konum sabit, hafifçe aşağı doğru bakıyor (-0.27 radyan Pitch)
    myScene.setCamera({0.0f, 2.0f, -7.0f}, {-0.27f, 0.0f, 0.0f})
           // Yıldızlar
           .addObject(starVertices, 10, starIndices, 16, {-1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, redPlastic)
           .addObject(starVertices, 10, starIndices, 16, { 1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, blueMetal)
           // Eksen çubukları (Tam merkezde statik duruyorlar)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {0.0f, -1.5708f, 0.0f}, axisRed)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {-1.5708f, 0.0f, 0.0f}, axisGreen)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, axisBlue)
           // Işıklar
           .addLight({0.0f, 5.0f, -4.0f}, {1.0f, 1.0f, 1.0f}, 1.2f)
           .addLight({-3.0f, -2.0f, -3.0f}, {0.2f, 0.2f, 0.9f}, 0.5f);

    // ==============================================================================
    // 4. ANA RENDER DÖNGÜSÜ
    // ==============================================================================
    double targetFPS = 600000.0;
    auto target_frame_duration = std::chrono::duration<double, std::milli>(1000.0 / targetFPS);
    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    while (!target.shouldClose()) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        timeTracker += 0.02f;

        // EKRANI TEMİZLE
        cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

        // OBJELERİ DÖNDÜR
        auto& mutableObjects = const_cast<std::vector<Object3D>&>(myScene.getObjects());

        // Sadece yıldızlar dönüyor, eksenler sabit kalıyor
        if (mutableObjects.size() >= 2) {
            mutableObjects[0].rotation.y = timeTracker;
            mutableObjects[0].rotation.x = timeTracker * 0.3f;
            mutableObjects[1].rotation.y = -timeTracker;
            mutableObjects[1].rotation.x = timeTracker * 0.3f;
        }

        // DİKKAT: Kamera güncelleme kodu döngü içinden tamamen kaldırıldı!
        // Kamera, yukarıda setCamera ile atadığımız sabit koordinatta bekliyor.

        // GRAFİK ÇİZİMİ
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // EKRANA YANSIT
        unsigned char* d_pbo_vram_address = target.mapVRAM();
        visionEngine.copyToDeviceUchar(d_pbo_vram_address);
        target.unmapAndRender();

        // FPS Hesaplama
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time);
        if (processing_time < target_frame_duration) std::this_thread::sleep_for(target_frame_duration - processing_time);

        frameCount++;
        if (frameCount % 60 == 0) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double fps = 1000.0 / (std::chrono::duration<double, std::milli>(t_end - t_start).count() / 60.0);
            std::cout << "Render FPS: " << std::fixed << std::setprecision(1) << fps << "    \r" << std::flush;
            t_start = std::chrono::high_resolution_clock::now();
        }

        glfwPollEvents();
    }

    return 0;
}