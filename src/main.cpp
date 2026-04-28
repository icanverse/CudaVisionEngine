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

    // 1. EKRAN VE MOTOR BOYUTLARI (Sabit Çözünürlük)
    int width = 1280;
    int height = 720;
    int channels = 3;

    // 2. GİRİŞ ÇIKIŞ VE MOTORLARI BAŞLAT
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Saf 3D Render");
    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);

    // ==============================================================================
    // 3. SAHNEYİ KUR
    // ==============================================================================
    Scene myScene;

    // Gemini Yıldızı Geometrisi
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

    // Sahneye SADECE BİR TANE kızıl obje ekliyoruz
    myScene.setCamera({0.0f, 0.0f, -5.0f}, {0.0f, 0.0f, 0.0f}) // Kamera 5 metre geride
           .addObject(starVertices, 10, starIndices, 16,
                      {0.0f, 0.0f, 0.0f},        // Tam Merkeze koyduk
                      {0.0f, 0.0f, 0.0f},
                      {0.9f, 0.1f, 0.1f})        // Kızıl / Kırmızı Renk
           .addLight({2.0f, 5.0f, -4.0f}, {1.0f, 1.0f, 1.0f}, 1.0f)   // Ana Işık
           .addLight({-3.0f, -2.0f, -3.0f}, {0.2f, 0.2f, 0.9f}, 0.5f); // Sol alttan hafif mavi dolgu ışığı (Kızılı öne çıkarır)

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

        // Animasyon zamanlayıcısı
        timeTracker += 0.02f;

        // --- A) EKRANI TEMİZLE ---
        // Video yüklemediğimiz için, her karenin başında VRAM tuvalimizi (0) siyah ile dolduruyoruz.
        cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

        // --- B) SİMÜLASYON: Objeyi Döndür ---
        auto& mutableObjects = const_cast<std::vector<Object3D>&>(myScene.getObjects());
        if (!mutableObjects.empty()) {
            mutableObjects[0].rotation.y = timeTracker;         // Kendi etrafında fırıl fırıl (Yaw)
            mutableObjects[0].rotation.x = timeTracker * 0.3f;  // Hafifçe de öne arkaya yatsın (Pitch)
        }

        // --- C) GRAFİK: 3D Çizimi Yap ---
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // --- D) EKRANA YANSIT ---
        unsigned char* d_pbo_vram_address = target.mapVRAM();
        visionEngine.copyToDeviceUchar(d_pbo_vram_address);
        target.unmapAndRender();

        // FPS Sabitleyici ve Ekrana Yazdırma
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