#include <iostream>
#include <cuda_runtime.h>

#ifdef _WIN32
#include <windows.h>
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

// MOTOR BAŞLIK DOSYALARI
#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Renderer3D.cuh"
#include "io/GlfwInteropTarget.h"
#include "Graphics/Scene.cuh"
#include "io/Graphics/SceneBuilder.h"
#include "Compute/ParticleSystem/ParticleSystem.cuh"

// YENİ UI SİSTEMİ
#include "UI/MainUI.h"

int main() {
    std::cout << "[Kivilcim] Adim 1: Siber-Arayuz Penceresi Olusturuluyor...\n";
    int width = 1366, height = 768, channels = 3;
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sirca UI");

    std::cout << "[Kivilcim] Adim 2: CUDA Donanimi Uyandiriliyor...\n";
    cudaSetDevice(0);
    cudaFree(0);

    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);

    std::cout << "[Kivilcim] Adim 3: 3D Sahne ve Materyaller Yukleniyor...\n";
    Scene myScene = SceneBuilder::build("assets-graphics/scenes/scene_ui.kvlcm");

    std::cout << "[Kivilcim] Adim 3.5: Parcacik Fiziği Baslatiliyor...\n";
    ParticleSystem kivilcimSistemi(50);

    cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

    std::cout << "[Kivilcim] Adim 4: ImGui Sirca Arayuzu Kuruluyor...\n";
    // UI Sınıfımızı Başlatıyoruz
    MainUI sircaUI(target.getWindow());

    float timeTracker = 0.0f;
    std::cout << "[Kivilcim] Motor Aktif. Sistem hazir.\n";

    // --- ANA DÖNGÜ ---
    while (!target.shouldClose()) {
        glfwPollEvents();

        // UI Yeni Kare
        sircaUI.newFrame();

        // Panelleri Çizdir
        sircaUI.renderPanels();

        // Motor Fiziği ve Render
        timeTracker += 0.016f;
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);
        kivilcimSistemi.update(0.016f, timeTracker);

        // VRAM Katmanlarını Birleştir
        unsigned char* d_pbo = target.mapVRAM();
        if (d_pbo) {
            visionEngine.copyToDeviceUchar(d_pbo);
            kivilcimSistemi.draw(d_pbo, width, height);
        }
        target.unmapAndRender();

        // UI Verisini PBO'nun Üstüne Bas
        sircaUI.renderDrawData();

        glfwSwapBuffers(target.getWindow());
    }

    // MainUI'nin Yıkıcısı (Destructor) kapanış işlerini otomatik halleder.
    return 0;
}