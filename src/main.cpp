#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
// MOTOR BAŞLIK DOSYALARI
#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Renderer3D.cuh"
#include "io/GlfwInteropTarget.h"

// YENİ MİMARİ: ORKESTRA ŞEFİ
#include "io/Graphics/ObjectLoader.h"
#include "io/Graphics/KvlcmParser.h"
#include "io/Graphics/SceneDescription.h"
#include "io/Graphics/SceneBuilder.h"

// NVIDIA Optimus sürücüsünü bu EXE için ayrık GPU'yu (RTX 4070) kullanmaya zorlar
#ifdef _WIN32
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}
#endif

int main() {
    std::cout << "\n================ [RADAR TESTI] ================\n";
    if (std::filesystem::exists("assets-graphics")) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator("assets-graphics")) {
            if (!entry.is_directory()) {
                // Windows'un ters slajlarını ( \ ), C++'ın düz slajına ( / ) çevir
                std::string yol = entry.path().generic_string();
                std::cout << "Motor Sunu Goruyor: " << yol << "\n";
            }
        }
    } else {
        std::cout << "KRIZ: assets-graphics klasoru tamamen kayip!\n";
    }
    std::cout << "===============================================\n\n";
    // 0. CUDA BAĞLAMI BAŞLATMA
    cudaSetDevice(0);
    cudaFree(0);

    // 1. MOTOR KURULUMU (Test Çözünürlüğü: 640x480)
    int width = 1280;
    int height = 720;
    int channels = 3;

    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Sahne Testi (640x480)");

    // Tuvali simsiyah (0) yapmak için VRAM'i temizliyoruz (Junk memory kalmasın)
    cudaMemset(visionEngine.getDeviceData(), 0, width * height * channels * sizeof(float));

    // ==============================================================================
    // 2. SAHNEYİ DOSYADAN YÜKLE (Tüm o devasa array'ler bu satırın içine hapsoldu!)
    // ==============================================================================
    // main.cpp içindeki o satırı şu şekilde güncelle:
    Scene myScene = SceneBuilder::build("assets-graphics/scenes/scene_01.kvlcm");
    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    std::cout << "[Main] Cizim dongusu basladi! Guc kilidi acildi." << std::endl;

    // ==============================================================================
    // 3. ANA ÇİZİM DÖNGÜSÜ
    // ==============================================================================
    while (!target.shouldClose()) {
        timeTracker += 0.02f;

        // A) SİMÜLASYON KATI: Sahnedeki tüm objeleri kendi ekseninde yavaşça döndür
        auto& mutableObjects = const_cast<std::vector<Object3D>&>(myScene.getObjects());
        for (auto& obj : mutableObjects) {
            obj.rotation.y = timeTracker;
        }

        // B) GRAFİK KATI: 3D objeleri siyah tuval üzerine çiz
        graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

        // C) EKRANA YANSIT
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

    std::cout << "\n[Main] Motor guvenle kapatildi." << std::endl;
    return 0;
}