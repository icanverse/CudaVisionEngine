#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

// Eğer LUT okuyucu fonksiyonunu (loadCubeLUT) ayrı bir dosyaya yazdıysan buraya include et
// #include "io/LUTLoader.h"

#include "EngineFactory/EngineFactory.cuh"
#include "io/Video/Demuxer.h"
#include "io/Video/NvDecoder.h"
#include "io/GlfwInteropTarget.h"


int main() {
    std::cout << "[Main] ZERO-COPY Video Interop Motoru Baslatiliyor..." << std::endl;

    Demuxer demuxer("assets/PurpleModel.mp4");
    NvDecoder decoder;

    GlfwInteropTarget target(demuxer.getWidth(), demuxer.getHeight(), 3, "CudaVisionEngine - Fluent Video");
    EngineFactory engine(demuxer.getWidth(), demuxer.getHeight(), 3);

    // ==============================================================================
    // 1. OYUN DÖNGÜSÜ (GAME LOOP) ÖNCESİ HAZIRLIK (PRE-BAKING)
    // ==============================================================================
    std::vector<float> lutData;
    int lutSize = 0;

    std::cout << "[Main] MadMax LUT Dosyasi Okunuyor..." << std::endl;
    if (loadCubeLUT("assets/madmax.cube", lutData, lutSize)) {
        // CPU'da okunan veriyi, GPU'daki 3D Texture Donanımına (lutTexture) Fırınla!
        // Not: init3DTextureMemory metodunu EngineFactory.cuh içinde public yaptığını varsayıyoruz.
        engine.init3DTextureMemory(lutData.data(), lutSize, engine.d_lutArray, engine.lutTexture);
    } else {
        std::cerr << "[HATA] LUT Dosyasi yuklenemedi! Varsayilan renklerle devam ediliyor." << std::endl;
    }

    uint8_t* packetData = nullptr;
    int packetSize = 0;
    CUdeviceptr d_nv12Frame = 0;
    unsigned int pitch = 0;

    // Animasyon Değişkenleri
    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;
    float repHue = 10.0f;

    // ==============================================================================
    // 2. ANA OYUN DÖNGÜSÜ (150+ FPS)
    // ==============================================================================
    while (!target.shouldClose()) {

        // Kuryeden paketi al ve NVDEC donanımına fırlat
        if (demuxer.readPacket(&packetData, &packetSize)) {
            decoder.decodePacket(packetData, packetSize);
            demuxer.freePacket();
        }

        // VRAM'de çözülmüş kare varsa al ve Fluent Motoruna sok!
        while (decoder.getDecodedFrame(&d_nv12Frame, &pitch)) {

            // Dinamik Animasyon Matematiği
            repHue = std::fmod(repHue + 2.0f, 360.0f);
            timeTracker += 0.03f;
            float flareX = (std::sin(timeTracker) * 350.0f) + (demuxer.getWidth() / 2.0f);
            float flareY = (std::sin(timeTracker * 2.0f) * 200.0f) + (demuxer.getHeight() / 2.0f);

            // ==============================================================================
            // 3. FLUENT MOTOR (GPU SİHRİ)
            // ==============================================================================
            engine.loadNV12DevicePointer(d_nv12Frame, pitch);

            // Eğer LUT objesi başarıyla oluşturulduysa (sıfırdan farklıysa) renk motorunu çalıştır
            if (engine.lutTexture != 0) {
                engine.apply3DLUT(engine.lutTexture);
            }

            // ==============================================================================
            // 4. VRAM TRANSFER VE RENDER (SIFIR KOPYA)
            // ==============================================================================
            unsigned char* d_pbo_vram_address = target.mapVRAM();
            engine.copyToDeviceUchar(d_pbo_vram_address);
            target.unmapAndRender();

            // 5. TEMİZLİK (Memory Leak Önlemi)
            decoder.releaseFrame(d_nv12Frame);

            // Performans Ölçümü
            frameCount++;
            if (frameCount % 100 == 0) {
                auto t_end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

                double avg_latency_ms = elapsed_ms / 100.0;
                double fps = 1000.0 / avg_latency_ms;

                std::cout << "Guncel FPS: " << std::fixed << std::setprecision(1) << fps
                          << " | Gecikme: " << std::fixed << std::setprecision(2) << avg_latency_ms << " ms    \r" << std::flush;

                t_start = std::chrono::high_resolution_clock::now();
            }
        }

        // İşletim sisteminin pencereyi dondurmaması için GLFW eventlerini işle
        glfwPollEvents();
    }

    std::cout << "\nMotor basariyla kapatildi." << std::endl;
    return 0;
}