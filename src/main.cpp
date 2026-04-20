#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <thread> // YENİ EKLENDİ: İşlemciyi uyutmak (sleep) için gerekli kütüphane

// Eğer LUT okuyucu fonksiyonunu (loadCubeLUT) ayrı bir dosyaya yazdıysan buraya include et
// #include "io/LUTLoader.h"

#include "EngineFactory/EngineFactory.cuh"
#include "io/Video/Demuxer.h"
#include "io/Video/NvDecoder.h"
#include "io/GlfwInteropTarget.h"
#include "Kernels/VectorFieldVisualization2D.cuh"

int main() {
    std::cout << "[Main] ZERO-COPY Video Interop Motoru Baslatiliyor..." << std::endl;

    // Kendi test videonu buraya girdin
    Demuxer demuxer("assets/movcar.mp4");
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
        engine.init3DTextureMemory(lutData.data(), lutSize, engine.d_lutArray, engine.lutTexture);
    } else {
        std::cerr << "[HATA] LUT Dosyasi yuklenemedi! Varsayilan renklerle devam ediliyor." << std::endl;
    }

    uint8_t* packetData = nullptr;
    int packetSize = 0;
    CUdeviceptr d_nv12Frame = 0;
    unsigned int pitch = 0;

    // ==============================================================================
    // YENİ EKLENDİ: HIZ SINIRLAYICI AYARLARI (FRAME PACING)
    // ==============================================================================
    double targetVideoFPS = 60.0; // Videonun orijinal hızı (Genelde 24, 30 veya 60 olur)
    auto target_frame_duration = std::chrono::duration<double, std::milli>(1000.0 / targetVideoFPS);

    // Animasyon Değişkenleri
    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;
    float repHue = 10.0f;

    // ==============================================================================
    // 2. ANA OYUN DÖNGÜSÜ
    // ==============================================================================
    while (!target.shouldClose()) {

        // Kuryeden paketi al ve NVDEC donanımına fırlat
        if (demuxer.readPacket(&packetData, &packetSize)) {
            decoder.decodePacket(packetData, packetSize);
            demuxer.freePacket();
        }

        // VRAM'de çözülmüş kare varsa al ve Fluent Motoruna sok!
        while (decoder.getDecodedFrame(&d_nv12Frame, &pitch)) {

            // YENİ EKLENDİ: O anki karenin işlenmeye başlandığı anı kaydet
            auto frame_start_time = std::chrono::high_resolution_clock::now();

            // Dinamik Animasyon Matematiği
            repHue = std::fmod(repHue + 2.0f, 360.0f);
            timeTracker += 0.03f;
            float flareX = (std::sin(timeTracker) * 350.0f) + (demuxer.getWidth() / 2.0f);
            float flareY = (std::sin(timeTracker * 2.0f) * 200.0f) + (demuxer.getHeight() / 2.0f);

            // ==============================================================================
            // 3. FLUENT MOTOR (GPU SİHRİ)
            // ==============================================================================
            engine.loadNV12DevicePointer(d_nv12Frame, pitch)
                  .applyOpticalFlowLucasKanade(1.0f)
                  .applyVectorFieldColoring(0.8f);

            // ==============================================================================
            // 4. VRAM TRANSFER VE RENDER (SIFIR KOPYA)
            // ==============================================================================
            unsigned char* d_pbo_vram_address = target.mapVRAM();
            engine.copyToDeviceUchar(d_pbo_vram_address);
            target.unmapAndRender();

            // 5. TEMİZLİK (Memory Leak Önlemi)
            decoder.releaseFrame(d_nv12Frame);

            // ==============================================================================
            // YENİ EKLENDİ: FPS SABİTLEYİCİ (Uyutma Mekanizması)
            // ==============================================================================
            auto frame_end_time = std::chrono::high_resolution_clock::now();
            auto processing_time = std::chrono::duration<double, std::milli>(frame_end_time - frame_start_time);

            // Eğer motor işini hedeflenen süreden (örneğin 33ms) daha çabuk bitirdiyse, kalan süre kadar uyu!
            if (processing_time < target_frame_duration) {
                std::this_thread::sleep_for(target_frame_duration - processing_time);
            }

            // Performans Ölçümü
            frameCount++;
            if (frameCount % 30 == 0) { // Logları 100 yerine 30 karede bir (saniyede 1) basalım
                auto t_end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

                double avg_latency_ms = elapsed_ms / 30.0;
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
