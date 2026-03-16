#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip> // std::setprecision için eklendi
#include "io/StbImageSource.h"
#include "io/GlfwInteropTarget.h"
#include "EngineFactory.cuh"

int main() {
    std::cout << "[Main] ZERO-COPY Interop Motoru Baslatiliyor..." << std::endl;

    StbImageSource source("assets/starwars.jpg");
    unsigned char* rawFrame = source.grabNextFrame();
    if (!rawFrame) return -1;

    // Sıfır Gecikmeli Interop Monitörünü Başlat
    GlfwInteropTarget target(source.getWidth(), source.getHeight(), source.getChannels(), "CudaVisionEngine - Zero Copy");

    EngineFactory engine(source.getWidth(), source.getHeight(), source.getChannels());

    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    // THE GAME LOOP
    // THE GAME LOOP
    while (!target.shouldClose()) {
        // Dinamik değerlerimizi hesaplayalım
        // Sinüs dalgası -1 ile 1 arası döner, biz bunu 1.5 merkezli (0.5 - 2.5) Gamma'ya çevirelim
        float dynamicGamma = 1.5f + std::sin(timeTracker) * 1.0f;
        timeTracker += 0.02f;

        // 1. İşlemleri GPU'da Yap (Fluent Interface)
        engine.uploadFrame(rawFrame)
              .applyBoxBlur()       // Gürültüyü temizle
              .applyEdgeDetection() // Kenarları bul (Neon etkisi)
              .applySharpen()
                  .applyEmboss()          // Çizgileri belirginleştir
              .applyGamma(dynamicGamma); // NABIZ ETKİSİ: Dinamik parlaklık ve kontrast

        // 2. VRAM Kapısını Aç ve Hedef Adresi Al
        unsigned char* d_pbo_vram_address = target.mapVRAM();

        // 3. Pikselleri VRAM'den VRAM'e YAZ
        engine.copyToDeviceUchar(d_pbo_vram_address);

        // 4. Kapıyı Kapat ve Monitöre Çiz
        target.unmapAndRender();

        // Performans takibi aynı kalıyor...
        frameCount++;
        if (frameCount % 100 == 0) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            double avg_latency_ms = elapsed_ms / 100.0; // Kare başına düşen gecikme
            double fps = 1000.0 / avg_latency_ms;       // Saniyedeki kare sayısı

            std::cout << "Guncel FPS: " << std::fixed << std::setprecision(1) << fps
                      << " | Gecikme: " << std::fixed << std::setprecision(2) << avg_latency_ms << " ms    \r" << std::flush;

            t_start = std::chrono::high_resolution_clock::now();
        }
    }

    source.releaseFrame(rawFrame);
    std::cout << "\nMotor basariyla kapatildi." << std::endl;
    return 0;
}