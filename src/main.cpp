#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include "EngineFactory.cuh"
#include "io/Video/Demuxer.h"
#include "io/Video/NvDecoder.h"
#include "io/GlfwInteropTarget.h"

int main() {
    std::cout << "[Main] ZERO-COPY Video Interop Motoru Baslatiliyor..." << std::endl;

    // 1. Kurye ve Donanım Çözücü Hazırlığı
    Demuxer demuxer("assets/PurpleModel.mp4");
    NvDecoder decoder;

    // 2. Monitör ve Efekt Motoru (Senin Sınıfların)
    GlfwInteropTarget target(demuxer.getWidth(), demuxer.getHeight(), 3, "CudaVisionEngine - Fluent Video");
    EngineFactory engine(demuxer.getWidth(), demuxer.getHeight(), 3);

    uint8_t* packetData = nullptr;
    int packetSize = 0;
    CUdeviceptr d_nv12Frame = 0;
    unsigned int pitch = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    int repHue = 0;
    // THE GAME LOOP
    while (!target.shouldClose()) {

        // A) Kuryeden paketi al ve NVDEC donanımına fırlat
        if (demuxer.readPacket(&packetData, &packetSize)) {
            decoder.decodePacket(packetData, packetSize);
            demuxer.freePacket();
        }

        // B) VRAM'de çözülmüş kare varsa al ve senin Fluent Motoruna sok!
        while (decoder.getDecodedFrame(&d_nv12Frame, &pitch)) {

            float dynamicTemp = std::sin(timeTracker) * 0.5f;
            timeTracker += 0.02f;

            // İŞTE SENİN MİMARİN!
            // CPU'dan uploadFrame yapmak yerine doğrudan VRAM'deki NV12 adresini besliyoruz
            engine.loadNV12DevicePointer(d_nv12Frame, pitch)
                  .rgbToHsv()
                  .colorReplacement(270,70,repHue)
                  .hsvToRgb();
            repHue = repHue + 1;
            // C) VRAM Kapısını Aç ve Hedef Adresi Al
            unsigned char* d_pbo_vram_address = target.mapVRAM();

            // D) Pikselleri VRAM'den PBO'ya YAZ
            engine.copyToDeviceUchar(d_pbo_vram_address);

            // E) Kapıyı Kapat ve Monitöre Çiz
            target.unmapAndRender();

            // F) NVDEC Donanımındaki Kareyi Serbest Bırak (Memory Leak olmaması için hayati önem taşır)
            decoder.releaseFrame(d_nv12Frame);

            // G) Performans ve Gecikme (Latency) Ölçümü
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