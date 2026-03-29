#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

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

    uint8_t* packetData = nullptr;
    int packetSize = 0;
    CUdeviceptr d_nv12Frame = 0;
    unsigned int pitch = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;
    int repHue = 0;

    // LOOP
    while (!target.shouldClose()) {

        // Kuryeden paketi al ve NVDEC donanımına fırlat
        if (demuxer.readPacket(&packetData, &packetSize)) {
            decoder.decodePacket(packetData, packetSize);
            demuxer.freePacket();
        }

        // VRAM'de çözülmüş kare varsa al ve Fluent Motoruna sok!
        while (decoder.getDecodedFrame(&d_nv12Frame, &pitch)) {
            float dynamicTemp = std::sin(timeTracker) * 0.5f;
            timeTracker += 0.02f;

            // VRAM'deki NV12 adresini besliyoruz
            engine.loadNV12DevicePointer(d_nv12Frame, pitch)
                  .rgbToHsv()
                  .colorReplacement(270, 70, repHue)
                  .hsvToRgb();

            repHue = repHue + 1;

            //  VRAM Kapısını Aç ve Hedef Adresi Al
            unsigned char* d_pbo_vram_address = target.mapVRAM();

            // Pikselleri VRAM'den PBO'ya YAZ
            engine.copyToDeviceUchar(d_pbo_vram_address);

            // Kapıyı Kapat ve Monitöre Çiz
            target.unmapAndRender();

            // NVDEC Donanımındaki Kareyi Serbest Bırak (Memory Leak önlemi)
            decoder.releaseFrame(d_nv12Frame);

            // Performans ve Gecikme (Latency) Ölçümü
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
