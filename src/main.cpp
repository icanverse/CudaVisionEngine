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



    // Döngüye girmeden ÖNCE animasyon değişkenlerini tanımla

    auto t_start = std::chrono::high_resolution_clock::now();

    int frameCount = 0;

    float timeTracker = 0.0f;

    float repHue = 10.0f; // <-- Buraya taşıdık!



    // LOOP

    while (!target.shouldClose()) {



        // Kuryeden paketi al ve NVDEC donanımına fırlat

        if (demuxer.readPacket(&packetData, &packetSize)) {

            decoder.decodePacket(packetData, packetSize);

            demuxer.freePacket();

        }



        // VRAM'de çözülmüş kare varsa al ve Fluent Motoruna sok!

        // VRAM'de çözülmüş kare varsa al ve Fluent Motoruna sok!
        while (decoder.getDecodedFrame(&d_nv12Frame, &pitch)) {

            // 1. RENK ANİMASYONU HESABI
            // Hue değerini artır ve 360'ta sıfırla (pürüzsüz döngü)
            repHue = std::fmod(repHue + 2.0f, 360.0f);

            // 2. DİNAMİK IŞIK KOORDİNATLARI
            timeTracker += 0.03f;
            float flareX = (std::sin(timeTracker) * 350.0f) + (demuxer.getWidth() / 2.0f);
            float flareY = (std::sin(timeTracker * 2.0f) * 200.0f) + (demuxer.getHeight() / 2.0f);

            // 3. FLUENT MOTOR (KRİTİK ZİNCİR)
            engine.loadNV12DevicePointer(d_nv12Frame, pitch)
                  .rgbToHsv()                            // <--- Önce HSV'ye geç (ZORUNLU)
                  .colorReplacement(270, 70, repHue)     // Moru bul (270), yeni renge (repHue) boya
                  .hsvToRgb()                            // <--- Geri RGB'ye dön (ZORUNLU)
                  .renderProceduralFlare(flareX, flareY, 30.0f, 0.9f); // Işığı en son ekle

            // 4. VRAM TRANSFER VE RENDER
            unsigned char* d_pbo_vram_address = target.mapVRAM();
            engine.copyToDeviceUchar(d_pbo_vram_address);
            target.unmapAndRender();

            // 5. TEMİZLİK
            decoder.releaseFrame(d_nv12Frame);

            // ... (Performans ölçümü aynı kalacak) ...






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