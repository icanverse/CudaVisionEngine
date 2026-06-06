#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream> // DOSYA OKUMA İÇİN EKLENDİ
#include <string>
#include <cuda_runtime.h>

// MOTOR BAŞLIK DOSYALARI
#include "EngineFactory/EngineFactory.cuh"
#include "Graphics/Scene.cuh"
#include "Graphics/Renderer3D.cuh"

// CANLI AKIŞ VE EKRAN ÇIKTISI BAŞLIKLARI
#include "io/NetworkStream.h"
#include "io/Video/NvDecoder.h"
#include "io/GlfwInteropTarget.h"

// NVIDIA Optimus sürücüsünü bu EXE için ayrık GPU'yu (RTX 4070) kullanmaya zorlar
#ifdef _WIN32
extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}
#endif

// --- YENİ: GÜVENLİ URL OKUYUCU FONKSİYON ---
std::string getRtspUrlFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::string url;
    if (file.is_open()) {
        std::getline(file, url);
        file.close();
    } else {
        std::cerr << "[Hata] " << filename << " dosyasi bulunamadi!" << std::endl;
        std::cerr << "[Bilgi] Lutfen proje ana dizininde bir '" << filename << "' dosyasi olusturup icine RTSP linkini yapistirin." << std::endl;
        exit(1); // Dosya yoksa programı güvenli bir şekilde durdur
    }
    return url;
}

int main() {
    std::cout << "[Main] Kivilcim Canli Telefon Yayini Modu Baslatiliyor..." << std::endl;

    // 0. CUDA ZORLA BAŞLATMA
    cudaSetDevice(0);
    cudaFree(0);

    // 1. CANLI AKIŞ ADRESİ (GÜVENLİ OKUMA)
    // URL artık kodun içinde değil, dışarıdaki bir text dosyasından okunuyor!
    std::string rtspUrl = getRtspUrlFromFile("NetworkConfig.txt");

    std::cout << "[Main] Telefona baglaniliyor: " << rtspUrl << std::endl;
    NetworkStream phoneInput(rtspUrl);
    NvDecoder decoder;

    int width = 1280;
    int height = 720;
    int channels = 3;

    // 2. MOTORLARI VE PENCEREYİ BAŞLAT
    EngineFactory visionEngine(width, height, channels);
    Renderer3D graphicsRenderer(width, height, channels);
    GlfwInteropTarget target(width, height, channels, "Kivilcim - Canli Telefon AR Entegrasyonu");

    // ==============================================================================
    // 3. SAHNEYİ KUR VE GEOMETRİLERİ TANIMLA
    // ==============================================================================
    Scene myScene;

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

    float3 stickVertices[8] = {
        {-0.02f, -0.02f, 0.0f}, { 0.02f, -0.02f, 0.0f}, { 0.02f,  0.02f, 0.0f}, {-0.02f,  0.02f, 0.0f},
        {-0.02f, -0.02f, 3.0f}, { 0.02f, -0.02f, 3.0f}, { 0.02f,  0.02f, 3.0f}, {-0.02f,  0.02f, 3.0f}
    };
    int3 stickIndices[12] = {
        {0,1,2}, {0,2,3}, {4,6,5}, {4,7,6}, {0,4,5}, {0,5,1},
        {3,2,6}, {3,6,7}, {0,3,7}, {0,7,4}, {1,5,6}, {1,6,2}
    };

    Material redPlastic = {{0.9f, 0.1f, 0.1f}, 0.1f, 0.8f, 0.3f, 16.0f};
    Material blueMetal  = {{0.2f, 0.4f, 0.9f}, 0.1f, 0.4f, 1.0f, 128.0f};
    Material axisRed   = {{1.0f, 0.0f, 0.0f}, 0.8f, 0.2f, 0.0f, 1.0f};
    Material axisGreen = {{0.0f, 1.0f, 0.0f}, 0.8f, 0.2f, 0.0f, 1.0f};
    Material axisBlue  = {{0.0f, 0.0f, 1.0f}, 0.8f, 0.2f, 0.0f, 1.0f};

    myScene.setCamera({0.0f, 1.5f, -6.0f}, {-0.22f, 0.0f, 0.0f})
           .addObject(starVertices, 10, starIndices, 16, {-1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, redPlastic)
           .addObject(starVertices, 10, starIndices, 16, { 1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, blueMetal)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {0.0f, -1.5708f, 0.0f}, axisRed)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {-1.5708f, 0.0f, 0.0f}, axisGreen)
           .addObject(stickVertices, 8, stickIndices, 12, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, axisBlue)
           .addLight({0.0f, 5.0f, -4.0f}, {1.0f, 1.0f, 1.0f}, 1.2f)
           .addLight({-3.0f, -2.0f, -3.0f}, {0.2f, 0.2f, 0.9f}, 0.5f);

    // ==============================================================================
    // 4. ANA CANLI AKIŞ DÖNGÜSÜ
    // ==============================================================================
    AVPacket packet;
    av_init_packet(&packet);

    auto t_start = std::chrono::high_resolution_clock::now();
    int frameCount = 0;
    float timeTracker = 0.0f;

    std::cout << "[Main] Canli dongu basladi. Gecikmesiz render aktif." << std::endl;

    while (!target.shouldClose()) {
        timeTracker += 0.02f;

        if (phoneInput.readLivePacket(&packet)) {
            decoder.decodePacket(packet.data, packet.size);
            av_packet_unref(&packet);
        }

        CUdeviceptr d_nv12Frame = 0;
        unsigned int pitch = 0;

        while (decoder.getDecodedFrame(&d_nv12Frame, &pitch)) {
            visionEngine.loadNV12DevicePointer(d_nv12Frame, pitch);

            auto& mutableObjects = const_cast<std::vector<Object3D>&>(myScene.getObjects());
            if (mutableObjects.size() >= 2) {
                mutableObjects[0].rotation.y = timeTracker;
                mutableObjects[0].rotation.x = timeTracker * 0.3f;
                mutableObjects[1].rotation.y = -timeTracker;
                mutableObjects[1].rotation.x = timeTracker * 0.3f;
            }

            graphicsRenderer.render(visionEngine.getDeviceData(), myScene, timeTracker);

            unsigned char* d_pbo_vram_address = target.mapVRAM();
            visionEngine.copyToDeviceUchar(d_pbo_vram_address);
            target.unmapAndRender();

            decoder.releaseFrame(d_nv12Frame);

            frameCount++;
            if (frameCount % 60 == 0) {
                auto t_end = std::chrono::high_resolution_clock::now();
                double fps = 1000.0 / (std::chrono::duration<double, std::milli>(t_end - t_start).count() / 60.0);
                std::cout << "Live AR FPS: " << std::fixed << std::setprecision(1) << fps << "    \r" << std::flush;
                t_start = std::chrono::high_resolution_clock::now();
            }
        }
        glfwPollEvents();
    }

    return 0;
}