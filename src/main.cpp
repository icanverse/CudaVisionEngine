#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "../include/main.h"
#include "EngineFactory.cuh"
#include "OperationWrapper.cuh"

int main() {
    std::cout << "[Main] İşlem başlatılıyor..." << std::endl;

    // --- 1. GÖRSEL YÜKLEME SÜRESİ (CPU/IO) ---
    auto start_load = std::chrono::high_resolution_clock::now();
    EngineFactory engine("assets/hotel.jpeg");
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> load_time = end_load - start_load;
    std::cout << "[Zamanlama] Görsel Yükleme: " << load_time.count() << " ms\n";

    int width = engine.getWidth();
    int height = engine.getHeight();
    int channels = engine.getChannels();
    size_t dataSize = width * height * channels * sizeof(float);

    // 2. GPU Bellek Yönetimi
    // Evrişim işleminde in-place çalışamadığımız için sadece BİR TANE ekstra çıktı belleğine ihtiyacımız var.
    float* d_output;
    cudaMalloc(&d_output, dataSize);

    // --- 3. DETAYLI GPU İŞLEM SÜRELERİ (CUDA EVENTS) ---
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);

    std::cout << "[Main] GPU üzerinde keskinleştirme (Sharpen) uygulanıyor..." << std::endl;

    // Kronometreyi başlat
    cudaEventRecord(t_start);

    // Keskinleştirme Filtresi (Doğrudan RGB üzerinde çalışır)
    OperationWrapper::sharpen(engine.getDeviceData(), d_output, width, height, channels);

    // Kronometreyi durdur ve bekle
    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);

    // Süreyi hesapla
    float ms_sharpen = 0;
    cudaEventElapsedTime(&ms_sharpen, t_start, t_stop);

    std::cout << "--------------------------------------------------\n";
    std::cout << "[GPU Zamanlama Detayları]\n";
    std::cout << "  -> Sharpen (Keskinleştirme) Kernel : " << ms_sharpen << " ms\n";
    std::cout << "--------------------------------------------------\n";
    // -----------------------------------------

    // 4. Veriyi Güncelle (Artık orjinal resim yerine d_output'u tutuyoruz)
    engine.updateDeviceData(d_output);

    // --- 5. KAYDETME SÜRESİ (CPU/IO) ---
    auto start_save = std::chrono::high_resolution_clock::now();
    engine.saveImage("assets/hotel_output_sharpened.jpg");
    auto end_save = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> save_time = end_save - start_save;
    std::cout << "[Zamanlama] Görsel Kaydetme: " << save_time.count() << " ms\n";

    // 6. Temizlik
    cudaFree(d_output);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);

    std::cout << "[Main] İşlem tamamlandı! Çıktıyı kontrol et." << std::endl;
    return 0;
}