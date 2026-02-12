#include <iostream>
#include <cuda_runtime.h>
#include "../include/main.h"
#include "EngineFactory.cuh"
#include "OperationWrapper.cuh" // smoothing2D wrapper'ının olduğu dosya

int main() {
    // 1. Motoru Başlat (Resim yüklenir, GPU'ya atılır, normalize edilir)
    EngineFactory engine("assets/blonde.jpg");

    // 2. Gerekli Bilgileri Çek
    int width = engine.getWidth();
    int height = engine.getHeight();
    int channels = engine.getChannels();

    // İşlem yapılacak veri sayısı
    size_t numElements = width * height * channels;
    size_t dataSize = numElements * sizeof(float);

    // 3. Çıktı İçin GPU'da GEÇİCİ Yer Ayır
    // (Smoothing işlemi Input -> Output şeklinde yapılmalıdır)
    float* d_output_temp;
    cudaMalloc(&d_output_temp, dataSize);

    // 4. Kernel İşlemini Uygula
    // Input: engine.getDeviceData()
    // Output: d_output_temp
    int kernelSize = 3; // Örnek: 15x15 blur (tek sayı olmalı)

    std::cout << "[Main] Applying Smoothing Kernel (" << kernelSize << "x" << kernelSize << ")..." << std::endl;

    OperationWrapper::smoothing2D(
        engine.getDeviceData(),
        d_output_temp,
        width,
        height,
        channels,
        kernelSize
    );

    // Hata kontrolü ve senkronizasyon
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }

    // 5. Sonucu Engine'e Geri Yükle
    // İşlenmiş veriyi (d_output_temp) engine'in kendi d_data'sına kopyalarız.
    // Böylece saveImage çağırdığımızda işlenmiş halini kaydeder.
    engine.updateDeviceData(d_output_temp);

    // 6. Resmi Kaydet
    engine.saveImage("assets/output_smoothed.png");

    // 7. Temizlik
    // Sadece main içinde oluşturduğumuz geçici diziyi siliyoruz.
    // EngineFactory kendi d_data'sını destructor'ında zaten silecek.
    cudaFree(d_output_temp);

    return 0;
}