#include <iostream>
#include <cuda_runtime.h>
#include "../include/main.h"
#include "EngineFactory.cuh"
#include "OperationWrapper.cuh"

int main() {
    // 1. Motoru Başlat
    EngineFactory engine("assets/blonde.jpg");

    int width = engine.getWidth();
    int height = engine.getHeight();
    int channels = engine.getChannels();
    size_t dataSize = width * height * channels * sizeof(float);

    // 2. GPU Bellek Yönetimi
    float* d_hsv_temp;
    float* d_rgb_restored;
    cudaMalloc(&d_hsv_temp, dataSize);
    cudaMalloc(&d_rgb_restored, dataSize);

    // 3. RGB -> HSV Dönüşümü
    std::cout << "[Main] Converting RGB to HSV..." << std::endl;
    OperationWrapper::rgbToHsv(engine.getDeviceData(), d_hsv_temp, width, height, channels);
    cudaDeviceSynchronize();

    // --- COLOR REPLACEMENT FİLTRESİ ---
    // Örnek: Kırmızıları (0°) bul ve onları Siberpunk Mavisine (240°) çevir.
    float targetHue = 0.0f;       // Hedef: Kırmızı
    float tolerance = 35.0f;      // Tolerans: +- 35 derece
    float replacementHue = 240.0f; // Yeni Renk: Mavi

    std::cout << "[Main] Replacing Colors (Red -> Blue)..." << std::endl;
    OperationWrapper::colorReplacement(d_hsv_temp, width, height, channels, targetHue, tolerance, replacementHue);
    cudaDeviceSynchronize();
    // ----------------------------------

    // 4. HSV -> RGB Geri Dönüş
    std::cout << "[Main] Restoring back to RGB..." << std::endl;
    OperationWrapper::hsvToRgb(d_hsv_temp, d_rgb_restored, width, height, channels);
    cudaDeviceSynchronize();

    // 5. Kaydet
    engine.updateDeviceData(d_rgb_restored);
    engine.saveImage("assets/output_color_replaced.png");

    // 6. Temizlik
    cudaFree(d_hsv_temp);
    cudaFree(d_rgb_restored);

    std::cout << "[Main] Done! Output: output_color_replaced.png" << std::endl;
    return 0;
}