#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "ElementaryMatrixOp.cuh"
#include "EngineFactory.cuh"
#include "OperationWrapper.cuh"

#include "stb_image.h"
#include "stb_image_write.h"

EngineFactory::EngineFactory(const char* filename) : d_data(nullptr) {
    std::cout << "[EngineFactory] Loading image: " << filename << "..." << std::endl;

    // 1. Resmi CPU'ya Yükle
    unsigned char* temp_cpu_data = stbi_load(filename, &width, &height, &channels, 0);
    if (!temp_cpu_data) {
        std::cerr << "Error: Failed to load image " << filename << std::endl;
        exit(1);
    }

    totalElementCount = width * height * channels;

    // 2. GPU Bellek Ayırma (Float veri için)
    allocateMemory();

    // 3. Normalizasyon İşlemi (Char -> Float)
    // Geçici olarak GPU'da unsigned char alanı oluştur
    unsigned char* d_temp_uchar;
    size_t ucharSizeBytes = totalElementCount * sizeof(unsigned char);
    cudaMalloc(&d_temp_uchar, ucharSizeBytes);

    // Veriyi kopyala (Host -> Device)
    cudaMemcpy(d_temp_uchar, temp_cpu_data, ucharSizeBytes, cudaMemcpyHostToDevice);

    // Grid/Block Hesabı
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel Çağrısı: Normalize Et
    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_temp_uchar, d_data, totalElementCount);
    cudaDeviceSynchronize();

    // 4. Temizlik (Geçici alanlar)
    cudaFree(d_temp_uchar);
    stbi_image_free(temp_cpu_data);

    std::cout << "[EngineFactory] Image loaded and normalized on GPU." << std::endl;
}

EngineFactory::~EngineFactory() {
    cleanUp();
}

void EngineFactory::allocateMemory() {
    size_t floatSizeBytes = totalElementCount * sizeof(float);
    cudaError_t err1 = cudaMalloc(&d_data, floatSizeBytes);
    cudaError_t err2 = cudaMalloc(&d_temp_data, floatSizeBytes);
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA Malloc Failed: " << cudaGetErrorString(err1) << cudaGetErrorString(err2)<< std::endl;
        exit(1);
    }
}

void EngineFactory::cleanUp() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
    if (d_temp_data) { cudaFree(d_temp_data); d_temp_data = nullptr; }
}



void EngineFactory::saveImage(const char* filename) {
    std::cout << "[EngineFactory] Saving to " << filename << "..." << std::endl;

    // 1. GPU'da Çıktı İçin Geçici Yer Ayır (unsigned char)
    unsigned char* d_output_uchar;
    size_t ucharSizeBytes = totalElementCount * sizeof(unsigned char);
    cudaMalloc(&d_output_uchar, ucharSizeBytes);

    // 2. Kernel Ayarları
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;

    // 3. Denormalize Et (Float -> Char)
    k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output_uchar, totalElementCount);
    cudaDeviceSynchronize();

    // 4. Sonucu CPU'ya Çek
    std::vector<unsigned char> host_output(totalElementCount);
    cudaMemcpy(host_output.data(), d_output_uchar, ucharSizeBytes, cudaMemcpyDeviceToHost);

    // 5. Dosyaya Yaz (PNG formatında)
    stbi_write_png(filename, width, height, channels, host_output.data(), width * channels);

    // 6. Temizlik
    cudaFree(d_output_uchar);
    std::cout << "[EngineFactory] Saved successfully!" << std::endl;
}

EngineFactory& EngineFactory::rgbToHsv() {
    // d_data'yı oku, d_temp_data'ya (HSV olarak) yaz
    OperationWrapper::rgbToHsv(d_data, d_temp_data, width, height, channels);

    // İşaretçileri (Pointers) takas et!
    // Artık asıl verimiz d_temp_data'nın gösterdiği yer oldu.
    // Hiçbir cudaMemcpy (kopyalama) yapmadan 0 zaman maliyetiyle işlemi bitirdik!
    std::swap(d_data, d_temp_data);

    return *this; // Zincirin devam etmesi için kendini döndür
}

EngineFactory& EngineFactory::hsvToRgb() {
    OperationWrapper::hsvToRgb(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

// --- FİLTRELER VE TON AYARLAMALARI (In-Place Mimarisi) ---

EngineFactory& EngineFactory::applyTemperature(float temperature) {
    // Temperature doğrudan orijinal d_data üzerinde (in-place) çalışır
    OperationWrapper::temperatureAdjustment(d_data, width, height, channels, temperature);
    return *this;
}

EngineFactory& EngineFactory::applyShadowsHighlights(float shadowAmount, float highlightAmount) {
    OperationWrapper::shadowsHighlightsAdjustment(d_data, width, height, channels, shadowAmount, highlightAmount);
    return *this;
}

EngineFactory& EngineFactory::applyGamma(float gamma) {
    OperationWrapper::gammaCorrectionAdjustment(d_data, width, height, channels, gamma);
    return *this;
}