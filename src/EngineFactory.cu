#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "ElementaryMatrixOp.cuh"
#include "EngineFactory.cuh"

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
    cudaError_t err = cudaMalloc(&d_data, floatSizeBytes);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc Failed: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void EngineFactory::cleanUp() {
    if (d_data) {
        cudaFree(d_data);
        d_data = nullptr;
    }
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