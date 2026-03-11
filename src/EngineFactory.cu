#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "ElementaryMatrixOp.cuh"
#include "EngineFactory.cuh"
#include "OperationWrapper.cuh"

// DİKKAT: stb_image.h ve stb_image_write.h TAMAMEN SİLİNDİ!

// 1. TERTEMİZ CONSTRUCTOR (Sadece Bellek Ayırır)
EngineFactory::EngineFactory(int w, int h, int c) : width(w), height(h), channels(c), d_data(nullptr), d_temp_data(nullptr) {
    totalElementCount = width * height * channels;
    allocateMemory();
    std::cout << "[EngineFactory] Motor hazir. VRAM rezerve edildi: " << width << "x" << height << std::endl;
}

EngineFactory::~EngineFactory() {
    cleanUp();
}

void EngineFactory::allocateMemory() {
    size_t floatSizeBytes = totalElementCount * sizeof(float);
    cudaError_t err1 = cudaMalloc(&d_data, floatSizeBytes);
    cudaError_t err2 = cudaMalloc(&d_temp_data, floatSizeBytes);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        // Hata formatı düzeltildi (Araya " | " konuldu)
        std::cerr << "CUDA Malloc Failed: " << cudaGetErrorString(err1) << " | " << cudaGetErrorString(err2) << std::endl;
        exit(1);
    }
}

void EngineFactory::copyToDeviceUchar(unsigned char* d_dest_uchar) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;

    // Float'tan Char'a çevir ama sonucu RAM'e değil, bize verilen VRAM adresine (d_dest_uchar) yaz!
    k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_dest_uchar, totalElementCount);
    cudaDeviceSynchronize();
}

void EngineFactory::cleanUp() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
    if (d_temp_data) { cudaFree(d_temp_data); d_temp_data = nullptr; }
}

// 2. RAM'den VRAM'e Veri Akışı (Eski Constructor'daki STB Load mantığı buraya taşındı)
EngineFactory& EngineFactory::uploadFrame(const unsigned char* cpu_data) {
    unsigned char* d_temp_uchar;
    size_t ucharSizeBytes = totalElementCount * sizeof(unsigned char);
    cudaMalloc(&d_temp_uchar, ucharSizeBytes);

    // RAM'den VRAM'e hızlı kopyalama
    cudaMemcpy(d_temp_uchar, cpu_data, ucharSizeBytes, cudaMemcpyHostToDevice);

    // Float'a çevirme ve normalize etme (0.0f - 1.0f)
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;
    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_temp_uchar, d_data, totalElementCount);
    cudaDeviceSynchronize();

    cudaFree(d_temp_uchar);
    return *this;
}

// 3. VRAM'den RAM'e Veri Çekişi (Eski saveImage'deki STB Write mantığı buraya taşındı)
void EngineFactory::downloadFrame(unsigned char* cpu_data) {
    unsigned char* d_output_uchar;
    size_t ucharSizeBytes = totalElementCount * sizeof(unsigned char);
    cudaMalloc(&d_output_uchar, ucharSizeBytes);

    // Char'a çevirme ve denormalize etme (0 - 255)
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;
    k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output_uchar, totalElementCount);
    cudaDeviceSynchronize();

    // VRAM'den RAM'e kopyalama (CPU'ya teslim et)
    cudaMemcpy(cpu_data, d_output_uchar, ucharSizeBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_output_uchar);
}

// --- AKICI ARAYÜZ (FLUENT INTERFACE) FONKSİYONLARI ---

EngineFactory& EngineFactory::rgbToHsv() {
    OperationWrapper::rgbToHsv(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::hsvToRgb() {
    OperationWrapper::hsvToRgb(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applyTemperature(float temperature) {
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