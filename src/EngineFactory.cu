#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "ElementaryMatrixOp.cuh"
#include "EngineFactory.cuh"
#include "OperationWrapper.cuh"

// Constructor sadece Bellek Ayırır
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

///
/// >>> Akıcı Arayüz -- Fluent Interface Mimarisi için
///


// >
// Renk Uzayı Dönüşümleri
//

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

EngineFactory& EngineFactory::rgbToYuv() {
    OperationWrapper::rgbToYuv(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::yuvToRgb() {
    OperationWrapper::yuvToRgb(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::loadNV12DevicePointer(CUdeviceptr d_nv12, int pitch) {
    // 1. Donanımdan gelen 8-bit (unsigned char) RGB verisini tutmak için geçici VRAM alanı ayır
    unsigned char* d_temp_uchar;
    size_t ucharSizeBytes = totalElementCount * sizeof(unsigned char);
    cudaMalloc(&d_temp_uchar, ucharSizeBytes);

    // 2. NVDEC'ten gelen NV12'yi 8-bit RGB olarak d_temp_uchar içine çöz
    // (DİKKAT: '.' yerine '::' kullandık)
    OperationWrapper::kernelNV12toRGB((const unsigned char*)d_nv12, d_temp_uchar, width, height, pitch);

    // 3. SİHİRLİ DOKUNUŞ: 8-bit RGB'yi senin motorunun anladığı Float (0.0f - 1.0f) formatına dönüştür!
    // (Tıpkı uploadFrame metodunda yaptığın gibi)
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;
    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_temp_uchar, d_data, totalElementCount);
    cudaDeviceSynchronize();

    // 4. Geçici 8-bit alanını temizle (Memory Leak olmaması için)
    cudaFree(d_temp_uchar);

    // Zincirleme reaksiyon (Fluent) devam etsin diye kendini dön
    return *this;
}

// Renk Uzayına Bağlı Hazır Gelişmiş İşlemler

EngineFactory& EngineFactory::isolateColor(float targetHue, float tolerance) {
    OperationWrapper::isolateColor(d_data, width, height, channels, targetHue, tolerance);
    return *this;
}

EngineFactory &EngineFactory::colorReplacement(float targetHue, float tolerance, float replacementHue) {
    OperationWrapper::colorReplacement(d_data, width, height, channels, targetHue, tolerance,  replacementHue);
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

EngineFactory& EngineFactory::applyEmboss() {
    OperationWrapper::applyEmboss(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applyBoxBlur() {
    OperationWrapper::applyBoxBlur(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data); // Sıfır maliyetle kopyalama!
    return *this;
}

EngineFactory& EngineFactory::applySharpen() {
    OperationWrapper::applySharpen(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

EngineFactory& EngineFactory::applyEdgeDetection() {
    OperationWrapper::applyEdgeDetection(d_data, d_temp_data, width, height, channels);
    std::swap(d_data, d_temp_data);
    return *this;
}

