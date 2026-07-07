#include <iostream>
#include <cuda_runtime.h>
#include "../../include/EngineFactory/EngineFactory.cuh"

#include <string>

#include "OperationWrapper.cuh"
#include "Kernels/Normalization.cuh"

// Constructor sadece Bellek Ayırır
EngineFactory::EngineFactory(int w, int h, int c) : width(w), height(h), channels(c), d_data(nullptr), d_temp_data(nullptr), d_global_min(nullptr), d_global_max(nullptr) {
    totalElementCount = width * height * channels;
    allocateMemory();

    std::cout << "[EngineFactory] Motor hazir. VRAM rezerve edildi: " << width << "x" << height << std::endl;
}

EngineFactory::~EngineFactory() {
    cleanUp();
}
void EngineFactory::allocateMemory() {
    size_t floatSizeBytes = totalElementCount * sizeof(float);
    size_t flowSizeBytes = (width * height) * sizeof(float);
    cudaError_t err;

    auto checkErr = [&](cudaError_t e, const std::string& msg) {
        if (e != cudaSuccess) {
            std::cerr << "[CUDA HATA] " << msg << ": " << cudaGetErrorString(e) << std::endl;
            exit(1);
        }
    };

    checkErr(cudaMalloc(&d_data, floatSizeBytes), "d_data");
    checkErr(cudaMalloc(&d_temp_data, floatSizeBytes), "d_temp_data");
    checkErr(cudaMalloc(&d_mask_data, floatSizeBytes), "d_mask_data");
    checkErr(cudaMalloc(&d_global_min, sizeof(float)), "d_global_min");
    checkErr(cudaMalloc(&d_global_max, sizeof(float)), "d_global_max");
    checkErr(cudaMalloc(&d_prev_data, floatSizeBytes), "d_prev_data");
    checkErr(cudaMalloc(&d_flow_u, flowSizeBytes), "d_flow_u");
    checkErr(cudaMalloc(&d_flow_v, flowSizeBytes), "d_flow_v");

    // İlk kareler için temizlik
    cudaMemset(d_prev_data, 0, floatSizeBytes);
    cudaMemset(d_flow_u, 0, flowSizeBytes);
    cudaMemset(d_flow_v, 0, flowSizeBytes);
}

void EngineFactory::copyToDeviceUchar(unsigned char* d_dest_uchar) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;

    // Float'tan Char'a çevir ama sonucu RAM'e değil, bize verilen VRAM adresine (d_dest_uchar) yaz!
    k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_dest_uchar, totalElementCount);
    cudaDeviceSynchronize();
}
void EngineFactory::initTextureMemory(cudaArray_t& targetArray, cudaTextureObject_t& targetTexture, int texWidth, int texHeight) {

    // Eğer önceden doluysa sızıntı (leak) olmaması için temizle
    if (targetTexture) { cudaDestroyTextureObject(targetTexture); targetTexture = 0; }
    if (targetArray) { cudaFreeArray(targetArray); targetArray = nullptr; }

    // 1. Kanal Formatı: float4 (R, G, B, A)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // 2. Array'i bize verilen "targetArray" referansına ayır!
    cudaError_t err = cudaMallocArray(&targetArray, &channelDesc, texWidth, texHeight);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocArray Hatasi: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    // 3. Kaynak Belirleyici
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = targetArray; // Evrensel array'i bağla

    // 4. Filtreleme Kuralları (Bilinear Smoothing)
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // 5. Objeyi Yarat ve "targetTexture" referansına bağla!
    err = cudaCreateTextureObject(&targetTexture, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "cudaCreateTextureObject Hatasi: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }

    std::cout << "[EngineFactory] Donanimsal Texture Bellek Uretildi (" << texWidth << "x" << texHeight << ")" << std::endl;
}

void EngineFactory::cleanUp() {
    if (d_data) { cudaFree(d_data); d_data = nullptr; }
    if (d_temp_data) { cudaFree(d_temp_data); d_temp_data = nullptr; }
    if (d_mask_data) { cudaFree(d_mask_data); d_mask_data = nullptr; }
    if (d_global_min) { cudaFree(d_global_min); d_global_min = nullptr; }
    if (d_global_max) { cudaFree(d_global_max); d_global_max = nullptr; }
    if (flareTexture) { cudaDestroyTextureObject(flareTexture);flareTexture = 0; }
    if (d_flareArray) {cudaFreeArray(d_flareArray);d_flareArray = nullptr; }

    if (d_prev_data) cudaFree(d_prev_data);
    if (d_flow_u) cudaFree(d_flow_u);
    if (d_flow_v) cudaFree(d_flow_v);
}

// RAM'den VRAM'e Veri Akışı
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

// VRAM'den RAM'e Veri Çekişi
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

//Grafik Kısmında Post Process Kullanımı için

// Parçacık motorundan (veya herhangi bir 8-bit VRAM kaynağından) veriyi devralır
EngineFactory& EngineFactory::loadFromVRAM(unsigned char* d_source_uchar) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;

    // 8-bit RGB (0-255) verisini Float (0.0f - 1.0f) formatına dönüştürüp d_data içine alır
    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_source_uchar, d_data, totalElementCount);
    cudaDeviceSynchronize();

    // Zincirleme reaksiyon (Fluent) devam etsin diye kendini dön
    return *this;
}

///
/// >>> Akıcı Arayüz -- Fluent Interface Mimarisi için
///

EngineFactory& EngineFactory::loadNV12DevicePointer(CUdeviceptr d_nv12, int pitch) {
    // Donanımdan gelen 8-bit (unsigned char) RGB verisini tutmak için geçici VRAM alanı ayır
    unsigned char* d_temp_uchar;
    size_t ucharSizeBytes = totalElementCount * sizeof(unsigned char);
    cudaMalloc(&d_temp_uchar, ucharSizeBytes);

    // NVDEC'ten gelen NV12'yi 8-bit RGB olarak d_temp_uchar içine çöz
    OperationWrapper::kernelNV12toRGB((const unsigned char*)d_nv12, d_temp_uchar, width, height, pitch);

    // 8-bit RGB'yi motorunun anladığı Float (0.0f - 1.0f) formatına dönüştür!
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElementCount + threadsPerBlock - 1) / threadsPerBlock;
    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_temp_uchar, d_data, totalElementCount);
    cudaDeviceSynchronize();

    // Geçici 8-bit alanını temizle
    cudaFree(d_temp_uchar);

    // Zincirleme reaksiyon (Fluent) devam etsin diye kendini dön
    return *this;
}

void EngineFactory::saveCurrentFrameAsPrevious() {
    // Mevcut d_data içeriğini d_prev_data'ya kopyala
    cudaMemcpy(d_prev_data, d_data, totalElementCount * sizeof(float), cudaMemcpyDeviceToDevice);
}

EngineFactory& EngineFactory::loadMesh(const float3* cpu_vertices, int numVerts, const int3* cpu_indices, int numTris) {
    this->numTriangles = numTris;

    // VRAM'de köşeler (vertices) için yer ayır ve kopyala
    size_t vertBytes = numVerts * sizeof(float3);
    cudaMalloc(&d_vertices, vertBytes);
    cudaMemcpy(d_vertices, cpu_vertices, vertBytes, cudaMemcpyHostToDevice);

    // VRAM'de üçgen bağları (indices) için yer ayır ve kopyala
    size_t indBytes = numTriangles * sizeof(int3);
    cudaMalloc(&d_indices, indBytes);
    cudaMemcpy(d_indices, cpu_indices, indBytes, cudaMemcpyHostToDevice);

    std::cout << "[EngineFactory] 3D Mesh yuklendi: " << numTriangles << " ucgen." << std::endl;
    return *this; // Zincirleme kullanım için
}

