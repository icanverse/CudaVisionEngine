#include <iostream>
#include <cuda_runtime.h>
#include "../../include/EngineFactory/EngineFactory.cuh"
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
    cudaError_t err1 = cudaMalloc(&d_data, floatSizeBytes);
    cudaError_t err2 = cudaMalloc(&d_temp_data, floatSizeBytes);
    cudaError_t err3 = cudaMalloc(&d_mask_data, floatSizeBytes);
    cudaError_t err4 = cudaMalloc(&d_global_min, sizeof(float));
    cudaError_t err5 = cudaMalloc(&d_global_max, sizeof(float));


    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess){
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




