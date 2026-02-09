#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "../include/ElementaryMatrixOp.cuh"

class GeneralOperations {
    private:
        // Görsel Özellikleri
        unsigned int width;
        unsigned int height;
        unsigned int depth;
        unsigned int channels;
        size_t totalSizeBytes;

        // Pointer
        float* h_data; // Host (Pinned Memory)
        float* d_data; // Device (GPU)

    /// Ortak Bellek ayırma fonksyionu
    void allocateMemory() {
        if (depth == 0) {
            depth = 1;
            bool isHaveDepth = false;
            std::cout << "Have Not Depth" << std::endl;
        }

        size_t size = (size_t)width * height * depth * channels;    // Boyut
        totalSizeBytes = size * sizeof(float);                      // Toplam Boyut

        std::cout << "Allocating " << totalSizeBytes / 1024.0 / 1024.0 << " MB" << std::endl;

        // Pinned Memory
        cudaError_t errHost = cudaMallocHost(&h_data, totalSizeBytes);
        cudaError_t errDev = cudaMalloc(&d_data, totalSizeBytes);

        if (errHost != cudaSuccess || errDev != cudaSuccess) {
            std::cerr << "Memory allocation failed!" << std::endl;
            exit(1);
        }
    }
public:
    GeneralOperations(const char* filename) {
        std::cout << "Loading image: " << filename << "..." << std::endl;

        int w, h, c;
        // Resmi yüklüyoruz
        unsigned char* temp_cpu_data = stbi_load(filename, &w, &h, &c, 0);
        if (!temp_cpu_data) {
            std::cerr << "Error: Failed to load image " << filename << std::endl;
            exit(1);
        }

        width = w;
        height = h;
        channels = c;
        depth = 1;
        allocateMemory(); // d_data (float) için yer ayırıyoruz

        // GPU için geçici yer ayır
        unsigned char* d_temp_uchar;
        size_t ucharSizeBytes = width * height * channels * sizeof(unsigned char);
        cudaMalloc(&d_temp_uchar, ucharSizeBytes);

        // Host ---> Device (Kopyalama)
        cudaMemcpy(d_temp_uchar, temp_cpu_data, ucharSizeBytes, cudaMemcpyHostToDevice);

        int totalElements = width * height * channels;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

        std::cout << "Kernel direkt cagrisi yapiliyor (Grid: " << blocksPerGrid << ")..." << std::endl;

        k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_temp_uchar, d_data, totalElements);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel Hatasi: " << cudaGetErrorString(err) << std::endl;
        }

        cudaDeviceSynchronize();

        cudaFree(d_temp_uchar);         // GPU'daki char alanını sil
        stbi_image_free(temp_cpu_data); // CPU'daki stb verisini sil

        std::cout << "Image loaded to GPU successfully." << std::endl;
    }

    ~GeneralOperations() {
        if (d_data) cudaFree(d_data);
        if (h_data) cudaFreeHost(h_data);
    }

    void saveImage(const char* filename) {
        std::cout << "Saving to " << filename << " using GPU Denormalization..." << std::endl;

        int totalElements = width * height * channels;

        // 1. GPU'da Çıktı İçin Geçici Yer Ayır (unsigned char)
        unsigned char* d_output_uchar;
        size_t ucharSizeBytes = totalElements * sizeof(unsigned char);
        cudaMalloc(&d_output_uchar, ucharSizeBytes);

        // 2. Kernel Ayarları
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

        // 3. KERNEL ÇAĞIR (Denormalize: Float -> Char)
        // d_data (Input Float) ---> d_output_uchar (Output Char)
        k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output_uchar, totalElements);

        cudaDeviceSynchronize(); // Bitmesini bekle

        // 4. Sonucu CPU'ya Çek (Sadece char halini çekiyoruz, float halini değil!)
        std::vector<unsigned char> host_output(totalElements);
        cudaMemcpy(host_output.data(), d_output_uchar, ucharSizeBytes, cudaMemcpyDeviceToHost);

        // 5. Dosyaya Yaz (stb_image_write)
        stbi_write_png(filename, width, height, channels, host_output.data(), width * channels);

        // 6. Temizlik (Geçici GPU alanını sil)
        cudaFree(d_output_uchar);

        std::cout << "Saved successfully!" << std::endl;
    }

    void downloadToHost() {
        cudaMemcpy(h_data, d_data, totalSizeBytes, cudaMemcpyDeviceToHost);
    }

};

int main() {
    GeneralOperations myImage("blonde.jpg");
    myImage.saveImage("output_direct_kernel.png");
    return 0;
}