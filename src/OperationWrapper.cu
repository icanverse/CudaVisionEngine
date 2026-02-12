//
// Created by Can on 13.02.2026.
//

#include "OperationWrapper.cuh"
#include "ElementaryMatrixOp.cuh"
#include "Smoothing.cuh"
#include <cstdio>
// --- Implementasyon ---

void OperationWrapper::normalize(unsigned char* d_input, float* d_output, int width, int height) {
    int totalElements = width * height;

    // 1D Grid Hesabı
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    k_normalizeImage<<<gridSize, blockSize>>>(d_input, d_output, totalElements);

    checkKernelError("Normalize Image");
}

void OperationWrapper::denormalize(float* d_input, unsigned char* d_output, int width, int height) {
    int totalElements = width * height;

    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    k_denormalizeImage<<<gridSize, blockSize>>>(d_input, d_output, totalElements);

    checkKernelError("Denormalize Image");
}

void OperationWrapper::smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize) {
    // 1. Blok Boyutlarını Belirle (Genellikle 16x16 veya 32x32 idealdir)
    dim3 blockSize(16, 16);

    // 2. Izgara (Grid) Boyutlarını Hesapla
    // (width + 15) / 16 formülü, tam bölünmeyen durumlarda fazladan blok eklenmesini sağlar
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // 3. Kernel'ı Başlat
    // Not: A ve Result işaretçilerinin zaten GPU (Device) belleğinde olduğu varsayılmaktadır.
    ::smoothing2D<<<gridSize, blockSize>>>(A, Result, width, height, channels, kernelSize);

    // 4. Hata Kontrolü (Opsiyonel ama Tavsiye Edilir)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Gerekirse senkronizasyon (İsteğe bağlı, CPU'nun devam etmeden beklemesini sağlar)
    // cudaDeviceSynchronize();
}

void OperationWrapper::add(const float* d_A, const float* d_B, float* d_C, int size, bool useSharedMem) {
    // 2D Grid Hesabı
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    if (useSharedMem) {
        // op_control = true (Toplama)
       // matrix_add_with_sharedmem<<<grid, block>>>(d_A, d_B, d_C, size, true);
    } else {
        // Naive versiyon
        matrix_add<<<grid, block>>>(d_A, d_B, d_C, size);
    }

    checkKernelError("Matrix Add");
}

void OperationWrapper::subtract(const float* d_A, const float* d_B, float* d_C, int size) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    // op_control = false (Çıkarma) - Sadece shared mem kernelinde implemente edilmiş
    //matrix_add_with_sharedmem<<<grid, block>>>(d_A, d_B, d_C, size, false);

    checkKernelError("Matrix Subtract");
}

void OperationWrapper::multiply(const float* d_A, const float* d_B, float* d_C, int size) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    matrix_mul<<<grid, block>>>(d_A, d_B, d_C, size);

    checkKernelError("Matrix Multiply");
}

void OperationWrapper::getSubMatrix(const float* d_in, float* d_out, int removeCol, int removeRow, int currentSize) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((currentSize + block.x - 1) / block.x, (currentSize + block.y - 1) / block.y);

    //findSubMatrix<<<grid, block>>>(d_in, d_out, removeCol, removeRow, currentSize);

    checkKernelError("Get SubMatrix");
}