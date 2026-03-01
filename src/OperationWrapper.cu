//
// Created by Can on 13.02.2026.
//

#include "OperationWrapper.cuh"
#include "ElementaryMatrixOp.cuh"
#include "Smoothing.cuh"
#include "ColorSpaceConverter.cuh"
#include <cstdio>

#include "ColorOperation.cuh"
#include "ToneAdjustment.cuh"

void OperationWrapper::calculateGrid(int width, int height, dim3& gridSize, dim3& blockSize) {
    blockSize = dim3(16, 16);
    gridSize = dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
}


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
    // 1. Blok Boyutlarını Belirle
    dim3 blockSize(16, 16);

    // 2. Izgara (Grid) Boyutlarını Hesapla
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // --- KRİTİK DÜZELTME BAŞLANGICI ---

    // 3. Shared Memory Boyutunu Hesapla
    // Kernel içinde kullandığımız formül: (Tile + 2*Radius) * (Tile + 2*Radius)
    int radius = kernelSize / 2;
    size_t sharedMemSize = (blockSize.x + 2 * radius) * (blockSize.y + 2 * radius) * sizeof(float);

    // 4. Kernel'ı Başlat (3. parametre olarak sharedMemSize eklendi)
    ::smoothing2D<<<gridSize, blockSize, sharedMemSize>>>(A, Result, width, height, channels, kernelSize);

    // --- KRİTİK DÜZELTME BİTİŞİ ---

    // 5. Hata Kontrolü
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in smoothing2D: %s\n", cudaGetErrorString(err));
    }

    // Kernel bitene kadar CPU'yu beklet (Debugging için iyidir)
    cudaDeviceSynchronize();
}

void OperationWrapper::rgbToHsv(const float* d_input, float* d_output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::rgbToHsv<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    checkKernelError("Convert RGB to HSV");

    cudaDeviceSynchronize();

}

void OperationWrapper::hsvToRgb(const float* d_input, float* d_output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::hsvToRgb<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    checkKernelError("Convert HSV to RGB");

    cudaDeviceSynchronize();
}

void OperationWrapper::isolateColor(float *d_hsv, int width, int height, int channels, float targetHue, float tolerance) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::isolateColor<<<gridSize, blockSize>>>(d_hsv, width, height, channels, targetHue, tolerance);

    checkKernelError("İsolate Color");

    cudaDeviceSynchronize();
}

void OperationWrapper::colorReplacement(float *d_hsv, int width, int height, int channels, float targetHue, float tolerance, float replacementHue) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::colorReplacement<<<gridSize, blockSize>>>(d_hsv, width, height, channels, targetHue, tolerance, replacementHue);

    checkKernelError("İsolate Color");

    cudaDeviceSynchronize();
}

void OperationWrapper::brightnessAdjustment(float *d_hsv, int width, int height, int channels, float value) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::brightnessAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, value);

    checkKernelError("Brightness Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::saturationAdjustment(float *d_hsv, int width, int height, int channels, float value) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::saturationAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, value);

    checkKernelError("Saturation Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::contrastAdjustment(float *d_hsv, int width, int height, int channels, float contrastFactor, float midpoint) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::contrastAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, contrastFactor, midpoint);

    checkKernelError("Contrast Adjustment");

    cudaDeviceSynchronize();
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