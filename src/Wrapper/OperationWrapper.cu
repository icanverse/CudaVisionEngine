#include "OperationWrapper.cuh"
#include "../../include/Kernels/ElementaryMatrixOp.cuh"
#include "../../include/Kernels/Smoothing.cuh"

#include <cstdio>

#include "Kernels/Flare.cuh"
#include "Kernels/Reduction.cuh"
#include "Kernels/LogTransformation.cuh"
#include "Kernels/LUT_3D.cuh"
#include "Kernels/MaskOperation.cuh"
#include "Kernels/Normalization.cuh"

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

void OperationWrapper::retinexNormalize(float *input, const float *global_min, const float *global_max, int width, int height, int channels) {
    int total_pixels = width * height;

    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::retinexNormalize<<<gridSize, blockSize>>>(input, global_min, global_max,total_pixels, channels);

    checkKernelError("Retinex Normalize");
}

void OperationWrapper::k_MinMaxReduction(const float *input, float *global_min, float *global_max, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    int threads = 256;
    int total_pixels = width * height;
    int blocks = (total_pixels + threads - 1) / threads;
    size_t sharedMemSize = 2 * threads * sizeof(float);

    ::k_MinMaxReduction<<<blocks, threads, sharedMemSize>>>(input, global_min, global_max, width, height, channels);

    checkKernelError("MinMaxReduction");
}

void OperationWrapper::smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize) {
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    int radius = kernelSize / 2;
    size_t sharedMemSize = (blockSize.x + 2 * radius) * (blockSize.y + 2 * radius) * sizeof(float);

    ::smoothing2D<<<gridSize, blockSize, sharedMemSize>>>(A, Result, width, height, channels, kernelSize);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in smoothing2D: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}



/// Dönüşümler

void OperationWrapper::logTransformation(float *input, float* output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::logTransformationVChannel<<<gridSize, blockSize>>>(input, output, width, height, channels);

    checkKernelError("Log Dönüşüm");

    cudaDeviceSynchronize();
}
void OperationWrapper::add(const float* d_A, const float* d_B, float* d_C, int size, bool useSharedMem) {
    // 2D Grid Hesabı
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);


    ::add<<<grid, block>>>(d_A, d_B, d_C, size);


    checkKernelError("Matrix Add");
}

void OperationWrapper::subtract(const float* d_A, const float* d_B, float* d_C, int size) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);



    checkKernelError("Matrix Subtract");
}

void OperationWrapper::multiply(const float* d_A, const float* d_B, float* d_C, int size) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    mul<<<grid, block>>>(d_A, d_B, d_C, size);

    checkKernelError("Matrix Multiply");
}



void OperationWrapper::getSubMatrix(const float* d_in, float* d_out, int removeCol, int removeRow, int currentSize) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((currentSize + block.x - 1) / block.x, (currentSize + block.y - 1) / block.y);

    //findSubMatrix<<<grid, block>>>(d_in, d_out, removeCol, removeRow, currentSize);

    checkKernelError("Get SubMatrix");
}

void OperationWrapper::applyRetinexNormalize(float* d_data, const float* d_global_min, const float* d_global_max, int total_pixels, int channels) {

    int threads1D = 256;
    int blocks1D = (total_pixels + threads1D - 1) / threads1D;

    ::retinexNormalize<<<blocks1D, threads1D>>>(d_data, d_global_min, d_global_max, total_pixels, channels);

    cudaDeviceSynchronize();
}

// =========================================================
// PROCEDURAL EFFECTS & TEXTURE MAPPING
// =========================================================

void OperationWrapper::generateFlareHSV(float* data, int width, int height, int channels,
                                        float flareX, float flareY,
                                        float baseHue, float baseSaturation, float falloff) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Senin efsanevi tek satırlık grid hesabın!

    ::generateFlareHSV<<<gridSize, blockSize>>>(data, width, height, channels,
                                                flareX, flareY, baseHue, baseSaturation, falloff);

    checkKernelError("Generate Flare HSV");

    cudaDeviceSynchronize();
}

void OperationWrapper::applyTextureBlendKernel(float* data, int width, int height, int channels,
                                               cudaTextureObject_t overlayTex, int texWidth, int texHeight,
                                               float targetX, float targetY, float opacity, bool isAdditive) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::applyTextureBlend<<<gridSize, blockSize>>>(data, width, height, channels,
                                                       overlayTex, texWidth, texHeight,
                                                       targetX, targetY, opacity, isAdditive);

    checkKernelError("Apply Evrensel Texture Blend");

    cudaDeviceSynchronize();
}

void OperationWrapper::apply3DLUT(float* data, int width, int height, int channels, cudaTextureObject_t lutTexture) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::apply3DLUT<<<gridSize, blockSize>>>(data, width, height, channels, lutTexture);

    checkKernelError("Apply 3D LUT");
    cudaDeviceSynchronize();
}
