//
// Created by Can on 13.02.2026.
//

#include "../include/Thresholding.cuh"


__global__ void thresholding(const float* source, float* result, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = y * width + x;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = source[index];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    if (x < width && y < height) {
        if (tile[threadIdx.y][threadIdx.x] > 128) {
            result[index] = 255;
        } else {
            result[index] = 0;
        }
    }
}