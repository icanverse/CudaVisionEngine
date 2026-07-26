//
// Created by Can on 22.03.2026.
//

#include "../../include/Kernels/Reduction.cuh"

__device__ __forceinline__ void atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__global__ void k_InitGlobalMinMax(float* global_min, float* global_max) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *global_min = 1e9f;   // Min için +Sonsuz
        *global_max = -1e9f;  // Max için -Sonsuz
    }
}

__global__ void k_MinMaxReduction(const float* input, float* global_min, float* global_max, int width, int height, int channels) {
    extern __shared__ float s_data[];
    float* s_min = s_data;
    float* s_max = (float*)&s_data[blockDim.x];

    unsigned int dx = threadIdx.x;
    unsigned int tx = dx + blockIdx.x * blockDim.x;
    int total_pixels = width * height;

    if (tx < total_pixels) {
        float v = input[tx * channels + 2];
        s_min[dx] = v;
        s_max[dx] = v;
    } else {
        s_min[dx] = 1e9f;
        s_max[dx] = -1e9f;
    }

    __syncthreads();


    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (dx < stride) {
            s_min[dx] = fminf(s_min[dx], s_min[dx + stride]);
            s_max[dx] = fmaxf(s_max[dx], s_max[dx + stride]);
        }
        __syncthreads();
    }

    if (dx == 0) {
        atomicMinFloat(global_min, s_min[0]);
        atomicMaxFloat(global_max, s_max[0]);
    }
}

namespace Reduction {
    void launchMinMaxReduction(const float* d_input, float* d_global_min, float* d_global_max, int width, int height, int channels) {

        int total_pixels = width * height;
        int threads = 256;
        int blocks = (total_pixels + threads - 1) / threads;
        size_t sharedMemSize = 2 * threads * sizeof(float);

        k_InitGlobalMinMax<<<1, 1>>>(d_global_min, d_global_max);

        k_MinMaxReduction<<<blocks, threads, sharedMemSize>>>(d_input, d_global_min, d_global_max, width, height, channels);

        cudaDeviceSynchronize();
    }
}