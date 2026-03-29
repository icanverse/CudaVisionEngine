#include "Kernels/MaskOperation.cuh"

__global__ void blendVChannel(const float* base, const float* detail, float* output, int width, int height, int channels, float strength) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int idx = (y * width + x) * channels;

        output[idx]     = base[idx];
        output[idx + 1] = base[idx + 1];

        float v_base   = base[idx + 2];
        float v_detail = detail[idx + 2];

        output[idx + 2] = fminf(1.0f, v_base + v_detail * strength * (1.0f - v_base));
    }
}