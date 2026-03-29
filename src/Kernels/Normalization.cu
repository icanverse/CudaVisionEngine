#include "../../include/Kernels/Normalization.cuh"

__global__ void k_normalizeImage(unsigned char* input, float* output, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalElements) {

        unsigned char val = input[idx];

        output[idx] = (float)val / 255.0f;
    }
}

__global__ void k_denormalizeImage(float* input, unsigned char* output, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        float val = input[idx] * 255.0f;

        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;


        output[idx] = (unsigned char)val;
    }
}

__global__ void retinexNormalize(float* input, const float* global_min, const float* global_max, int total_pixels, int channels) {
    unsigned int dx = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int v_index = dx * channels + 2;

    if (dx < total_pixels) {
        float v = input[v_index];
        float min = *global_min;
        float max = *global_max;
        float range = max - min;

        if (range < 0.0000000000001f) {
            range = 1.0f;
        }

        float new_v = (v - min)/range;

        new_v = fmaxf(0.0f,fminf(1.0f,new_v));

        input[v_index] = new_v;
    }
}
