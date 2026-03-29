//
// Created by Can on 1.03.2026.
//

#include "../../include/Kernels/ColorOperation.cuh"

__global__ void isolateColor(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance) {
    unsigned int dx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float h = d_hsv[index];

        float difference = fabsf(targetHue - h);
        float real_difference;

        if (difference > 180.0f) {
            real_difference = 360.0f - difference;
        } else {
            real_difference = difference;
        }

        if (tolerance < real_difference) {
            d_hsv[index + 1] = 0.0f;
        }
    }
}

__global__ void colorReplacement(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance, float replacementHue) {
    unsigned int dx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float h = d_hsv[index];

        float difference = fabsf(targetHue - h);
        float real_difference;

        if (difference > 180.0f) {
            real_difference = 360.0f - difference;
        } else {
            real_difference = difference;
        }

        if (real_difference <= tolerance) {

            float shift = replacementHue - targetHue;
            float new_h = h + shift;

            d_hsv[index] = fmodf(new_h + 360.0f, 360.0f);
        }
    }
}