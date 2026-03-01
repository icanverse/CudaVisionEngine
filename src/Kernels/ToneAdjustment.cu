//
// Created by Can on 2.03.2026.
//

#include "ToneAdjustment.cuh"

__global__ void saturationAdjustment(float* d_hsv, int width, int height, int channels, float value) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float saturation = d_hsv[index + 1];

        float new_saturation = fminf(1.0f, fmaxf(0.0f, saturation * value));

        d_hsv[index + 1] = new_saturation;
    }
}

__global__ void brightnessAdjustment(float* d_hsv, int width, int height, int channels, float value) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float brightness = d_hsv[index + 2];

        float new_brightness = fminf(1.0f, fmaxf(0.0f, brightness + value));

        d_hsv[index + 2] = new_brightness;
    }
}


__global__ void contrastAdjustment(float* d_hsv, int width, int height, int channels, float contrastFactor, float midpoint) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float v = d_hsv[index + 2];

        // Tahterevalli Matematiği
        float new_v = (v - midpoint) * contrastFactor + midpoint;

        // Kontrast değerleri çok çabuk 0'ın altına veya 1'in üstüne taşırır.
        // Bu yüzden clamp (sınırlandırma) BURADA HAYATİ DERECEDE ÖNEMLİDİR!
        new_v = fminf(1.0f, fmaxf(0.0f, new_v));

        d_hsv[index + 2] = new_v;
    }
}