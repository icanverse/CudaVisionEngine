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

__global__ void shadowsHighlightsAdjustment(float* d_hsv, int width, int height, int channels, float shadowAmount, float highlightAmount) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float value = d_hsv[index + 2];
        float shadow_weight = 1.0f - fminf(1.0f, value * 2.0f);
        float highlight_weight = fmaxf(0.0f, (value - 0.5f) * 2.0f);

        float new_value = value + (shadowAmount * shadow_weight) + (highlightAmount * highlight_weight);
        new_value = fminf(1.0f, fmaxf(0.0f, new_value));
        d_hsv[index + 2] = new_value;

    }
}

__global__ void temperatureAdjustment(float* d_rgb, int width, int height, int channels, float temperature) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float red = d_rgb[index];
        float blue = d_rgb[index + 2];

        float new_red = fminf(1.0f, fmaxf(0.0f, red + temperature));
        float new_blue = fminf(1.0f, fmaxf(0.0f, blue - temperature));

        d_rgb[index] = new_red;
        d_rgb[index + 2] = new_blue;
    }
}

__global__ void gammaCorrectionAdjustment(float* d_hsv, int width, int height, int channels, float gamma){
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float value = d_hsv[index + 2];
        float new_value = powf(value,gamma);

        float clamp_value = fminf(1.0f, fmaxf(0.0f, new_value));

        d_hsv[index + 2] = clamp_value;

    }

}

