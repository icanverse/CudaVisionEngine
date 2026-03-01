//
// Created by Can on 1.03.2026.
//

#include "../../include/ColorOperation.cuh"

__global__ void isolateColor(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance) {
    unsigned int dx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx < width && dy < height) {
        // DÜZELTME 1: channels çarpanı eklendi (Parantezlere dikkat!)
        unsigned int index = (dx + dy * width) * channels;

        float h = d_hsv[index];
        // S değerini sadece yazacağımız için okumaya gerek yok, performansı artırır.

        float difference = fabsf(targetHue - h);
        float real_difference;

        // DÜZELTME 2: İçerideki "float" kelimeleri kaldırıldı
        if (difference > 180.0f) {
            real_difference = 360.0f - difference;
        } else {
            real_difference = difference;
        }

        // Eğer fark toleranstan büyükse, Doygunluğu (S) sıfırla
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

        // DÜZELTME 1: Fark toleranstan KÜÇÜK VEYA EŞİTSE (Yani hedef rengi bulduysak)
        if (real_difference <= tolerance) {

            // DÜZELTME 2: Kaydırma miktarı (Shift) iki ana renk arasındaki sabit farktır
            float shift = replacementHue - targetHue;
            float new_h = h + shift;

            // 360 derecelik renk çemberinin içinde tut
            d_hsv[index] = fmodf(new_h + 360.0f, 360.0f);
        }
    }
}