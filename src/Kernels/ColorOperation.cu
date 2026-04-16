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

__global__ void chromaticAberration(const float* input, float* output, int width, int height, int channels, float intensity) {
    int dx = threadIdx.x + blockDim.x * blockIdx.x;
    int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if (dx >= width || dy >= height) return;

    float centerX = width / 2.0f;
    float centerY = height / 2.0f;

    // Normalize Edilmiş Mesafe (-1.0 ile 1.0 arasında)
    float normX = ((float)dx - centerX) / centerX;
    float normY = ((float)dy - centerY) / centerY;

    // Piksel Kayma Miktarı (Intensity ile çarpılmış yön)
    int shiftX = (int)(normX * intensity);
    int shiftY = (int)(normY * intensity);

    // Kırmızı, Yeşil ve Mavi için yeni okuma koordinatları
    // Sınır kontrolü (Clamp): 0'ın altına veya width/height'in üstüne çıkma
    int r_dx = max(0, min(width - 1, dx - shiftX)); // Kırmızı içe doğru kaysın
    int r_dy = max(0, min(height - 1, dy - shiftY));

    int g_dx = dx; // Yeşil sabit kalsın (Merkez her zaman nettir)
    int g_dy = dy;

    int b_dx = max(0, min(width - 1, dx + shiftX)); // Mavi dışa doğru kaysın
    int b_dy = max(0, min(height - 1, dy + shiftY));

    // İndeksleri hesapla ve İlgili kanalları oradan oku
    int r_index = (r_dy * width + r_dx) * channels;
    int g_index = (g_dy * width + g_dx) * channels;
    int b_index = (b_dy * width + b_dx) * channels;

    // Orijinal (input) veriden farklı yerlerdeki R, G, B kanallarını topla
    float r = input[r_index];       // Kırmızıyı kaymış yerden oku
    float g = input[g_index + 1];   // Yeşili merkezden oku (+1)
    float b = input[b_index + 2];   // Maviyi ters kaymış yerden oku (+2)

    // Mevcut piksele yaz!
    int out_index = (dy * width + dx) * channels;
    output[out_index]     = r;
    output[out_index + 1] = g;
    output[out_index + 2] = b;

    if (channels == 4) {
        output[out_index + 3] = input[out_index + 3];
    }

}