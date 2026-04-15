#include "../../include/Kernels/Flare.cuh"

__global__ void generateFlareHSV(float* data, int width, int height, int channels,
                                 float flareX, float flareY,
                                 float baseHue, float baseSaturation, float falloff) {

    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int index = (dy * width + dx) * channels;

    if (dx >= width || dy >= height) return;

    // Öklid Mesafesi
    float distanceX = (float)dx - flareX;
    float distanceY = (float)dy - flareY;
    float d = sqrtf((distanceX * distanceX) + (distanceY * distanceY));

    // Parlaklık Hesaplama (Üstel Hesaplama)
    float v = expf(-(d * falloff));

    // Performans opt. ~early exit
    if (v < 0.001f) return;

    data[index]     = baseHue;
    data[index + 1] = baseSaturation;
    data[index + 2] = v;               

    // Texture belleği (cudaArray) genellikle 4 kanallı (float4) olmayı sever.
    // Bellek hizalaması bozulmasın diye 4. kanalı (Alpha) 1.0f ile dolduruyoruz.
    if (channels == 4) {
        data[index + 3] = 1.0f;
    }
}