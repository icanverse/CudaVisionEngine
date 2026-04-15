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


/// Additive (Toplamalı) Blend ~Sadece ışık, parlama
/// Alpha (Normal) Blend ~Standart görüntü katmanlaması
/// İki işlemi de destekleyen kernel    :::
__global__ void applyTextureBlend(float* data, int width, int height, int channels,
                                        cudaTextureObject_t overlayTex, int texWidth, int texHeight,
                                        float targetX, float targetY, float opacity, bool isAdditive) {

    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= width || dy >= height) return;

    // Evrensel Koordinat Hizalama (Merkezi targetX, targetY'ye oturt)
    float u = (dx - targetX + (texWidth / 2.0f)) / (float)texWidth;
    float v = (dy - targetY + (texHeight / 2.0f)) / (float)texHeight;

    // Texture'dan renk ve saydamlığı oku
    float4 overlayPixel = tex2D<float4>(overlayTex, u, v);

    // Texture'ın kendi Alpha kanalı ile kullanıcının dışarıdan verdiği Opacity'yi harmanla
    float alpha = overlayPixel.w * opacity;

    // Sadece ekrana basılacak bir renk/saydamlık varsa işlem yap (Performans optimizasyonu)
    if (alpha > 0.001f || (isAdditive && (overlayPixel.x > 0 || overlayPixel.y > 0 || overlayPixel.z > 0))) {

        int idx = (dy * width + dx) * channels;

        float rOriginal = data[idx];
        float gOriginal = data[idx + 1];
        float bOriginal = data[idx + 2];

        if (isAdditive) {
            // IŞIK MODU
            data[idx]     = fminf(rOriginal + (overlayPixel.x * opacity), 1.0f);
            data[idx + 1] = fminf(gOriginal + (overlayPixel.y * opacity), 1.0f);
            data[idx + 2] = fminf(bOriginal + (overlayPixel.z * opacity), 1.0f);
        } else {
            // NORMAL MOD
            data[idx]     = (rOriginal * (1.0f - alpha)) + (overlayPixel.x * alpha);
            data[idx + 1] = (gOriginal * (1.0f - alpha)) + (overlayPixel.y * alpha);
            data[idx + 2] = (bOriginal * (1.0f - alpha)) + (overlayPixel.z * alpha);
        }
    }
}