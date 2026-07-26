#include "../../include/Kernels/DepthMapKernel.cuh"

__global__ void seedHeatSourcesKernel(
    cudaSurfaceObject_t depthMap,
    int width,
    int height,
    float2* points,
    float* depthValues,
    int* pointLineIndices,
    int* isSegmentEnd,
    int totalPoints
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= totalPoints) return;
    if (isSegmentEnd[index] == 1) return;

    int lineId = pointLineIndices[index];
    float depth = depthValues[lineId];

    // UV koordinatlarını gerçek piksel koordinatlarına (X, Y) çevirir
    float x1 = points[index].x * width;
    float y1 = points[index].y * height;
    float x2 = points[index + 1].x * width;
    float y2 = points[index + 1].y * height;

    // GPU üzerinde DDA (Digital Differential Analyzer) :: Çizgi Çizme Algoritması
    float dx = x2 - x1;
    float dy = y2 - y1;
    float steps = fmaxf(fabsf(dx), fabsf(dy)); // Hangi eksen daha uzunsa o kadar adım at

    float xInc = dx / steps;
    float yInc = dy / steps;

    float x = x1;
    float y = y1;

    // Noktadan noktaya piksel piksel ilerle ve değeri yüzeye (Surface) yaz
    for (int i = 0; i <= steps; i++) {
        int px = roundf(x);
        int py = roundf(y);

        // Ekran sınırları içinde miyiz?
        if (px >= 0 && px < width && py >= 0 && py < height) {
            // CUDA Surface API: Yazılacak değer, yüzey, X ofseti (byte cinsinden!), Y ofseti
            surf2Dwrite(depth, depthMap, px * sizeof(float), py);
        }

        x += xInc;
        y += yInc;
    }
}


// Isı Kaynaklarını Basma (Rasterizastion)
void launchIsoDepthKernel(
    cudaSurfaceObject_t outputDepthMap,
    int width,
    int height,
    const IsoLineData& lineData
) {
    if (lineData.totalPoints <= 1) return; // Çizilecek bir şey yok

    int threadsPerBlock = 256;

    int blocksPerGrid = (lineData.totalPoints + threadsPerBlock - 1) / threadsPerBlock;

    seedHeatSourcesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        outputDepthMap,
        width,
        height,
        lineData.points,
        lineData.depthValues,
        lineData.pointLineIndices,
        lineData.isSegmentEnd,
        lineData.totalPoints
    );

    cudaDeviceSynchronize();
}

__global__ void applyNormalMapFromDepth(
    float* d_data,          // Çıktı: Hedef Normal Map (RGB veya RGBA görsel tuvali)
    const float* depthMap,  // Girdi: Laplace veya IDW ile doldurduğumuz tek kanallı Z haritası
    int width,
    int height,
    int channels,
    float intensity
) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;

    int margin = 5;
    if (dx < margin || dy < margin || dx >= width - margin || dy >= height - margin) return;

    int index1D = dy * width + dx;
    int index3D = index1D * channels;

    // Merkezi Farklar (Central Differences) ile komşu piksellerden türev alma
    float leftZ   = depthMap[dy * width + (dx - 1)];
    float rightZ  = depthMap[dy * width + (dx + 1)];
    float topZ    = depthMap[(dy - 1) * width + dx];
    float bottomZ = depthMap[(dy + 1) * width + dx];

    float u = ((rightZ - leftZ) / 2.0f) * intensity;
    float v = ((bottomZ - topZ) / 2.0f) * intensity;

    float normal_x = max(min(1.0f, u), -1.0f);
    float normal_y = max(min(1.0f, v), -1.0f);

    float normal_zz = 1.0f - normal_x * normal_x - normal_y * normal_y;
    float normal_z;

    if (normal_zz <= 0.0f) {
        normal_z = 0.0f;
    } else {
        normal_z = sqrtf(normal_zz);
    }

    normal_x = ( normal_x + 1.0f ) / 2.0f;
    normal_y = ( normal_y + 1.0f ) / 2.0f;
    normal_z = ( normal_z + 1.0f ) / 2.0f;

    float r = normal_x;
    float g = normal_y;
    float b = normal_z;

    d_data[index3D]     = r;
    d_data[index3D + 1] = g;
    d_data[index3D + 2] = b;

    if (channels == 4) {
        d_data[index3D + 3] = 1.0f;
    }
}