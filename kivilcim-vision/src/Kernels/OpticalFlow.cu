#include "../include/Kernels/OpticalFlow.cuh"

/// Paylaşımlı bellek kullanan Lucas Kanade Optical Flow'u
__global__ void opticalFlowLucasKanade(const float* currentFrame, const float* previousFrame,
                                       int width, int height, int channels, float* flowU, float* flowV){

    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float s_curr[SHARED_WIDTH][SHARED_WIDTH];
    __shared__ float s_prev[SHARED_WIDTH][SHARED_WIDTH];

    // 256 Thread, 22x22 = 484 piksellik alanı paylaşacak
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Thread'in blok içindeki 1D sırası (0 - 255)
    int totalThreads = blockDim.x * blockDim.y;       // 256
    int totalShared = SHARED_WIDTH * SHARED_WIDTH;    // 484

    // Veriyi L1'e getir
    // Veriyi L1'e (Shared Memory) getir
    for (int i = tid; i < totalShared; i += totalThreads) {
        int sy = i / SHARED_WIDTH;  // shared mem'de y koordinatı
        int sx = i % SHARED_WIDTH;  // shared mem'de x koordinatı

        // Global koordinatlar
        int gx = (int)(blockIdx.x * TILE_SIZE) + sx - PADDING;
        int gy = (int)(blockIdx.y * TILE_SIZE) + sy - PADDING;

        s_curr[sy][sx] = getLuma(currentFrame, width, height, channels, gx, gy);
        s_prev[sy][sx] = getLuma(previousFrame, width, height, channels, gx, gy);
    }

    // Senkronize
    __syncthreads();

    // Optik Akış
    if (dx >= width || dy >= height) return;

    int shared_x = threadIdx.x + PADDING;
    int shared_y = threadIdx.y + PADDING;

    float sumXX = 0.0f, sumYY = 0.0f, sumXY = 0.0f, sumXT = 0.0f, sumYT = 0.0f;

    for (int wy = -RADIUS; wy <= RADIUS; wy++) {
        for (int wx = -RADIUS; wx <= RADIUS; wx++) {

            int cx = shared_x + wx;
            int cy = shared_y + wy;

            /// Gradyanlar

            // X yönü gradyanı
            float Ix = (s_curr[cy][cx + 1] - s_curr[cy][cx - 1]) / 2.0f;

            // Y yönü gradyanı
            float Iy = (s_curr[cy + 1][cx] - s_curr[cy - 1][cx]) / 2.0f;

            // Zaman gradyanı (Şimdiki - Önceki)
            float It = s_curr[cy][cx] - s_prev[cy][cx];

            sumXX += (Ix * Ix);
            sumYY += (Iy * Iy);
            sumXY += (Ix * Iy);
            sumXT += (Ix * It);
            sumYT += (Iy * It);
        }
    }

    // Matris Çözümü
    float determinant = (sumXX * sumYY) - (sumXY * sumXY);

    float u = 0.0f;
    float v = 0.0f;

    if (determinant > 0.0001f) {
        u = -(sumYY * sumXT - sumXY * sumYT) / determinant;
        v = -(sumXX * sumYT - sumXY * sumXT) / determinant;
    }

    int outIndex = dy * width + dx;
    flowU[outIndex] = u;
    flowV[outIndex] = v;
}



__device__ float getLuma(const float* d_rgb, int width, int height, int channels, int x, int y ) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));

    int index = (x + y * width) * channels;

    float r = d_rgb[index];
    float g = d_rgb[index + 1];
    float b = d_rgb[index + 2];

    return (0.299f * r) + (0.587f * g) + (0.114f * b);
}