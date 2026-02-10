#include "../include/EdgeDetection.cuh"

/// Sobel Kenar Filtresi Yöntemi --- Paylaşımlı Bellek Kullanıyor
__global__ void sobel_edge_det(const float* A, float* Result, int width, int height) {
    // Shared Mem
    __shared__ float s_data[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];

    //
    unsigned int dx = threadIdx.x;
    unsigned int dy = threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + dx;
    unsigned int row = blockIdx.y * blockDim.y + dy;

    int s_col = dx + RADIUS;
    int s_row = dy + RADIUS;

    if (col < width && row < height) {
        s_data[s_row][s_col] = A[row * width + col];
    } else {
        s_data[s_row][s_col] = 0.0f; // Sınır dışı sıfır olsun
    }

    // Sol Halo
    if (dx < RADIUS) {
        if (col >= RADIUS) // Resmin en solundan taşmayalım
            s_data[s_row][s_col - RADIUS] = A[row * width + (col - RADIUS)];
        else
            s_data[s_row][s_col - RADIUS] = 0.0f;
    }

    // Sağ Halo
    if (dx >= blockDim.x - RADIUS) {
        if (col + RADIUS < width) // Resmin en sağından taşmayalım
            s_data[s_row][s_col + RADIUS] = A[row * width + (col + RADIUS)];
        else
            s_data[s_row][s_col + RADIUS] = 0.0f;
    }

    // Üst Halo
    if (dy < RADIUS) {
        if (row >= RADIUS)
            s_data[s_row - RADIUS][s_col] = A[(row - RADIUS) * width + col];
        else
            s_data[s_row - RADIUS][s_col] = 0.0f;
    }

    // Alt Halo
    if (dy >= blockDim.y - RADIUS) {
        if (row + RADIUS < height)
            s_data[s_row + RADIUS][s_col] = A[(row + RADIUS) * width + col];
        else
            s_data[s_row + RADIUS][s_col] = 0.0f;
    }

    __syncthreads();

    if (col < width && row < height && col > 0 && row > 0 && col < width - 1 && row < height - 1) {
        float sumX = 0, sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                // DİKKAT: s_data indeksleri
                float pixel = s_data[s_row + i][s_col + j];

                // Not: Gx ve Gy global/constant memory'de tanımlı varsayıyoruz
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        Result[row * width + col] = sqrtf(sumX * sumX + sumY * sumY);
    }
}
