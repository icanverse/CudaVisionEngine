//
// Created by Can on 13.02.2026.
//

#include "../../include/Convolution.cuh"

__global__ void applyConvulationKernel(const float* input, float* output, int width, int height) {

}

__global__ void sharpen(const float* input, float* output, int width, int height, int channels) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int base_index = (dy * width + dx) * channels;

        for (int c = 0; c < channels; ++c) {

            // Alpha kanalıysa kopyala ve geç
            if (c == 3) {
                output[base_index + c] = input[base_index + c];
                continue;
            }

            float sum = 0.0f;

            // 3. Komşuları Gez (3x3 Matris)
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {

                    // DÜZELTME: Negatif olabilmeleri için 'int' kullanıyoruz
                    int nx = dx + kx;
                    int ny = dy + ky;

                    int clamped_x = max(0, min(nx, width - 1));
                    int clamped_y = max(0, min(ny, height - 1));

                    // GÖREV 3: Komşunun gerçek indeksi
                    unsigned int neighbor_index = (clamped_y * width + clamped_x) * channels + c;

                    // GÖREV 4 & 5: Ağırlığı belirle ve toplama ekle
                    float weight;
                    if (kx == 0 && ky == 0) {
                        weight = 9.0f;  // Merkez pikseli çok güçlü parlat
                    } else {
                        weight = -1.0f; // Etrafındaki pikselleri çıkar (Kontrastı aç)
                    }

                    // Komşunun değerini ağırlıkla çarp ve toplama ekle
                    sum += input[neighbor_index] * weight;
                }
            }

            // GÖREV 6: Clamp ve Geri Yazma
            // Keskinleştirme işlemi sınırları çok çabuk aşar, bu yüzden clamp şarttır.
            output[base_index + c] = fminf(1.0f, fmaxf(0.0f, sum));
        }
    }
}
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
