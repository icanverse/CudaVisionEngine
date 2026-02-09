//
// Created by Can on 9.02.2026.
//

#include "../include/ElementaryMatrixOp.cuh"

__global__ void k_normalizeImage(unsigned char* input, float* output, int totalElements) {
    // Her thread kendi kimliğini (ID) hesaplar
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Eğer bu thread'in ID'si resim boyutundan küçükse işini yapar
    if (idx < totalElements) {
        // 1. Veriyi oku
        unsigned char val = input[idx];

        // 2. Çevir (0-255 -> 0.0-1.0) ve yaz
        output[idx] = (float)val / 255.0f;
    }
}

__global__ void k_denormalizeImage(float* input, unsigned char* output, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        // 1. Geri çarp (0.0-1.0 * 255.0)
        float val = input[idx] * 255.0f;

        // 2. Taşmaları engelle (Clamp) - ÇOK ÖNEMLİ!
        // Matematiksel işlemler sonucu değer -0.1 veya 256.5 çıkabilir.
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;

        // 3. Tamsayıya çevir ve yaz
        output[idx] = (unsigned char)val;
    }
}

__global__ void matrix_add(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = x + y * size;
    if (x < size && y < size ) {
        dest_matrix[index] = source_matrix1[index] + source_matrix2[index];
    }
}

__global__ void matrix_add_with_sharedmem(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size, bool op_control) {
    __shared__ float s_mat1[TILE_SIZE + 1][TILE_SIZE + 1];
    __shared__ float s_mat2[TILE_SIZE + 1][TILE_SIZE + 1];

    unsigned int dx = threadIdx.x;
    unsigned int dy = threadIdx.y;

    int x = blockIdx.x * blockDim.x + dx;
    int y = blockIdx.y * blockDim.y + dy;
    int index = x + y * size;

    if (x < size && y < size) {
        s_mat1[dy][dx] = source_matrix1[index];
        s_mat2[dy][dx] = source_matrix2[index];
    } else {
        s_mat1[dy][dx] = 0.0f;
        s_mat2[dy][dx] = 0.0f;
    }

    __syncthreads();

    if (x < size && y < size) {
        if (op_control == true) {
            dest_matrix[index] = s_mat1[dy][dx] + s_mat2[dy][dx];
        } else {
            dest_matrix[index] = s_mat1[dy][dx] - s_mat2[dy][dx];
        }
    }
}

__global__ void matrix_mul(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size ) {
        float sum = 0;

        for (int i = 0; i < size; i++) {
            // 1'in satırı ile 2'nin sütununu çarpıyoruz
            sum += source_matrix1[y * size + i] * source_matrix2[i * size + x];
        }
        dest_matrix[y * size + x] = sum;
    }
}

__global__ void findSubMatrix(const float* source_matrix1, float* sub_matrix, unsigned int p, unsigned int q ,int size) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size ) {
        if (x != p && y != q) {

            int targetX = (x > p) ? x - 1 : x;
            int targetY = (y > q) ? y - 1 : y;

            int targetIndex = targetY * (size - 1) + targetX;

            sub_matrix[targetIndex] = source_matrix1[y * size + x];
        }
    }
}