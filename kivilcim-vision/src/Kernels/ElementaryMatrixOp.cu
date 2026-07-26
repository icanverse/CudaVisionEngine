#include "../../include/Kernels/ElementaryMatrixOp.cuh"

// __global__ void matrix_add(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size) {
//     unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//     unsigned int index = x + y * size;
//     if (x < size && y < size ) {
//         dest_matrix[index] = source_matrix1[index] + source_matrix2[index];
//     }
// }

__global__ void add(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size) {
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

    dest_matrix[index] = s_mat1[dy][dx] + s_mat2[dy][dx];

    __syncthreads();

}

__global__ void sub(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size) {
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

    dest_matrix[index] = s_mat1[dy][dx] - s_mat2[dy][dx];

    __syncthreads();
}

__global__ void mul(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size ) {
        float sum = 0;

        for (int i = 0; i < size; i++) {
            sum += source_matrix1[y * size + i] * source_matrix2[i * size + x];
        }
        dest_matrix[y * size + x] = sum;
    }
}

__global__ void subVCh(const float* original, const float* blurred, float* output, int width, int height, int channels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        unsigned int index = (y * width + x) * channels;

        output[index]     = original[index];
        output[index + 1] = original[index + 1];

        float v_orig = original[index + 2];
        float v_blur = blurred[index + 2];

        output[index + 2] = fmaxf(0.0f, v_orig - v_blur);
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