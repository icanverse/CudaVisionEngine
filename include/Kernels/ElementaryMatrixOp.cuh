//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_ELEMENTARYMATRIX_CUH
#define CUDAVISIONENGINE_ELEMENTARYMATRIX_CUH

#define TILE_SIZE 32

// __global__ void matrix_add(const float* source_matrix1, const float* source_matrix2, float* dest_matrix, int size);

__global__ void add(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size);
__global__ void sub(const float*source_matrix1, const float*source_matrix2, float*dest_matrix, int size);
__global__ void subVCh(const float* original, const float* blurred, float* output, int width, int height, int channels);
__global__ void mul(const float* source_matrix1, const float* source_matrix2, float* dest_matrix, int size);
__global__ void findSubMatrix(const float* source_matrix1, float* sub_matrix, unsigned int p, unsigned int q ,int size);


#endif //CUDAVISIONENGINE_ELEMENTARYMATRIX_CUH