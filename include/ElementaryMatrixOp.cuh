//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_ELEMENTARYMATRIX_CUH
#define CUDAVISIONENGINE_ELEMENTARYMATRIX_CUH

#define TILE_SIZE 32

__global__ void k_normalizeImage(unsigned char* input, float* output, int totalElements);
__global__ void k_denormalizeImage(float* input, unsigned char* output, int totalElements);
__global__ void matrix_add(const float* source_matrix1, const float* source_matrix2, float* dest_matrix, int size);
__global__ void matrix_add_with_sharedmem(const float* source_matrix1, const float* source_matrix2, float* dest_matrix, int size);
__global__ void matrix_mul(const float* source_matrix1, const float* source_matrix2, float* dest_matrix, int size);


#endif //CUDAVISIONENGINE_ELEMENTARYMATRIX_CUH