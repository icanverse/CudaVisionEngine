//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_CONVOLUTION_CUH
#define CUDAVISIONENGINE_CONVOLUTION_CUH


#define TILE_SIZE 16
#define RADIUS 1

__global__ void smoothing(const float* A,float* Result, int size);
__global__ void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize);

/// Sobel Filtresi Matrisleri
__constant__ int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};


#endif //CUDAVISIONENGINE_CONVOLUTION_CUH