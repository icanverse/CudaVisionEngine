//
// Created by Can on 10.02.2026.
//

#ifndef CUDAVISIONENGINE_EDGEDETECTION_CUH
#define CUDAVISIONENGINE_EDGEDETECTION_CUH

#define TILE_SIZE 16
#define RADIUS 1

__global__ void sobel_edge_det(const float* A, float* Result, int width, int height);

/// Sobel Filtresi Matrisleri
__constant__ int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
__constant__ int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};



#endif //CUDAVISIONENGINE_EDGEDETECTION_CUH