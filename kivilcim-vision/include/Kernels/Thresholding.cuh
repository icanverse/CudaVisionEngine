//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_THRESHOLDING_CUH
#define CUDAVISIONENGINE_THRESHOLDING_CUH


#define TILE_SIZE 32
__global__ void thresholding(const float* source, float* result, int width, int height);



#endif //CUDAVISIONENGINE_THRESHOLDING_CUH