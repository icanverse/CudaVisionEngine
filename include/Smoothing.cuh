//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_SMOOTHING_CUH
#define CUDAVISIONENGINE_SMOOTHING_CUH


__global__ void smoothing(const float* A,float* Result, int size);
__global__ void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize);



#endif //CUDAVISIONENGINE_SMOOTHING_CUH