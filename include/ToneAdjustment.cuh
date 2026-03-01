//
// Created by Can on 2.03.2026.
//

#ifndef CUDAVISIONENGINE_TONEADJUSTMENT_CUH
#define CUDAVISIONENGINE_TONEADJUSTMENT_CUH


__global__ void saturationAdjustment(float* d_hsv, int width, int height, int channels, float value);
__global__ void brightnessAdjustment(float* d_hsv, int width, int height, int channels, float value);
__global__ void contrastAdjustment(float* d_hsv, int width, int height, int channels, float contrastFactor, float midpoint);


#endif //CUDAVISIONENGINE_TONEADJUSTMENT_CUH