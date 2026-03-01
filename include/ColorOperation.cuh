//
// Created by Can on 1.03.2026.
//

#ifndef CUDAVISIONENGINE_COLOROPERATION_CUH
#define CUDAVISIONENGINE_COLOROPERATION_CUH


__global__ void isolateColor(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance);
__global__ void colorReplacement(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance, float replacementHue);

#endif //CUDAVISIONENGINE_COLOROPERATION_CUH