//
// Created by Can on 2.03.2026.
//

#ifndef CUDAVISIONENGINE_TONEADJUSTMENT_CUH
#define CUDAVISIONENGINE_TONEADJUSTMENT_CUH


__global__ void saturationAdjustment(float* d_hsv, int width, int height, int channels, float value);
__global__ void brightnessAdjustment(float* d_hsv, int width, int height, int channels, float value);
__global__ void contrastAdjustment(float* d_hsv, int width, int height, int channels, float contrastFactor, float midpoint);
__global__ void shadowsHighlightsAdjustment(float* d_hsv, int width, int height, int channels, float shadowAmount, float highlightAmount);
__global__ void temperatureAdjustment(float* d_rgb, int width, int height, int channels, float temperature);
__global__ void gammaCorrectionAdjustment(float* d_hsv, int width, int height, int channels, float gamma);




#endif //CUDAVISIONENGINE_TONEADJUSTMENT_CUH