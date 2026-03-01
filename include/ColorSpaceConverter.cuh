//
// Created by Can on 23.02.2026.
//

#ifndef CUDAVISIONENGINE_COLORSPACECONVERTER_CUH
#define CUDAVISIONENGINE_COLORSPACECONVERTER_CUH

__global__ void rgbToHsv(const float* A, float* Result, int width, int height, int chanel);
__global__ void hsvToRgb(const float* A, float* Result, int width, int height, int chanel);


#endif //CUDAVISIONENGINE_COLORSPACECONVERTER_CUH