#ifndef CUDAVISIONENGINE_COLORSPACECONVERTER_CUH
#define CUDAVISIONENGINE_COLORSPACECONVERTER_CUH

__global__ void rgbToHsv(const float* A, float* Result, int width, int height, int chanel);
__global__ void hsvToRgb(const float* A, float* Result, int width, int height, int chanel);

__global__ void rgbToYuv(const float* A, float* Result, int width, int height, int channel);
__global__ void yuvToRgb(const float* A, float* Result, int width, int height, int channels);

__global__ void kernelNV12toRGB(const unsigned char* pNV12, unsigned char* pRGB, int width, int height, int pitch);

#endif //CUDAVISIONENGINE_COLORSPACECONVERTER_CUH