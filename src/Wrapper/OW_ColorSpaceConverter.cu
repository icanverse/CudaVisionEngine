#include "OperationWrapper.cuh"
#include "ColorSpaceConverter.cuh"


void OperationWrapper::rgbToHsv(const float* d_input, float* d_output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::rgbToHsv<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    checkKernelError("Convert RGB to HSV");

    cudaDeviceSynchronize();

}

void OperationWrapper::hsvToRgb(const float* d_input, float* d_output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::hsvToRgb<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    checkKernelError("Convert HSV to RGB");

    cudaDeviceSynchronize();
}

void OperationWrapper::rgbToYuv(const float *A, float *Result, int width, int height, int channel) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::rgbToYuv<<<gridSize, blockSize>>>(A, Result, width, height, channel);

    checkKernelError("Convert RGB to YUV");

    cudaDeviceSynchronize();
}

void OperationWrapper::yuvToRgb(const float *A, float *Result, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::yuvToRgb<<<gridSize, blockSize>>>(A, Result, width, height, channels);

    checkKernelError("Convert YUV to RGB");

    cudaDeviceSynchronize();
}

void OperationWrapper::kernelNV12toRGB(const unsigned char *pNV12, unsigned char *pRGB, int width, int height, int pitch) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::kernelNV12toRGB<<<gridSize, blockSize>>>(pNV12, pRGB, width, height, pitch);

    checkKernelError("Convert YUV to RGB (NV12)");

    cudaDeviceSynchronize();
}