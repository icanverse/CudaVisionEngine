#include "OperationWrapper.cuh"
#include "../../include/Kernels/ToneAdjustment.cuh"

void OperationWrapper::brightnessAdjustment(float *d_hsv, int width, int height, int channels, float value) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::brightnessAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, value);

    checkKernelError("Brightness Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::saturationAdjustment(float *d_hsv, int width, int height, int channels, float value) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::saturationAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, value);

    checkKernelError("Saturation Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::contrastAdjustment(float *d_hsv, int width, int height, int channels, float contrastFactor, float midpoint) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::contrastAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, contrastFactor, midpoint);

    checkKernelError("Contrast Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::shadowsHighlightsAdjustment(float *d_hsv, int width, int height, int channels, float shadowAmount, float highlightAmount) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::shadowsHighlightsAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, shadowAmount, highlightAmount);

    checkKernelError("Shadows - Highlights Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::temperatureAdjustment(float *d_rgb, int width, int height, int channels, float temperature) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::temperatureAdjustment<<<gridSize, blockSize>>>(d_rgb, width, height, channels, temperature);

    checkKernelError("Temperature Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::gammaCorrectionAdjustment(float *d_hsv, int width, int height, int channels, float gamma) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::gammaCorrectionAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, gamma);

    checkKernelError("Gamma Correction Adjustment");

    cudaDeviceSynchronize();
}
