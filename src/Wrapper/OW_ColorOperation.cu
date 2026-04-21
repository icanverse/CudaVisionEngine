#include "OperationWrapper.cuh"
#include "../../include/Kernels/ColorOperation.cuh"

void OperationWrapper::isolateColor(float *d_hsv, int width, int height, int channels, float targetHue, float tolerance) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::isolateColor<<<gridSize, blockSize>>>(d_hsv, width, height, channels, targetHue, tolerance);

    checkKernelError("İsolate Color");

    cudaDeviceSynchronize();
}

void OperationWrapper::colorReplacement(float *d_hsv, int width, int height, int channels, float targetHue, float tolerance, float replacementHue) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::colorReplacement<<<gridSize, blockSize>>>(d_hsv, width, height, channels, targetHue, tolerance, replacementHue);

    checkKernelError("İsolate Color");

    cudaDeviceSynchronize();
}
