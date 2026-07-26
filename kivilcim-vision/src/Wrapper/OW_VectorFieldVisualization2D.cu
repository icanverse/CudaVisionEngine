#include "OperationWrapper.cuh"
#include "Kernels/VectorFieldVisualization2D.cuh"

void OperationWrapper::applyVectorFieldColoring(float *d_data, const float *flowU, const float* flowV,
                                               int width, int height, int channels, float intensity) {

    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::applyVectorFieldColoring<<<gridSize, blockSize>>>(d_data, flowU, flowV, width, height, channels, intensity);

    checkKernelError("Apply Vector Field Coloring");
    cudaDeviceSynchronize();
}

void OperationWrapper::applyNormalMapVisualization(float *d_data, const float *flowU, const float *flowV, int width,
                                                   int height, int channels, float intensity) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::applyNormalMapVisualization<<<gridSize, blockSize>>>(d_data, flowU, flowV, width, height, channels, intensity);

    checkKernelError("Apply Normal Map");
    cudaDeviceSynchronize();
}

void OperationWrapper::applyQuiverPlotVisualization(float *d_data, const float *flowU, const float *flowV, int width, int height, int channels, float intensity) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::applyQuiverPlotVisualization<<<gridSize, blockSize>>>(d_data, flowU, flowV, width, height, channels, intensity);

    checkKernelError("Apply Quiver Pilot");
    cudaDeviceSynchronize();
}

void OperationWrapper::applyJetScalarColorPalette(float *d_data, const float *flowU, const float *flowV, int width, int height, int channels, float maxSpeed) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::applyQuiverPlotVisualization<<<gridSize, blockSize>>>(d_data, flowU, flowV, width, height, channels, maxSpeed);

    checkKernelError("Apply Jet Scalar Color Palette");
    cudaDeviceSynchronize();
}

void OperationWrapper::applyLineIntegralConvolution(float *d_data, const float *flowU, const float *flowV, int width, int height, int channels, int steps) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::applyLineIntegralConvolution<<<gridSize, blockSize>>>(d_data, flowU, flowV, width, height, channels, steps);

    checkKernelError("Apply Line Integral Palette");
    cudaDeviceSynchronize();
}

