#ifndef CUDAVISIONENGINE_VECTORFIELDVISUALIZATION2D_CUH
#define CUDAVISIONENGINE_VECTORFIELDVISUALIZATION2D_CUH

#define PI 3.14159265358979323846f

__global__ void applyVectorFieldColoring(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity);
__global__ void applyNormalMapVisualization(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity);
__global__ void applyQuiverPlotVisualization(float* data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity);
__global__ void applyJetScalarColorPalette(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float maxSpeed);

__device__ float random_noise(int x, int y);
__global__ void applyLineIntegralConvolution(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, int steps);

#endif //CUDAVISIONENGINE_VECTORFIELDVISUALIZATION2D_CUH