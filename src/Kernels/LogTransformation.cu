#include "../../include/Kernels/LogTransformation.cuh"

__global__ void logTransformationVChannel(float* input, float* output, int width, int height, int channels) {
    unsigned int dx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int dy = threadIdx.y + blockDim.y * blockIdx.y;

    if ( dx < width && dy < height ) {
        unsigned int index = (dx + dy * width) * channels;

        float pixels_value = fmaxf(0.0f, input[index + 2]);
        float log_value = logf(pixels_value + 1.0f);

        output[index] = input[index];
        output[index + 1] = input[index + 1];
        output[index + 2] = log_value;

    }
}