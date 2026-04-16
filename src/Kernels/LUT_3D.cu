#include "../../include/Kernels/LUT_3D.cuh"

__global__ void apply3DLUT(float* d_rgb, int width, int height, int channels, cudaTextureObject_t lutTexture) {
    unsigned int dx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx < width && dy < height) {
        unsigned int index = (dx + dy * width) * channels;

        float r = d_rgb[index];
        float g = d_rgb[index + 1];
        float b = d_rgb[index + 2];

        float4 newColor = tex3D<float4>(lutTexture, r, g, b);

        d_rgb[index] = newColor.x;
        d_rgb[index + 1] = newColor.y;
        d_rgb[index + 2] = newColor.z;

    }
}