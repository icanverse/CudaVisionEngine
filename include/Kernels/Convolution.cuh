#ifndef CUDAVISIONENGINE_CONVOLUTION_CUH
#define CUDAVISIONENGINE_CONVOLUTION_CUH

#define MAX_KERNEL_SIZE 31
#define TILE_SIZE 16

namespace Convolution {
    void launchConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel, dim3 blocks, dim3 threads);
    void launchConvolutionVChannel(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel, dim3 blocks, dim3 threads);

}


#endif //CUDAVISIONENGINE_CONVOLUTION_CUH