//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_CONVOLUTION_CUH
#define CUDAVISIONENGINE_CONVOLUTION_CUH

#define MAX_KERNEL_SIZE 11
#define TILE_SIZE 16

namespace Convolution {
    // DİKKAT: Parametre sırası ve 'const' kelimeleri birebir aynı olmalı!
    void launchConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel, dim3 blocks, dim3 threads);
    void launchConvolution_withSharedMemory(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel, dim3 blocks, dim3 threads);
}


#endif //CUDAVISIONENGINE_CONVOLUTION_CUH