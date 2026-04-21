#include "OperationWrapper.cuh"
#include "../../include/Kernels/Convolution.cuh"

void OperationWrapper::applyConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel) {

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    Convolution::launchConvolution(input, output, width, height, channels, kernelSize, h_kernel, blocks, threads);

    checkKernelError("Apply Convolution");
    cudaDeviceSynchronize();
}

void OperationWrapper::applyConvolutionVChannel(const float *input, float *output, int width, int height, int channels, int kernelSize, const float* h_kernel) {
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);


    Convolution::launchConvolutionVChannel(input, output, width, height, channels, kernelSize, h_kernel, blocks, threads);

    checkKernelError("Apply Convolution");
    cudaDeviceSynchronize();
}

void OperationWrapper::applyBoxBlur(const float* input, float* output, int width, int height, int channels) {
    int kSize = 3;
    float w = 1.0f / 9.0f; // Tüm piksellerin eşit ortalaması
    float kernel[9] = {
        w, w, w,
        w, w, w,
        w, w, w
    };
    applyConvolution(input, output, width, height, channels, kSize, kernel);
}

void OperationWrapper::applySharpen(const float* input, float* output, int width, int height, int channels) {
    int kSize = 3;
    float kernel[9] = {
        0.0f, -1.0f,  0.0f,
       -1.0f,  5.0f, -1.0f,
        0.0f, -1.0f,  0.0f
   };
    applyConvolution(input, output, width, height, channels, kSize, kernel);
}

void OperationWrapper::applyEdgeDetection(const float* input, float* output, int width, int height, int channels) {
    int kSize = 3;
    float kernel[9] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };
    applyConvolution(input, output, width, height, channels, kSize, kernel);
}

void OperationWrapper::applyGaussianBlur5x5(const float* input, float* output, int width, int height, int channels) {
    int kSize = 5;
    float kernel[25] = {
        1/273.f,  4/273.f,  7/273.f,  4/273.f, 1/273.f,
        4/273.f, 16/273.f, 26/273.f, 16/273.f, 4/273.f,
        7/273.f, 26/273.f, 41/273.f, 26/273.f, 7/273.f,
        4/273.f, 16/273.f, 26/273.f, 16/273.f, 4/273.f,
        1/273.f,  4/273.f,  7/273.f,  4/273.f, 1/273.f
    };
    applyConvolution(input, output, width, height, channels, kSize, kernel);
}

void OperationWrapper::applySobelX(const float* input, float* output, int width, int height, int channels) {
    float kernel[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    applyConvolution(input, output, width, height, channels, 3, kernel);
}

void OperationWrapper::applySobelY(const float* input, float* output, int width, int height, int channels) {
    float kernel[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    applyConvolution(input, output, width, height, channels, 3, kernel);
}

void OperationWrapper::applyEmboss(const float* input, float* output, int width, int height, int channels) {
    float kernel[9] = {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };
    applyConvolution(input, output, width, height, channels, 3, kernel);
}

void OperationWrapper::applyGaussianBlurVChannel(const float* input, float* output, int width, int height, int channels) {
    int kSize = 31;
    float kernel[961];

    float sigma = 5.0f;  // büyük surround için
    int half = kSize / 2;
    float sum = 0.0f;

    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float val = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + half) * kSize + (x + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < kSize * kSize; i++) {
        kernel[i] /= sum;
    }

    applyConvolutionVChannel(input, output, width, height, channels, kSize, kernel);
}