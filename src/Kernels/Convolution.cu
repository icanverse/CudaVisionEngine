#include "../../include/Convolution.cuh"

__constant__ float c_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];


__global__ void applyConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize) {
    // Thread Hazırlığı
    int dx = threadIdx.x;
    int dy = threadIdx.y;

    int tx = blockIdx.x * blockDim.x + dx;
    int ty = blockIdx.y * blockDim.y + dy;

    // Paylaşımlı Bellek Hazırlığı
    int radius = kernelSize / 2;

    // Paylaşımlı bellek genişliği (Örn: 16 + 2*1 = 18)
    int sharedW = blockDim.x + 2 * radius;

    // Dinamik paylaşımlı bellek tanımı (Launch sırasında boyutu verilecek)
    extern __shared__ float s_tile[];

    for (int c = 0; c < channels; c++) {
        for (int i = dy; i < sharedW; i += blockDim.y) {
            for (int j = dx; j < sharedW; j += blockDim.x) {

                // Global Memory'deki gerçek koordinat (Merkezden yarıçap kadar geriden başla)
                int gx = blockIdx.x * blockDim.x + j - radius;
                int gy = blockIdx.y * blockDim.y + i - radius;

                gx = min(max(gx, 0), width - 1);
                gy = min(max(gy, 0), height - 1);

                s_tile[i * sharedW + j] = input[(gy * width + gx) * channels + c];
            }
        }

        __syncthreads();

        if (tx < width && ty < height) {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {

                    float pixelVal = s_tile[(dy + radius + ky) * sharedW + (dx + radius + kx)];

                    int kernelIndex = (ky + radius) * kernelSize + (kx + radius);
                    sum += pixelVal * c_kernel[kernelIndex];
                }
            }
            output[(ty * width + tx) * channels + c] = sum;
        }

        __syncthreads();
    }
}

namespace Convolution {
    void launchConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel, dim3 blocks, dim3 threads) {

        size_t kernelBytes = kernelSize * kernelSize * sizeof(float);
        cudaMemcpyToSymbol(c_kernel, h_kernel, kernelBytes);

        int radius = kernelSize / 2;
        int sharedDim = threads.x + 2 * radius;
        size_t sharedMemSize = sharedDim * sharedDim * sizeof(float);

        applyConvolution<<<blocks, threads, sharedMemSize>>>(input, output, width, height, channels, kernelSize);

        cudaDeviceSynchronize();
    }
}

///// Paylaşımlı Bellek Kullanmayan Konvülasyon İşlemi

// __global__ void applyConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize) {
//     int dx = threadIdx.x;
//     int dy = threadIdx.y;
//
//     int tx = dx + blockDim.x * blockIdx.x;
//     int ty = dy + blockDim.y * blockIdx.y;
//
//     if (tx < width && ty < height) {
//
//         int halfSize = kernelSize / 2;
//         int radius = kernelSize / 2;
//
//         for (int c = 0; c < channels; c++) {
//
//             float sum = 0.0f;
//
//             for (int kx = -halfSize; kx <= halfSize; kx++) {
//                 for (int ky = -halfSize; ky <= halfSize; ky++) {
//                     int nx = min(max(tx + kx, 0), width - 1);
//                     int ny = min(max(ty + ky, 0), height - 1);
//
//                     float pixelVal = input[(ny*width + nx) * channels + c];
//
//                     int kernelIndex = (ky + halfSize) * kernelSize + (kx + halfSize);
//                     float weight = c_kernel[kernelIndex];
//
//                     sum += pixelVal * weight;
//                 }
//             }
//
//             output[(ty * width + tx) * channels + c] = sum;
//         }
//
//     } else {
//         return;
//     }
//
// }
