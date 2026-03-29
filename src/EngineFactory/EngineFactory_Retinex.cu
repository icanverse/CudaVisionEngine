#include "../../include/EngineFactory/EngineFactory.cuh"
#include "../../include/Kernels/LogTransformation.cuh"
#include "../../include/Kernels/ElementaryMatrixOp.cuh"
#include "../../include/Kernels/Reduction.cuh"
#include "../../include/Kernels/Normalization.cuh"
#include "../OperationWrapper.cuh"
#include <cuda_runtime.h>

#include "Kernels/MaskOperation.cuh"

EngineFactory &EngineFactory::applyRetinex() {
    int total_pixels = width * height;
    int threads1D = 256;

    dim3 threads2D(16, 16);
    dim3 blocks2D((width + threads2D.x - 1) / threads2D.x,
                  (height + threads2D.y - 1) / threads2D.y);

    // Gaussian Bulanıklaştırma
    // Swap sonrası: d_data = bulanık, d_temp_data = orijinal
    applyGaussianBlurVChannel();

    // Log Dönüşümü
    logTransformationVChannel<<<blocks2D, threads2D>>>(d_temp_data, d_temp_data, width, height, channels);
    logTransformationVChannel<<<blocks2D, threads2D>>>(d_data,      d_data,      width, height, channels);
    cudaDeviceSynchronize();

    // Çıkarma (log orijinal - log bulanık)
    ::subVCh<<<blocks2D, threads2D>>>(d_temp_data, d_data, d_temp_data, width, height, channels);
    cudaDeviceSynchronize();
    std::swap(d_data, d_temp_data);

    // Min-Max Reduction
    k_InitGlobalMinMax<<<1, 1>>>(d_global_min, d_global_max);
    cudaDeviceSynchronize();
    size_t sharedMemSize = 2 * threads1D * sizeof(float);
    ::k_MinMaxReduction<<<(total_pixels + threads1D - 1) / threads1D, threads1D, sharedMemSize>>>(d_data, d_global_min, d_global_max, width, height, channels);
    cudaDeviceSynchronize();

    // Normalizasyon
    OperationWrapper::applyRetinexNormalize(d_data, d_global_min, d_global_max, total_pixels, channels);
    cudaDeviceSynchronize();

    // Orijinal ile Harman (d_mask = orijinal HSV)

    ::blendVChannel<<<blocks2D, threads2D>>>(d_mask_data, d_data, d_data, width, height, channels, 1.0f);    cudaDeviceSynchronize();

    return *this;
}
