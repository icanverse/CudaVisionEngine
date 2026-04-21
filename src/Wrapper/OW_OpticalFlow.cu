#include "OperationWrapper.cuh"
#include "Kernels/OpticalFlow.cuh"

void OperationWrapper::opticalFlowLucasKanade(const float *currentFrame, const float *previousFrame, int width, int height, int channels, float *flowU, float *flowV) {

    // 1. BLOK VE GRİD HESABI (Özel Durum)
    // Shared Memory kernelimiz TILE_SIZE = 16 üzerine kurulduğu için blok boyutunu kilitliyoruz!
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // 2. KERNELİ ATEŞLE
    // (Eğer kernelinin adını calculateOpticalFlowShared yerine opticalFlowLucasKanadeKernel yaptıysan adını değiştir)
    ::opticalFlowLucasKanade<<<gridSize, blockSize>>>(
        currentFrame,
        previousFrame,
        width,
        height,
        channels,
        flowU,
        flowV
    );

    // 3. GÜVENLİK VE SENKRONİZASYON
    checkKernelError("Optical Flow Lucas-Kanade");
    cudaDeviceSynchronize();
}

