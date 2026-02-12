#include "../include/Smoothing.cuh"


__global__ void smoothing(const float* A,float* Result, int size) {
    unsigned int mythread = threadIdx.x + blockIdx.x * blockDim.x;

    if (mythread >= size) {
        return;
    }

    if (mythread == 0) {
        Result[mythread] = (A[mythread] + A[mythread + 1]) / 2;
    }
    else if (mythread == size - 1) {
        Result[mythread] = (A[mythread] + A[mythread - 1]) / 2;
    }
    else {
        Result[mythread] = (A[mythread - 1] + A[mythread] + A[mythread + 1]) / 3;
    }
}

__global__ void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;

    int k_size = kernelSize / 2;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        int count = 0;

        // 3. Kernel Döngüsü
        for (int i = -k_size; i <= k_size; i++) {
            for (int j = -k_size; j <= k_size; j++) {
                int n_x = x + i;
                int n_y = y + j;

                // Sınır Kontrolü
                if (n_x >= 0 && n_x < width && n_y >= 0 && n_y < height) {
                    // Doğru İndeksleme: (Satır * Genişlik + Sütun) * Kanal + MevcutKanal
                    int neighborIndex = (n_y * width + n_x) * channels + c;
                    sum += A[neighborIndex];
                    count++;
                }
            }
        }

        // 4. Yazma İndeksi
        int currentIndex = (y * width + x) * channels + c;

        // Bölme işlemi (Strength paydaya eklenmemeli, o interpolasyon içindir)
        Result[currentIndex] = sum / (float)count;
    }
}