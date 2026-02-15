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

/// 2D Temel Blur efekti, paylaşımlı bellek kullanıyor
__global__ void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize) {
    extern __shared__ float s_data[];               // dinamik paylaşımlı bellek
    int radius = kernelSize / 2;
    int shmem_width = blockDim.x + 2 * radius;
    // int shmem_height = blockDim.y + 2 * radius;

    int th_x = threadIdx.x;                         // thread indeksi
    int th_y = threadIdx.y;
    int glob_x = blockIdx.x * blockDim.x + th_x;    // global indeks
    int glob_y = blockIdx.y * blockDim.y + th_y;

    for (int c = 0; c < channels; ++c) {

        // Bloktaki threadler iş birliği ile (coalesced) TILE + HALO shared mem'e yükler
        int num_pixels_sm = (blockDim.x + 2 * radius) * (blockDim.y + 2 * radius);
        int total_threads = blockDim.x * blockDim.y;
        int thread_id_linear = th_y * blockDim.x + th_x;

        // "Virtual Loop": Thread sayısından daha fazla piksel yüklememiz gerektiği için döngü kuruyoruz
        for (int i = thread_id_linear; i < num_pixels_sm; i += total_threads) {
            // Shared memory içindeki yerel koordinatlar
            int local_y = i / shmem_width;
            int local_x = i % shmem_width;

            // Bu yerel koordinatın karşılık geldiği global koordinat (Halo ofseti ile)
            int global_x = (blockIdx.x * blockDim.x) + local_x - radius;
            int global_y = (blockIdx.y * blockDim.y) + local_y - radius;

            // Sınır Kontrolü ve Yükleme
            if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
                int inputIndex = (global_y * width + global_x) * channels + c;
                s_data[i] = A[inputIndex];
            } else {
                s_data[i] = 0.0f; // Sınır dışı için 0 (veya clamp)
            }
        }

        __syncthreads();

        if (glob_x < width && glob_y < height) {
            float sum = 0.0f;
            int count = 0;

            // Shared memory üzerinde konvolüsyon
            // Shared memory'de bizim pikselimizin merkezi (ty + radius, tx + radius) konumundadır.
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    // Shared Memory indeksi
                    int sm_y = (th_y + radius) + i; // Yön: satır
                    int sm_x = (th_x + radius) + j; // Yön: sütun

                    int sm_index = sm_y * shmem_width + sm_x;

                    // Global sınır kontrolünü tekrar yapmaya gerek yok, yüklerken 0 atadık.
                    // Ancak "Count" hesabı için orijinal global koordinatlara bakmak gerekebilir
                    // Basit ortalama için yüklerken sınır kontrolü yaptıysak şuna bakabiliriz:

                    int neighbor_gx = glob_x + j;
                    int neighbor_gy = glob_y + i;

                    if (neighbor_gx >= 0 && neighbor_gx < width && neighbor_gy >= 0 && neighbor_gy < height) {
                        sum += s_data[sm_index];
                        count++;
                    }
                }
            }

            int outputIndex = (glob_y * width + glob_x) * channels + c;
            if (count > 0)
                Result[outputIndex] = sum / (float)count;
            else
                Result[outputIndex] = 0.0f;
        }

        __syncthreads();
    }
}