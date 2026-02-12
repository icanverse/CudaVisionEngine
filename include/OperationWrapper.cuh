//
// Created by Can on 13.02.2026.
//

#ifndef CUDAVISIONENGINE_OPERATIONWRAPPER_CUH
#define CUDAVISIONENGINE_OPERATIONWRAPPER_CUH
#include <cstdio>
class OperationWrapper {
public:
    // Görüntü İşleme Fonksiyonları
    static void normalize(unsigned char* d_input, float* d_output, int width, int height);
    static void denormalize(float* d_input, unsigned char* d_output, int width, int height);

    // Matris İşlemleri
    static void add(const float* d_A, const float* d_B, float* d_C, int size, bool useSharedMem = true);
    static void subtract(const float* d_A, const float* d_B, float* d_C, int size);
    static void multiply(const float* d_A, const float* d_B, float* d_C, int size);

    // Alt Matris Bulma (Kofaktör/Determinant hesapları için)
    static void getSubMatrix(const float* d_in, float* d_out, int removeCol, int removeRow, int currentSize);

    static void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize);

private:
    // Yardımcı: Hata kontrolü
    static void checkKernelError(const char* operationName) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error [%s]: %s\n", operationName, cudaGetErrorString(err));
        }
    }
};

#endif //CUDAVISIONENGINE_OPERATIONWRAPPER_CUH