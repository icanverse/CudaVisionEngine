#ifndef CUDAVISIONENGINE_OPERATIONWRAPPER_CUH
#define CUDAVISIONENGINE_OPERATIONWRAPPER_CUH

#include <cstdio>

class OperationWrapper {

private:
    static void calculateGrid(int width, int height, dim3& gridSize, dim3& blockSize);


public:
    // Görüntü İşleme Fonksiyonları
    static void normalize(unsigned char* d_input, float* d_output, int width, int height);
    static void denormalize(float* d_input, unsigned char* d_output, int width, int height);

    // Matris İşlemleri ve Konvülasyon
    static void add(const float* d_A, const float* d_B, float* d_C, int size, bool useSharedMem = true);
    static void subtract(const float* d_A, const float* d_B, float* d_C, int size);
    static void multiply(const float* d_A, const float* d_B, float* d_C, int size);

    // Konvülasyon ve Hazır Filtreler
    static void applyConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel);
    static void applyBoxBlur(const float* input, float* output, int width, int height, int channels);
    static void applySharpen(const float* input, float* output, int width, int height, int channels);
    static void applyEdgeDetection(const float* input, float* output, int width, int height, int channels);


    // Alt Matris Bulma (Kofaktör/Determinant hesapları için)
    static void getSubMatrix(const float* d_in, float* d_out, int removeCol, int removeRow, int currentSize);

    // Blur İşlemi

    static void sharpen(const float* input, float* output, int width, int height, int channels);
    static void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize);

    // Renk Uzayı
    static void rgbToHsv(const float* d_input, float* d_output, int width, int height, int channels);
    static void hsvToRgb(const float* A, float* Result, int width, int height, int chanel);

    // Renk Uzayına Bağlı İşlemler
    static void isolateColor(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance);
    static void colorReplacement(float* d_hsv, int width, int height, int channels, float targetHue, float tolerance, float replacementHue);

    // Ton Ayarlamaları
    static void saturationAdjustment(float* d_hsv, int width, int height, int channels, float value);
    static void brightnessAdjustment(float* d_hsv, int width, int height, int channels, float value);
    static void contrastAdjustment(float* d_hsv, int width, int height, int channels, float contrastFactor, float midpoint);
    static void shadowsHighlightsAdjustment(float* d_hsv, int width, int height, int channels, float shadowAmount, float highlightAmount);
    static void temperatureAdjustment(float* d_rgb, int width, int height, int channels, float temperature);
    static void gammaCorrectionAdjustment(float* d_hsv, int width, int height, int channels, float gamma);
    static void applyGaussianBlur5x5(const float* input, float* output, int width, int height, int channels);
    static void applySobelX(const float* input, float* output, int width, int height, int channels);
    static void applySobelY(const float* input, float* output, int width, int height, int channels);
    static void applyEmboss(const float* input, float* output, int width, int height, int channels);





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