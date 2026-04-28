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

    static void retinexNormalize(float* input, const float* global_min, const float* global_max, int width, int height, int channels);

    // Renk Uzayı
    static void rgbToHsv(const float* d_input, float* d_output, int width, int height, int channels);
    static void hsvToRgb(const float* A, float* Result, int width, int height, int chanel);
    static void rgbToYuv(const float* A, float* Result, int width, int height, int channel);
    static void yuvToRgb(const float* A, float* Result, int width, int height, int channels);
    static void kernelNV12toRGB(const unsigned char* pNV12, unsigned char* pRGB, int width, int height, int pitch);

    // Matris İşlemleri
    static void add(const float* d_A, const float* d_B, float* d_C, int size, bool useSharedMem = true);
    static void subtract(const float* d_A, const float* d_B, float* d_C, int size);
    static void multiply(const float* d_A, const float* d_B, float* d_C, int size);

    // Diğer Matris işlemler
    static void getSubMatrix(const float* d_in, float* d_out, int removeCol, int removeRow, int currentSize);
    static void logTransformation(float* input, float* output, int width, int height, int channels);
    static void k_MinMaxReduction(const float* input, float* global_min, float* global_max, int width, int height, int channels);

    // Konvülasyon ve Hazır Filtreler
    static void applyConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel);
    static void applyBoxBlur(const float* input, float* output, int width, int height, int channels);
    static void applySharpen(const float* input, float* output, int width, int height, int channels);
    static void applyEdgeDetection(const float* input, float* output, int width, int height, int channels);


    static void applyConvolutionVChannel(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel);
    static void applyGaussianBlurVChannel(const float* input, float* output, int width, int height, int channels);

    // Blur İşlemi
    static void smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize);


    // Renk Uzayına Bağlı Hazır Gelişmiş İşlemler
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

    //
    static void applyRetinexNormalize(float* d_data, const float* d_global_min, const float* d_global_max, int total_pixels, int channels);

    // Flare Maskesi Üretim Sarmalayıcısı
    static void generateFlareHSV(float* data, int width, int height, int channels,
                                 float flareX, float flareY,
                                 float baseHue, float baseSaturation, float falloff);

    // Evrensel Texture Blend Sarmalayıcısı
    static void applyTextureBlendKernel(float* data, int width, int height, int channels,
                                        cudaTextureObject_t overlayTex, int texWidth, int texHeight,
                                        float targetX, float targetY, float opacity, bool isAdditive);

    static void apply3DLUT(float* d_rgb, int width, int height, int channels, cudaTextureObject_t lutTexture);


    static void opticalFlowLucasKanade(const float* currentFrame, const float* previousFrame,
                                           int width, int height, int channels, float* flowU, float* flowV);



    static void applyVectorFieldColoring(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity);
    static void applyNormalMapVisualization(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity);
    static void applyQuiverPlotVisualization(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float intensity);
    static void applyJetScalarColorPalette(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, float maxSpeed);
    static void applyLineIntegralConvolution(float* d_data, const float* flowU, const float* flowV, int width, int height, int channels, int steps);

private:
    static void checkKernelError(const char* operationName) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error [%s]: %s\n", operationName, cudaGetErrorString(err));
        }
    }
};

#endif //CUDAVISIONENGINE_OPERATIONWRAPPER_CUH