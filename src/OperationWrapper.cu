//
// Created by Can on 13.02.2026.
//

#include "OperationWrapper.cuh"
#include "ElementaryMatrixOp.cuh"
#include "Smoothing.cuh"
#include "ColorSpaceConverter.cuh"
#include <cstdio>

#include "ColorOperation.cuh"
#include "ToneAdjustment.cuh"
#include "Convolution.cuh"

void OperationWrapper::calculateGrid(int width, int height, dim3& gridSize, dim3& blockSize) {
    blockSize = dim3(16, 16);
    gridSize = dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
}


void OperationWrapper::normalize(unsigned char* d_input, float* d_output, int width, int height) {
    int totalElements = width * height;

    // 1D Grid Hesabı
    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    k_normalizeImage<<<gridSize, blockSize>>>(d_input, d_output, totalElements);

    checkKernelError("Normalize Image");
}

void OperationWrapper::denormalize(float* d_input, unsigned char* d_output, int width, int height) {
    int totalElements = width * height;

    int blockSize = 256;
    int gridSize = (totalElements + blockSize - 1) / blockSize;

    k_denormalizeImage<<<gridSize, blockSize>>>(d_input, d_output, totalElements);

    checkKernelError("Denormalize Image");
}

void OperationWrapper::smoothing2D(const float* A, float* Result, int width, int height, int channels, int kernelSize) {
    // 1. Blok Boyutlarını Belirle
    dim3 blockSize(16, 16);

    // 2. Izgara (Grid) Boyutlarını Hesapla
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // --- KRİTİK DÜZELTME BAŞLANGICI ---

    // 3. Shared Memory Boyutunu Hesapla
    // Kernel içinde kullandığımız formül: (Tile + 2*Radius) * (Tile + 2*Radius)
    int radius = kernelSize / 2;
    size_t sharedMemSize = (blockSize.x + 2 * radius) * (blockSize.y + 2 * radius) * sizeof(float);

    // 4. Kernel'ı Başlat (3. parametre olarak sharedMemSize eklendi)
    ::smoothing2D<<<gridSize, blockSize, sharedMemSize>>>(A, Result, width, height, channels, kernelSize);

    // --- KRİTİK DÜZELTME BİTİŞİ ---

    // 5. Hata Kontrolü
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in smoothing2D: %s\n", cudaGetErrorString(err));
    }

    // Kernel bitene kadar CPU'yu beklet (Debugging için iyidir)
    cudaDeviceSynchronize();
}

void OperationWrapper::rgbToHsv(const float* d_input, float* d_output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::rgbToHsv<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    checkKernelError("Convert RGB to HSV");

    cudaDeviceSynchronize();

}

void OperationWrapper::hsvToRgb(const float* d_input, float* d_output, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::hsvToRgb<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    checkKernelError("Convert HSV to RGB");

    cudaDeviceSynchronize();
}

void OperationWrapper::rgbToYuv(const float *A, float *Result, int width, int height, int channel) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::rgbToYuv<<<gridSize, blockSize>>>(A, Result, width, height, channel);

    checkKernelError("Convert RGB to YUV");

    cudaDeviceSynchronize();
}

void OperationWrapper::yuvToRgb(const float *A, float *Result, int width, int height, int channels) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::yuvToRgb<<<gridSize, blockSize>>>(A, Result, width, height, channels);

    checkKernelError("Convert YUV to RGB");

    cudaDeviceSynchronize();
}

void OperationWrapper::kernelNV12toRGB(const unsigned char *pNV12, unsigned char *pRGB, int width, int height, int pitch) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::kernelNV12toRGB<<<gridSize, blockSize>>>(pNV12, pRGB, width, height, pitch);

    checkKernelError("Convert YUV to RGB (NV12)");

    cudaDeviceSynchronize();
}


void OperationWrapper::isolateColor(float *d_hsv, int width, int height, int channels, float targetHue, float tolerance) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::isolateColor<<<gridSize, blockSize>>>(d_hsv, width, height, channels, targetHue, tolerance);

    checkKernelError("İsolate Color");

    cudaDeviceSynchronize();
}

void OperationWrapper::colorReplacement(float *d_hsv, int width, int height, int channels, float targetHue, float tolerance, float replacementHue) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize); // Tek satırda tertemiz!

    ::colorReplacement<<<gridSize, blockSize>>>(d_hsv, width, height, channels, targetHue, tolerance, replacementHue);

    checkKernelError("İsolate Color");

    cudaDeviceSynchronize();
}

void OperationWrapper::brightnessAdjustment(float *d_hsv, int width, int height, int channels, float value) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::brightnessAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, value);

    checkKernelError("Brightness Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::saturationAdjustment(float *d_hsv, int width, int height, int channels, float value) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::saturationAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, value);

    checkKernelError("Saturation Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::contrastAdjustment(float *d_hsv, int width, int height, int channels, float contrastFactor, float midpoint) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::contrastAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, contrastFactor, midpoint);

    checkKernelError("Contrast Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::shadowsHighlightsAdjustment(float *d_hsv, int width, int height, int channels, float shadowAmount, float highlightAmount) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::shadowsHighlightsAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, shadowAmount, highlightAmount);

    checkKernelError("Shadows - Highlights Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::temperatureAdjustment(float *d_rgb, int width, int height, int channels, float temperature) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::temperatureAdjustment<<<gridSize, blockSize>>>(d_rgb, width, height, channels, temperature);

    checkKernelError("Temperature Adjustment");

    cudaDeviceSynchronize();
}

void OperationWrapper::gammaCorrectionAdjustment(float *d_hsv, int width, int height, int channels, float gamma) {
    dim3 gridSize, blockSize;
    calculateGrid(width, height, gridSize, blockSize);

    ::gammaCorrectionAdjustment<<<gridSize, blockSize>>>(d_hsv, width, height, channels, gamma);

    checkKernelError("Gamma Correction Adjustment");

    cudaDeviceSynchronize();
}


void OperationWrapper::add(const float* d_A, const float* d_B, float* d_C, int size, bool useSharedMem) {
    // 2D Grid Hesabı
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    if (useSharedMem) {
        // op_control = true (Toplama)
       // matrix_add_with_sharedmem<<<grid, block>>>(d_A, d_B, d_C, size, true);
    } else {
        // Naive versiyon
        matrix_add<<<grid, block>>>(d_A, d_B, d_C, size);
    }

    checkKernelError("Matrix Add");
}

void OperationWrapper::subtract(const float* d_A, const float* d_B, float* d_C, int size) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    // op_control = false (Çıkarma) - Sadece shared mem kernelinde implemente edilmiş
    //matrix_add_with_sharedmem<<<grid, block>>>(d_A, d_B, d_C, size, false);

    checkKernelError("Matrix Subtract");
}

void OperationWrapper::multiply(const float* d_A, const float* d_B, float* d_C, int size) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y);

    matrix_mul<<<grid, block>>>(d_A, d_B, d_C, size);

    checkKernelError("Matrix Multiply");
}

void OperationWrapper::applyConvolution(const float* input, float* output, int width, int height, int channels, int kernelSize, const float* h_kernel) {

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    // DÜZELTME BURADA: Namespace kullanarak çağırıyoruz!
    Convolution::launchConvolution(input, output, width, height, channels, kernelSize, h_kernel, blocks, threads);

    checkKernelError("Apply Convolution");
    cudaDeviceSynchronize();
}

// ==========================================================
// HAZIR KONVOLÜSYON FİLTRELERİ (PRESETLER)
// ==========================================================

// 1. Kutu Bulanıklaştırma (Box Blur - 3x3)
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

// 2. Keskinleştirme (Sharpen - 3x3)
void OperationWrapper::applySharpen(const float* input, float* output, int width, int height, int channels) {
    int kSize = 3;
    // Merkezdeki pikseli çok parlat (5), etrafındakileri karart (-1)
    float kernel[9] = {
        0.0f, -1.0f,  0.0f,
       -1.0f,  5.0f, -1.0f,
        0.0f, -1.0f,  0.0f
   };
    applyConvolution(input, output, width, height, channels, kSize, kernel);
}

// 3. Kenar Bulma (Laplacian Edge Detection - 3x3)
void OperationWrapper::applyEdgeDetection(const float* input, float* output, int width, int height, int channels) {
    int kSize = 3;
    // Sadece renk değişiminin (türev) olduğu, yani zıtlık olan sınır çizgilerini bulur
    float kernel[9] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };
    applyConvolution(input, output, width, height, channels, kSize, kernel);
}

// 1. GAUSSIAN BLUR (5x5) - Canny'nin ilk adımıdır, Box Blur'dan çok daha kalitelidir
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

// 2. SOBEL X (Dikey Kenarlar)
void OperationWrapper::applySobelX(const float* input, float* output, int width, int height, int channels) {
    float kernel[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    applyConvolution(input, output, width, height, channels, 3, kernel);
}

// 3. SOBEL Y (Yatay Kenarlar)
void OperationWrapper::applySobelY(const float* input, float* output, int width, int height, int channels) {
    float kernel[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    applyConvolution(input, output, width, height, channels, 3, kernel);
}

// 4. EMBOSS (Kabartma Efekti)
void OperationWrapper::applyEmboss(const float* input, float* output, int width, int height, int channels) {
    float kernel[9] = {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };
    applyConvolution(input, output, width, height, channels, 3, kernel);
}


void OperationWrapper::getSubMatrix(const float* d_in, float* d_out, int removeCol, int removeRow, int currentSize) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((currentSize + block.x - 1) / block.x, (currentSize + block.y - 1) / block.y);

    //findSubMatrix<<<grid, block>>>(d_in, d_out, removeCol, removeRow, currentSize);

    checkKernelError("Get SubMatrix");
}