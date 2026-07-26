#ifndef KIVILCIM_CUDA_VISION_ENGINEFACTORY_CUH
#define KIVILCIM_CUDA_VISION_ENGINEFACTORY_CUH

#include "../../kivilcim-core/include/Cuda/CudaBuffer.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

class EngineFactory {
private:
    int width;
    int height;
    int channels;
    std::size_t totalElementCount;
    std::size_t totalPixelCount;

    // Sahiplik Core CudaBuffer siniflarindadir.
    Kivilcim::Core::Cuda::CudaBuffer<float> dataBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> tempDataBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> maskDataBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> globalMinBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> globalMaxBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> previousFrameBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> flowUBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float> flowVBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<unsigned char> byteScratchBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<float3> vertexBuffer;
    Kivilcim::Core::Cuda::CudaBuffer<int3> indexBuffer;

    // Gecis uyumlulugu: diger EngineFactory_*.cu dosyalari bu isimleri
    // kullanmaya devam eder. Bu pointerlar sahip degildir.
    float* d_data;
    float* d_temp_data;
    float* d_mask_data;
    float* d_global_min;
    float* d_global_max;
    float* d_prev_data;
    float* d_flow_u;
    float* d_flow_v;
    float3* d_vertices;
    int3* d_indices;

    cudaArray_t d_flareArray = nullptr;
    cudaTextureObject_t flareTexture = 0;
    int numTriangles = 0;

    void releaseTextureResources() noexcept;
    void refreshNonOwningViews() noexcept;
    void saveCurrentFrameAsPrevious();

public:
    cudaArray_t d_lutArray = nullptr;
    cudaTextureObject_t lutTexture = 0;

    EngineFactory(int w, int h, int c);
    ~EngineFactory();

    EngineFactory(const EngineFactory&) = delete;
    EngineFactory& operator=(const EngineFactory&) = delete;
    EngineFactory(EngineFactory&&) = delete;
    EngineFactory& operator=(EngineFactory&&) = delete;

    EngineFactory& uploadFrame(const unsigned char* cpuData);
    void downloadFrame(unsigned char* cpuData);

    void initTextureMemory(
        cudaArray_t& targetArray,
        cudaTextureObject_t& targetTexture,
        int textureWidth,
        int textureHeight
    );

    void init3DTextureMemory(
        const float* hostLutData,
        int lutSize,
        cudaArray_t& targetArray,
        cudaTextureObject_t& targetTexture
    );

    [[nodiscard]] int getWidth() const noexcept {
        return width;
    }

    [[nodiscard]] int getHeight() const noexcept {
        return height;
    }

    [[nodiscard]] int getChannels() const noexcept {
        return channels;
    }

    [[nodiscard]] std::size_t getElementCount() const noexcept {
        return totalElementCount;
    }

    [[nodiscard]] float* getDeviceData() noexcept {
        return dataBuffer.data();
    }

    [[nodiscard]] const float* getDeviceData() const noexcept {
        return dataBuffer.data();
    }

    void updateDeviceData(const float* newData);
    void copyToDeviceUchar(unsigned char* destinationDeviceData);

    EngineFactory& loadNV12DevicePointer(CUdeviceptr nv12DeviceData, int pitch);
    EngineFactory& loadFromVRAM(unsigned char* sourceDeviceData);

    EngineFactory& rgbToHsv();
    EngineFactory& hsvToRgb();
    EngineFactory& rgbToYuv();
    EngineFactory& yuvToRgb();
    EngineFactory& kernelNV12toRGB();
    EngineFactory& loadNV12DevicePointer();
    EngineFactory& retinexNormalize();
    EngineFactory& subVCh();

    EngineFactory& applyTemperature(float temperature);
    EngineFactory& applyShadowsHighlights(
        float shadowAmount,
        float highlightAmount
    );
    EngineFactory& applyGamma(float gamma);
    EngineFactory& logTransformation();

    EngineFactory& applyBoxBlur();
    EngineFactory& applySharpen();
    EngineFactory& applyEdgeDetection();
    EngineFactory& applyGaussianBlur5x5();
    EngineFactory& applySobelX();
    EngineFactory& applySobelY();
    EngineFactory& applyEmboss();
    EngineFactory& applyGaussianBlurVChannel();

    EngineFactory& isolateColor(float targetHue, float tolerance);
    EngineFactory& colorReplacement(
        float targetHue,
        float tolerance,
        float replacementHue
    );

    EngineFactory& applyRetinex();

    EngineFactory& blendTexture(
        cudaTextureObject_t texture,
        int textureWidth,
        int textureHeight,
        float targetX,
        float targetY,
        float opacity,
        bool isAdditive
    );

    EngineFactory& renderProceduralFlare(
        float x,
        float y,
        float hue,
        float opacity
    );

    EngineFactory& apply3DLUT(cudaTextureObject_t texture);
    EngineFactory& applyOpticalFlowLucasKanade(float strength = 1.0F);
    EngineFactory& applyVectorFieldColoring(float intensity = 1.0F);
    EngineFactory& applyNormalMapVisualization(float intensity = 1.0F);
    EngineFactory& applyQuiverPlotVisualization(float intensity = 1.0F);
    EngineFactory& applyJetScalarColorPalette(float maxSpeed);
    EngineFactory& applyLineIntegralConvolution(int steps);

    EngineFactory& loadMesh(
        const float3* cpuVertices,
        int vertexCount,
        const int3* cpuIndices,
        int triangleCount
    );

    EngineFactory& render3DScene(float time);
};

bool loadCubeLUT(
    const std::string& filepath,
    std::vector<float>& lutData,
    int& lutSize
);

#endif // KIVILCIM_CUDA_VISION_ENGINEFACTORY_CUH
