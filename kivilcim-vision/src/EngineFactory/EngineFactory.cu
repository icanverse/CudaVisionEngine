#include "EngineFactory/EngineFactory.cuh"

#include "../../kivilcim-core/include/Cuda/CudaError.h"
#include "Kernels/Normalization.cuh"
#include "OperationWrapper.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

std::size_t checkedProduct(
    int width,
    int height,
    int channels
) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::invalid_argument(
            "EngineFactory dimensions and channel count must be positive."
        );
    }

    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    const auto c = static_cast<std::size_t>(channels);

    if (w > std::numeric_limits<std::size_t>::max() / h) {
        throw std::overflow_error("EngineFactory pixel count overflow.");
    }

    const std::size_t pixels = w * h;
    if (pixels > std::numeric_limits<std::size_t>::max() / c) {
        throw std::overflow_error("EngineFactory element count overflow.");
    }

    return pixels * c;
}

std::size_t checkedPixelCount(int width, int height) {
    return checkedProduct(width, height, 1);
}

int blockCount(std::size_t elementCount, int threadsPerBlock) {
    const std::size_t blocks =
        (elementCount + static_cast<std::size_t>(threadsPerBlock) - 1) /
        static_cast<std::size_t>(threadsPerBlock);

    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CUDA grid size exceeds int range.");
    }

    return static_cast<int>(blocks);
}

} // namespace

EngineFactory::EngineFactory(int w, int h, int c)
    : width(w),
      height(h),
      channels(c),
      totalElementCount(checkedProduct(w, h, c)),
      totalPixelCount(checkedPixelCount(w, h)),
      dataBuffer(totalElementCount),
      tempDataBuffer(totalElementCount),
      maskDataBuffer(totalElementCount),
      globalMinBuffer(1),
      globalMaxBuffer(1),
      previousFrameBuffer(totalElementCount),
      flowUBuffer(totalPixelCount),
      flowVBuffer(totalPixelCount),
      byteScratchBuffer(totalElementCount),
      d_data(nullptr),
      d_temp_data(nullptr),
      d_mask_data(nullptr),
      d_global_min(nullptr),
      d_global_max(nullptr),
      d_prev_data(nullptr),
      d_flow_u(nullptr),
      d_flow_v(nullptr),
      d_vertices(nullptr),
      d_indices(nullptr) {
    refreshNonOwningViews();

    previousFrameBuffer.zero();
    flowUBuffer.zero();
    flowVBuffer.zero();

    std::cout
        << "[EngineFactory] Core bellek kaynaklari hazir: "
        << width << 'x' << height << 'x' << channels
        << '\n';
}

EngineFactory::~EngineFactory() {
    releaseTextureResources();
}

void EngineFactory::refreshNonOwningViews() noexcept {
    d_data = dataBuffer.data();
    d_temp_data = tempDataBuffer.data();
    d_mask_data = maskDataBuffer.data();
    d_global_min = globalMinBuffer.data();
    d_global_max = globalMaxBuffer.data();
    d_prev_data = previousFrameBuffer.data();
    d_flow_u = flowUBuffer.data();
    d_flow_v = flowVBuffer.data();
    d_vertices = vertexBuffer.data();
    d_indices = indexBuffer.data();
}

void EngineFactory::releaseTextureResources() noexcept {
    if (flareTexture != 0) {
        cudaDestroyTextureObject(flareTexture);
        flareTexture = 0;
    }

    if (d_flareArray != nullptr) {
        cudaFreeArray(d_flareArray);
        d_flareArray = nullptr;
    }

    if (lutTexture != 0) {
        cudaDestroyTextureObject(lutTexture);
        lutTexture = 0;
    }

    if (d_lutArray != nullptr) {
        cudaFreeArray(d_lutArray);
        d_lutArray = nullptr;
    }
}

void EngineFactory::updateDeviceData(const float* newData) {
    dataBuffer.copyFromDevice(newData, totalElementCount);
}

EngineFactory& EngineFactory::uploadFrame(
    const unsigned char* cpuData
) {
    byteScratchBuffer.copyFromHost(cpuData, totalElementCount);

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid =
        blockCount(totalElementCount, threadsPerBlock);

    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(
        byteScratchBuffer.data(),
        dataBuffer.data(),
        totalElementCount
    );

    Kivilcim::Core::Cuda::throwIfFailed(
        cudaGetLastError(),
        "k_normalizeImage launch"
    );
    Kivilcim::Core::Cuda::throwIfFailed(
        cudaDeviceSynchronize(),
        "k_normalizeImage synchronization"
    );

    return *this;
}

void EngineFactory::downloadFrame(unsigned char* cpuData) {
    if (cpuData == nullptr) {
        throw std::invalid_argument(
            "downloadFrame received a null destination."
        );
    }

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid =
        blockCount(totalElementCount, threadsPerBlock);

    k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(
        dataBuffer.data(),
        byteScratchBuffer.data(),
        totalElementCount
    );

    Kivilcim::Core::Cuda::throwIfFailed(
        cudaGetLastError(),
        "k_denormalizeImage launch"
    );
    Kivilcim::Core::Cuda::throwIfFailed(
        cudaDeviceSynchronize(),
        "k_denormalizeImage synchronization"
    );

    byteScratchBuffer.copyToHost(cpuData, totalElementCount);
}

EngineFactory& EngineFactory::loadFromVRAM(
    unsigned char* sourceDeviceData
) {
    if (sourceDeviceData == nullptr) {
        throw std::invalid_argument(
            "loadFromVRAM received a null source."
        );
    }

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid =
        blockCount(totalElementCount, threadsPerBlock);

    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(
        sourceDeviceData,
        dataBuffer.data(),
        totalElementCount
    );

    Kivilcim::Core::Cuda::throwIfFailed(
        cudaGetLastError(),
        "loadFromVRAM normalization launch"
    );
    Kivilcim::Core::Cuda::throwIfFailed(
        cudaDeviceSynchronize(),
        "loadFromVRAM normalization synchronization"
    );

    return *this;
}

EngineFactory& EngineFactory::loadNV12DevicePointer(
    CUdeviceptr nv12DeviceData,
    int pitch
) {
    if (nv12DeviceData == 0 || pitch <= 0) {
        throw std::invalid_argument(
            "loadNV12DevicePointer received invalid input."
        );
    }

    OperationWrapper::kernelNV12toRGB(
        reinterpret_cast<const unsigned char*>(nv12DeviceData),
        byteScratchBuffer.data(),
        width,
        height,
        pitch
    );

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid =
        blockCount(totalElementCount, threadsPerBlock);

    k_normalizeImage<<<blocksPerGrid, threadsPerBlock>>>(
        byteScratchBuffer.data(),
        dataBuffer.data(),
        totalElementCount
    );

    Kivilcim::Core::Cuda::throwIfFailed(
        cudaGetLastError(),
        "NV12 normalization launch"
    );
    Kivilcim::Core::Cuda::throwIfFailed(
        cudaDeviceSynchronize(),
        "NV12 normalization synchronization"
    );

    return *this;
}

void EngineFactory::copyToDeviceUchar(
    unsigned char* destinationDeviceData
) {
    if (destinationDeviceData == nullptr) {
        throw std::invalid_argument(
            "copyToDeviceUchar received a null destination."
        );
    }

    constexpr int threadsPerBlock = 256;
    const int blocksPerGrid =
        blockCount(totalElementCount, threadsPerBlock);

    k_denormalizeImage<<<blocksPerGrid, threadsPerBlock>>>(
        dataBuffer.data(),
        destinationDeviceData,
        totalElementCount
    );

    Kivilcim::Core::Cuda::throwIfFailed(
        cudaGetLastError(),
        "copyToDeviceUchar launch"
    );
    Kivilcim::Core::Cuda::throwIfFailed(
        cudaDeviceSynchronize(),
        "copyToDeviceUchar synchronization"
    );
}

void EngineFactory::saveCurrentFrameAsPrevious() {
    previousFrameBuffer.copyFromDevice(dataBuffer);
}

EngineFactory& EngineFactory::loadMesh(
    const float3* cpuVertices,
    int vertexCount,
    const int3* cpuIndices,
    int triangleCount
) {
    if (
        vertexCount < 0 ||
        triangleCount < 0 ||
        (vertexCount > 0 && cpuVertices == nullptr) ||
        (triangleCount > 0 && cpuIndices == nullptr)
    ) {
        throw std::invalid_argument("loadMesh received invalid mesh data.");
    }

    vertexBuffer.resizeDiscard(static_cast<std::size_t>(vertexCount));
    indexBuffer.resizeDiscard(static_cast<std::size_t>(triangleCount));

    if (vertexCount > 0) {
        vertexBuffer.copyFromHost(
            cpuVertices,
            static_cast<std::size_t>(vertexCount)
        );
    }

    if (triangleCount > 0) {
        indexBuffer.copyFromHost(
            cpuIndices,
            static_cast<std::size_t>(triangleCount)
        );
    }

    numTriangles = triangleCount;
    refreshNonOwningViews();

    std::cout
        << "[EngineFactory] 3D mesh yuklendi: "
        << numTriangles << " ucgen.\n";

    return *this;
}

void EngineFactory::initTextureMemory(
    cudaArray_t& targetArray,
    cudaTextureObject_t& targetTexture,
    int textureWidth,
    int textureHeight
) {
    if (textureWidth <= 0 || textureHeight <= 0) {
        throw std::invalid_argument(
            "Texture dimensions must be positive."
        );
    }

    if (targetTexture != 0) {
        Kivilcim::Core::Cuda::throwIfFailed(
            cudaDestroyTextureObject(targetTexture),
            "cudaDestroyTextureObject"
        );
        targetTexture = 0;
    }

    if (targetArray != nullptr) {
        Kivilcim::Core::Cuda::throwIfFailed(
            cudaFreeArray(targetArray),
            "cudaFreeArray"
        );
        targetArray = nullptr;
    }

    const cudaChannelFormatDesc channelDescription =
        cudaCreateChannelDesc<float4>();

    Kivilcim::Core::Cuda::throwIfFailed(
        cudaMallocArray(
            &targetArray,
            &channelDescription,
            static_cast<std::size_t>(textureWidth),
            static_cast<std::size_t>(textureHeight)
        ),
        "cudaMallocArray"
    );

    cudaResourceDesc resourceDescription{};
    resourceDescription.resType = cudaResourceTypeArray;
    resourceDescription.res.array.array = targetArray;

    cudaTextureDesc textureDescription{};
    textureDescription.addressMode[0] = cudaAddressModeBorder;
    textureDescription.addressMode[1] = cudaAddressModeBorder;
    textureDescription.filterMode = cudaFilterModeLinear;
    textureDescription.readMode = cudaReadModeElementType;
    textureDescription.normalizedCoords = 1;

    try {
        Kivilcim::Core::Cuda::throwIfFailed(
            cudaCreateTextureObject(
                &targetTexture,
                &resourceDescription,
                &textureDescription,
                nullptr
            ),
            "cudaCreateTextureObject"
        );
    } catch (...) {
        cudaFreeArray(targetArray);
        targetArray = nullptr;
        throw;
    }
}
