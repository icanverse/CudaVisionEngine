cmake_minimum_required(VERSION 4.1)

# LANGUAGES kısmına 'CUDA' eklemek zorunludur!
project(CudaVisionEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# =========================================================================
# 1. YÖNTEM: FetchContent İLE GLFW'Yİ OTOMATİK İNDİR VE DERLE
# =========================================================================
include(FetchContent)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
)

FetchContent_Declare(
        ffmpeg_bins
        URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
)

FetchContent_MakeAvailable(ffmpeg_bins)
FetchContent_GetProperties(ffmpeg_bins)
set(FFMPEG_DIR ${ffmpeg_bins_SOURCE_DIR})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

find_package(OpenGL REQUIRED)
# =========================================================================

# --- DEAR IMGUI KAYNAK DOSYALARI ---
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/include/ImGUI")
set(IMGUI_BACKENDS_DIR "${CMAKE_SOURCE_DIR}/include/ImGUI/backends")

set(IMGUI_SOURCES
        "${IMGUI_DIR}/imgui.cpp"
        "${IMGUI_DIR}/imgui_demo.cpp"
        "${IMGUI_DIR}/imgui_draw.cpp"
        "${IMGUI_DIR}/imgui_tables.cpp"
        "${IMGUI_DIR}/imgui_widgets.cpp"
        "${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp"
        "${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp"


)

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lib
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FFMPEG_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/nv_sdk_interface
        ${IMGUI_DIR}              # ImGui Core
        ${IMGUI_BACKENDS_DIR}     # ImGui Backends
)

link_directories(
        ${CMAKE_SOURCE_DIR}/lib
        ${FFMPEG_DIR}/lib)

file(GLOB_RECURSE SOURCE_FILES "src/*.cu" "src/*.cpp")

# Dikkat: IMGUI_SOURCES değişkeni buraya eklendi!
add_executable(CudaVisionEngine ${SOURCE_FILES} ${IMGUI_SOURCES}
        src/EngineFactory/EngineFactory.cu
        include/EngineFactory/EngineFactory.cuh
        src/Kernels/ElementaryMatrixOp.cu
        include/Kernels/ElementaryMatrixOp.cuh
        src/Kernels/Smoothing.cu
        include/Kernels/Smoothing.cuh
        src/Kernels/Convolution.cu
        include/Kernels/Convolution.cuh
        src/Kernels/Thresholding.cu
        include/Kernels/Thresholding.cuh
        src/Wrapper/OperationWrapper.cu
        include/OperationWrapper.cuh
        src/main.cpp
        include/main.h
        src/Kernels/ColorSpaceConverter.cu
        include/ColorSpaceConverter.cuh
        src/Kernels/ColorOperation.cu
        include/Kernels/ColorOperation.cuh
        src/Kernels/ToneAdjustment.cu
        include/Kernels/ToneAdjustment.cuh
        include/io/IDataSource.h
        include/io/IRenderTarget.h
        src/io/StbImageSource.cpp
        include/io/StbImageSource.h
        src/io/StbImageTarget.cpp
        include/io/StbImageTarget.h
        src/io/GlfwWindowTarget.cpp
        include/io/GlfwWindowTarget.h
        src/io/GlfwInteropTarget.cpp
        include/io/GlfwInteropTarget.h
        src/io/Video/Demuxer.cpp
        include/io/Video/Demuxer.h
        src/io/Video/NvDecoder.cpp
        include/io/Video/NvDecoder.h
        src/Kernels/LogTransformation.cu
        include/Kernels/LogTransformation.cuh
        src/Kernels/Reduction.cu
        include/Kernels/Reduction.cuh
        src/Kernels/Normalization.cu
        include/Kernels/Normalization.cuh
        src/Kernels/MaskOperation.cu
        include/Kernels/MaskOperation.cuh
        src/EngineFactory/EngineFactory_Retinex.cu
        src/EngineFactory/EngineFactory_ToneAdjustments.cu
        src/EngineFactory/EngineFactory_ColorSpaceConverter.cu
        src/EngineFactory/EngineFactory_ColorOperation.cu
        src/EngineFactory/EngineFactory_Convolution.cu
        src/EngineFactory/EngineFactory_Transformation.cu
        src/Kernels/Flare.cu
        include/Kernels/Flare.cuh
        src/EngineFactory/EngineFactory_Flare.cu
        src/EngineFactory/EngineFactory_3DLUT.cu
        src/Kernels/LUT_3D.cu
        include/Kernels/LUT_3D.cuh
        src/Kernels/OpticalFlow.cu
        include/Kernels/OpticalFlow.cuh
        src/EngineFactory/EngineFactory_OpticalFlow.cu
        src/Kernels/VectorFieldVisualization2D.cu
        include/Kernels/VectorFieldVisualization2D.cuh
        src/EngineFactory/EngineFactory_VectorFieldVisualization2D.cu
        src/Wrapper/OW_VectorFieldVisualization2D.cu
        src/Wrapper/OW_ToneAdjustment.cu
        src/Wrapper/OW__Convolution.cu
        src/Wrapper/OW_ColorSpaceConverter.cu
        src/Wrapper/OW_ColorOperation.cu
        src/Wrapper/OW_OpticalFlow.cu
        src/Kernels/Graphics/RayTracer.cu
        include/Kernels/Graphics/RayTracer.cuh
        include/Graphics/Scene.cuh
        include/Graphics/Types3D.cuh
        src/Graphics/Renderer3D.cu
        include/Graphics/Renderer3D.cuh
        src/Graphics/Scene.cpp
        include/io/NetworkStream.h
        include/io/Graphics/ObjectLoader.h
        include/io/Graphics/SceneBuilder.h
        include/io/Graphics/KvlcmParser.h
        include/io/Graphics/SceneDescription.h
        src/Kernels/Graphics/Shaders/Shaders.cu
        include/Graphics/Shaders.cuh
        include/Graphics/ElementaryNodes.h
        src/UI/MainUI.cpp
        include/UI/MainUI.h
        src/Compute/ParticleSystem/ParticleSystem.cu
        include/Compute/ParticleSystem/ParticleSystem.cuh
)

target_link_libraries(CudaVisionEngine
        PRIVATE CUDA::cudart
        glfw OpenGL::GL
        avcodec avformat avutil
        cuda nvcuvid
)

add_custom_command(TARGET CudaVisionEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${FFMPEG_DIR}/bin"
        "$<TARGET_FILE_DIR:CudaVisionEngine>"
)