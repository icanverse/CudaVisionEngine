cmake_minimum_required(VERSION 4.1)

# LANGUAGES kısmına 'CUDA' eklemek zorunludur!
# Bu satır CMake'e "nvcc derleyicisini bul ve kullan" emrini verir.
project(CudaVisionEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# CUDA Mimarisi ayarı (native = senin ekran kartına uygun olanı seçer)
set(CMAKE_CUDA_ARCHITECTURES "native")

include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lib
        ${CUDAToolkit_INCLUDE_DIRS}
)

file(GLOB_RECURSE SOURCE_FILES "src/*.cu" "src/*.cpp")
add_executable(CudaVisionEngine ${SOURCE_FILES}
        src/EngineFactory.cu
        include/EngineFactory.cuh
        src/ElementaryMatrixOp.cu
        include/ElementaryMatrixOp.cuh
        src/Smoothing.cu
        include/Smoothing.cuh
        src/Convolution.cu
        include/Convolution.cuh
        src/Thresholding.cu
        include/Thresholding.cuh
        src/OperationWrapper.cu
        include/OperationWrapper.cuh
        src/main.cpp
        include/main.h)


# Kütüphaneleri bağla
target_link_libraries(CudaVisionEngine PRIVATE CUDA::cudart)