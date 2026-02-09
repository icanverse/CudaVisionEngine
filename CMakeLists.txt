cmake_minimum_required(VERSION 4.1)
project(CudaVisionEngine LANGUAGES CXX CUDA)


include_directories(lib)
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include")


find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(CudaVisionEngine
        src/ElementaryMatrixOp.cu
        include/ElementaryMatrixOp.cuh
        src/EngineManagement.cu
        include/EngineManagement.cuh
        src/stb_impl.cpp
)

set_target_properties(CudaVisionEngine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CudaVisionEngine PRIVATE CUDA::cudart)