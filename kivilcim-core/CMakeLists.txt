cmake_minimum_required(VERSION 3.28)

project(kivilcim-core LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(kivilcim-core STATIC
    src/Cuda/CudaError.cpp
        include/Cuda/HardwareInfoData.h
        include/Cuda/HardwareDetector.h
        src/Cuda/HardwareDetector.cpp
        include/Cuda/HardwareDetector.h
)

add_library(Kivilcim::Core ALIAS kivilcim-core)

target_compile_features(kivilcim-core PUBLIC cxx_std_20)

target_include_directories(kivilcim-core
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(kivilcim-core
    PUBLIC
        CUDA::cudart
    PRIVATE
        glfw
)

set_target_properties(kivilcim-core PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
