cmake_minimum_required(VERSION 3.28)

# =========================================================================
# CUDA ARCHITECTURE
# project() çağrısından ÖNCE olmalı
# =========================================================================

if(WIN32)

    set(CMAKE_CUDA_COMPILER
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")
    set(CMAKE_CUDA_ARCHITECTURES 89)

elseif(UNIX)

    set(CMAKE_CUDA_COMPILER
            "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_ARCHITECTURES 87)

endif()
# =========================================================================
# PROJE
# =========================================================================

project(CudaVisionEngine
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Temel Sistem Bağımlılıkları
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

# =========================================================================
# BAĞIMLILIKLARIN İNDİRİLMESİ (FetchContent)
# =========================================================================
include(FetchContent)

# 1. GLFW (Arayüz ve Pencere Yönetimi İçin)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
)
FetchContent_MakeAvailable(glfw)

# 2. FFmpeg (Video I/O İşlemleri İçin)
if(WIN32)
    # Sadece Windows tarafında hazır derlenmiş kütüphaneyi indir
    FetchContent_Declare(
            ffmpeg_bins
            URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
    )
    FetchContent_MakeAvailable(ffmpeg_bins)
    FetchContent_GetProperties(ffmpeg_bins)
    set(FFMPEG_DIR ${ffmpeg_bins_SOURCE_DIR})
else()
    # Linux (Jetson) tarafında sisteme kurulu donanımsal FFmpeg'i bul
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)
endif()

# =========================================================================
# MODÜLLERİN (SUBDIRECTORIES) EKLENMESİ
# =========================================================================
add_subdirectory(kivilcim-core)
add_subdirectory(kivilcim-vision)
add_subdirectory(kivilcim-io)
add_subdirectory(kivilcim-graphics)
add_subdirectory(kivilcim-ui)