cmake_minimum_required(VERSION 4.1)

# Proje ve Diller
project(CudaVisionEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")

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
FetchContent_Declare(
        ffmpeg_bins
        URL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
)
FetchContent_MakeAvailable(ffmpeg_bins)
FetchContent_GetProperties(ffmpeg_bins)
set(FFMPEG_DIR ${ffmpeg_bins_SOURCE_DIR})

# =========================================================================
# MODÜLLERİN (SUBDIRECTORIES) EKLENMESİ
# =========================================================================

add_subdirectory(kivilcim-core)

# 1. Çekirdek Motor (Sadece C++ ve CUDA)
add_subdirectory(kivilcim-vision)

# 2. Girdi/Çıktı ve Veri İşleme (FFmpeg ve stb_image bağları)
add_subdirectory(kivilcim-io)

# 3. Görselleştirme (OpenGL ve CUDA Doku Köprüleri)
add_subdirectory(kivilcim-graphics)

# 4. Kullanıcı Arayüzü (Ana Çalıştırılabilir Dosya ve ImGui Burada Olacak)
add_subdirectory(kivilcim-ui)