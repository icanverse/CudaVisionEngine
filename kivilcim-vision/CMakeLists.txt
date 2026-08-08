cmake_minimum_required(VERSION 3.28)

project(kivilcim-vision LANGUAGES CXX CUDA)

# =========================================================================
# SOURCE DOSYALARI
# =========================================================================

file(GLOB_RECURSE VISION_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Klasör boşsa CMake'in tamamen çökmesini engelle
if(NOT VISION_SOURCES)
    set(VISION_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp")
    file(WRITE
            "${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp"
            "// Geçici dosya\n"
    )
endif()

# =========================================================================
# TARGET
# =========================================================================

add_library(kivilcim-vision STATIC
        ${VISION_SOURCES}
)

# =========================================================================
# TARGET PROPERTIES
# =========================================================================

set_target_properties(kivilcim-vision PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
)

# =========================================================================
# INCLUDE
# =========================================================================

target_include_directories(kivilcim-vision
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# =========================================================================
# DEPENDENCIES
# =========================================================================

target_link_libraries(kivilcim-vision
        PUBLIC
        kivilcim-core
        CUDA::cudart
)