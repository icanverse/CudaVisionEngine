project(kivilcim-vision)

# src klasörü altındaki tüm cpp ve cu dosyalarını topla
file(GLOB_RECURSE VISION_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Eğer klasör henüz boşsa CMake hata vermesin diye geçici bir kontrol
if(NOT VISION_SOURCES)
    set(VISION_SOURCES "dummy.cpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp" "// Geçici dosya\n")
endif()

add_library(kivilcim-vision STATIC ${VISION_SOURCES})

# Dışarıdan bu modülü kullananlar include klasörünü görebilsin
target_include_directories(kivilcim-vision PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
# Sadece CUDA bağımlılığı var
target_link_libraries(kivilcim-vision PUBLIC
        kivilcim-core
        CUDA::cudart
)