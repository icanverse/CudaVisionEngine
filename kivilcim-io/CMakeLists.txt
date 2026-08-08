# kivilcim-io/CMakeLists.txt

file(GLOB_RECURSE IO_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Klasör boşsa hata vermemesi için dummy dosya kontrolü
if(NOT IO_SOURCES)
    set(IO_SOURCES "dummy.cpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp" "// Geçici dosya\n")
endif()

add_library(kivilcim-io STATIC
        ${IO_SOURCES}
)

# Ortak özellikler
set_target_properties(kivilcim-io PROPERTIES
        CXX_STANDARD 20
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
)

target_include_directories(kivilcim-io PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# =========================================================================
# PLATFORM BAĞIMSIZ FFMPEG BAĞLANTISI
# =========================================================================
if(WIN32)
    # WINDOWS (RTX 4070) İÇİN AYARLAR
    target_include_directories(kivilcim-io PRIVATE
            "${FFMPEG_DIR}/include"
            "${CMAKE_SOURCE_DIR}/lib"
    )

    find_library(AVCODEC_LIBRARY NAMES avcodec PATHS "${FFMPEG_DIR}/lib" REQUIRED NO_DEFAULT_PATH)
    find_library(AVFORMAT_LIBRARY NAMES avformat PATHS "${FFMPEG_DIR}/lib" REQUIRED NO_DEFAULT_PATH)
    find_library(AVUTIL_LIBRARY NAMES avutil PATHS "${FFMPEG_DIR}/lib" REQUIRED NO_DEFAULT_PATH)

    target_link_libraries(kivilcim-io PUBLIC
            kivilcim-core
            glfw
            OpenGL::GL
            "${AVCODEC_LIBRARY}"
            "${AVFORMAT_LIBRARY}"
            "${AVUTIL_LIBRARY}"
    )
else()
    # LINUX / JETSON ORİN İÇİN AYARLAR
    # Ana CMake'teki pkg_check_modules ile bulunan FFMPEG değişkenlerini kullanıyoruz
    target_include_directories(kivilcim-io PRIVATE
            ${FFMPEG_INCLUDE_DIRS}
            "${CMAKE_SOURCE_DIR}/lib"
    )

    target_link_libraries(kivilcim-io PUBLIC
            kivilcim-core
            glfw
            OpenGL::GL
            ${FFMPEG_LIBRARIES}
    )
endif()