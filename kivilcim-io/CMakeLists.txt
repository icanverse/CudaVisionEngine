# kivilcim-io/CMakeLists.txt

file(GLOB_RECURSE IO_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(kivilcim-io STATIC
        ${IO_SOURCES}
)

target_include_directories(kivilcim-io PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(kivilcim-io PRIVATE
        "${FFMPEG_DIR}/include"
        "${CMAKE_SOURCE_DIR}/lib"
)

find_library(AVCODEC_LIBRARY
        NAMES avcodec
        PATHS "${FFMPEG_DIR}/lib"
        REQUIRED
        NO_DEFAULT_PATH
)

find_library(AVFORMAT_LIBRARY
        NAMES avformat
        PATHS "${FFMPEG_DIR}/lib"
        REQUIRED
        NO_DEFAULT_PATH
)

find_library(AVUTIL_LIBRARY
        NAMES avutil
        PATHS "${FFMPEG_DIR}/lib"
        REQUIRED
        NO_DEFAULT_PATH
)

target_link_libraries(kivilcim-io PUBLIC
        kivilcim-core
        glfw
        OpenGL::GL
        "${AVCODEC_LIBRARY}"
        "${AVFORMAT_LIBRARY}"
        "${AVUTIL_LIBRARY}"
)