# kivilcim-graphics/CMakeLists.txt
project(kivilcim-graphics)

file(GLOB_RECURSE GRAPHICS_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(NOT GRAPHICS_SOURCES)
    set(GRAPHICS_SOURCES "dummy.cpp")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp" "// Geçici dosya\n")
endif()

add_library(kivilcim-graphics STATIC ${GRAPHICS_SOURCES})

target_include_directories(kivilcim-graphics
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"

        PRIVATE
        "${CMAKE_SOURCE_DIR}/lib"
)
# OpenGL'e ve Çekirdeğe bağlı
target_link_libraries(kivilcim-graphics PUBLIC
        kivilcim-vision
        kivilcim-io
        OpenGL::GL
        glfw
)