set(IMGUI_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ImGUI"
)

set(IMGUI_BACKENDS_DIR
        "${IMGUI_DIR}/backends"
)

set(IMGUI_SOURCES
        "${IMGUI_DIR}/imgui.cpp"
        "${IMGUI_DIR}/imgui_demo.cpp"
        "${IMGUI_DIR}/imgui_draw.cpp"
        "${IMGUI_DIR}/imgui_tables.cpp"
        "${IMGUI_DIR}/imgui_widgets.cpp"

        "${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp"
        "${IMGUI_BACKENDS_DIR}/imgui_impl_opengl3.cpp"
        include/Data/LayerData.h
        include/Data/WorkspaceStateData.h
        include/Data/ToolData.h
        include/Data/MaskData.h
        include/Data/EditorData.h
        include/Data/ViewportData.h
        include/Data/HistoryData.h
        include/Data/ResourceData.h
        include/Data/PreferenceData.h
        src/ToolRegistry.cpp
        include/Data/ToolRegistry.h
)

file(GLOB_RECURSE UI_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_executable(CudaVisionEngine
        "${CMAKE_SOURCE_DIR}/main.cpp"
        "${CMAKE_SOURCE_DIR}/main.h"
        ${UI_SOURCES}
        ${IMGUI_SOURCES}
)

target_include_directories(CudaVisionEngine PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${IMGUI_DIR}"
        "${IMGUI_BACKENDS_DIR}"
)

target_link_libraries(CudaVisionEngine PRIVATE
        kivilcim-core
        kivilcim-io
        kivilcim-graphics
        glfw
        OpenGL::GL
)

add_custom_command(
        TARGET CudaVisionEngine
        POST_BUILD

        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${FFMPEG_DIR}/bin"
        "$<TARGET_FILE_DIR:CudaVisionEngine>"

        COMMENT "FFmpeg DLL dosyalari kopyalaniyor..."
)