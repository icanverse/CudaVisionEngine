#pragma once

#include "imgui.h"
#include "io/AssetsManager/IconManager.h"

#include <cstdint>
#include <string>

namespace ToolboxUI {

// OpenGL textures have their origin at the lower-left, while ImGui expects the
// upper-left. Keeping the flipped UVs here prevents individual toolboxes from
// accidentally drawing icons upside down.
inline bool IconButton(const char* id, Icon icon, const ImVec2& size) {
    const unsigned int textureId = IconManager::Get(icon);

    if (textureId == 0) {
        const std::string fallbackId = std::string("?##") + id;
        return ImGui::Button(fallbackId.c_str(), size);
    }

    return ImGui::ImageButton(
        id,
        (ImTextureID)(intptr_t)textureId,
        size,
        ImVec2(0.0f, 1.0f),
        ImVec2(1.0f, 0.0f),
        ImVec4(0.0f, 0.0f, 0.0f, 0.0f),
        ImVec4(1.0f, 1.0f, 1.0f, 1.0f)
    );
}

inline ImGuiWindowFlags FloatingPanelFlags() {
    return ImGuiWindowFlags_NoResize |
           ImGuiWindowFlags_NoMove |
           ImGuiWindowFlags_NoScrollbar |
           ImGuiWindowFlags_NoCollapse |
           ImGuiWindowFlags_NoTitleBar;
}

} // namespace ToolboxUI
