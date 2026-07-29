#include "Layers/QuickTopRightToolbox.h"
#include "Layers/ToolboxIconButton.h"
#include "imgui.h"
#include <algorithm> // std::clamp için gerekli

#include "Data/ToolRegistry.h"

void QuickTopRightToolbox::render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight) {
    if (!state) return; // Güvenlik kontrolü

    // YENİ: Veriyi doğrudan Kayıt Defterinden çekiyoruz
    const auto& availableTools = UIRegistry::ToolRegistry::GetTopRightTools();

    const float iconSide = std::clamp(displayHeight / 32.0f, 22.0f, 34.0f);
    const float padding = 8.0f;
    const float spacing = 4.0f;
    const float buttonWidth = iconSide + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float toolbarWidth = padding * 2.0f +
        static_cast<float>(availableTools.size()) * buttonWidth +
        static_cast<float>(availableTools.size() - 1) * spacing;
    const float toolbarHeight = padding * 2.0f + iconSide + ImGui::GetStyle().FramePadding.y * 2.0f;

    ImGui::SetCursorPos(ImVec2(displayWidth - toolbarWidth - 15.0f, 15.0f));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.78f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 15.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);

    ImGui::BeginChild("Sag Ust Arac Kutusu", ImVec2(toolbarWidth, toolbarHeight), true,
                      ToolboxUI::FloatingPanelFlags());

    for (std::size_t i = 0; i < availableTools.size(); ++i) {
        const auto& tool = availableTools[i];

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.35f, 0.0f, 1.0f));

        // `.c_str()` dönüşümleri eksiksiz eklendi
        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, ImVec2(iconSide, iconSide))) {
            // YENİ: Anlık eylemi doğrudan motorun state'ine ateşle
            state->tools.lastFiredAction = tool.id;
        }

        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());

        ImGui::PopStyleColor(3);
        if (i + 1 < availableTools.size()) ImGui::SameLine(0.0f, spacing);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}
