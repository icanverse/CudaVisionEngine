#include "UI/Layers/QuickRightToolbox.h"

#include "UI/Layers/ToolboxIconButton.h"
#include "imgui.h"

#include <algorithm>

QuickRightToolbox::QuickRightToolbox()
    : currentTool(RightToolAction::NONE) {
    availableTools = {
        {RightToolAction::CONTRAST,    "Kontrast##QuickRight", Icon::Contrast,    "Kontrast ayarlarini acar"},
        {RightToolAction::TEMPERATURE, "Sicaklik##QuickRight", Icon::Temperature, "Renk sicakligi ayarlarini acar"}
    };
}

void QuickRightToolbox::render(float displayWidth, float displayHeight) {
    const float iconSide = std::clamp(displayHeight / 32.0f, 22.0f, 34.0f);
    const float padding = 10.0f;
    const float rowHeight = iconSide + ImGui::GetStyle().FramePadding.y * 2.0f;
    const float panelWidth = iconSide + padding * 2.0f + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float panelHeight = padding * 2.0f + rowHeight * static_cast<float>(availableTools.size()) +
                              5.0f * static_cast<float>(availableTools.size() - 1);

    ImGui::SetCursorPos(ImVec2(displayWidth - panelWidth - 15.0f,
                              (displayHeight - panelHeight) * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.78f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 15.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);

    ImGui::BeginChild("Sag Arac Kutusu", ImVec2(panelWidth, panelHeight), true,
                      ToolboxUI::FloatingPanelFlags());
    for (const Tool& tool : availableTools) {
        const bool selected = currentTool == tool.id;
        ImGui::PushStyleColor(ImGuiCol_Button, selected
            ? ImVec4(0.85f, 0.45f, 0.0f, 1.0f)
            : ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.35f, 0.0f, 1.0f));

        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, ImVec2(iconSide, iconSide))) {
            currentTool = tool.id;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());

        ImGui::PopStyleColor(3);
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
    }
    ImGui::EndChild();

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

QuickRightToolbox::~QuickRightToolbox() = default;
