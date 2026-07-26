#include "Layers/QuickTopCenterToolbox.h"

#include "Layers/ToolboxIconButton.h"
#include "imgui.h"

#include <algorithm>

QuickTopCenterToolbox::QuickTopCenterToolbox()
    : currentTool(CenterToolAction::NONE) {
    availableTools = {
        {CenterToolAction::CIRCLE,     "Daire##TopCenter",      Icon::Circle,     "Daire ekler"},
        {CenterToolAction::LINE,       "Cizgi##TopCenter",      Icon::Line,       "Cizgi ekler"},
        {CenterToolAction::SQUARE,     "Dikdortgen##TopCenter", Icon::Square,     "Dikdortgen ekler"},
        {CenterToolAction::VECTOR,     "Vektor##TopCenter",     Icon::Vector,     "Vektor yolu ekler"},
        {CenterToolAction::BRUSH,      "Firca##TopCenter",      Icon::Brush,      "Firca aracini secer"},
        {CenterToolAction::ERASE,      "Silgi##TopCenter",      Icon::Erase,      "Silgi aracini secer"},
        {CenterToolAction::COLOR,      "Renk##TopCenter",       Icon::Color,      "Cizim rengini secer"},
        {CenterToolAction::TEXT,       "Metin##TopCenter",      Icon::Text,       "Metin ekler"},
        {CenterToolAction::TEXT_SIZE,  "MetinBoyutu##TopCenter",Icon::Text_Size,  "Metin boyutunu ayarlar"},
        {CenterToolAction::TEXT_COLOR, "MetinRengi##TopCenter", Icon::Text_Color, "Metin rengini ayarlar"}
    };
}

void QuickTopCenterToolbox::render(float displayWidth, float displayHeight) {
    const float iconSide = std::clamp(displayHeight / 32.0f, 22.0f, 34.0f);
    const float padding = 8.0f;
    const float spacing = 4.0f;
    const float buttonWidth = iconSide + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float panelWidth = padding * 2.0f +
        buttonWidth * static_cast<float>(availableTools.size()) +
        spacing * static_cast<float>(availableTools.size() - 1);
    const float panelHeight = padding * 2.0f + iconSide + ImGui::GetStyle().FramePadding.y * 2.0f;

    ImGui::SetCursorPos(ImVec2((displayWidth - panelWidth) * 0.5f, 58.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.78f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 15.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(padding, padding));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);

    ImGui::BeginChild("Ust Orta Arac Kutusu", ImVec2(panelWidth, panelHeight), true,
                      ToolboxUI::FloatingPanelFlags());

    for (std::size_t i = 0; i < availableTools.size(); ++i) {
        const Tool& tool = availableTools[i];
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
        if (i + 1 < availableTools.size()) ImGui::SameLine(0.0f, spacing);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

QuickTopCenterToolbox::~QuickTopCenterToolbox() = default;
