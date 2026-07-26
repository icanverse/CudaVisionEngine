#include "Layers/QuickLeftToolbox.h"

#include "Layers/ToolboxIconButton.h"
#include "imgui.h"

QuickLeftToolbox::QuickLeftToolbox()
    : currentTool(ActiveTool::NONE) {
    availableTools.push_back({
        ActiveTool::MOVE, "Kaydir", "Tuvali Kaydir (El Araci)",
        Icon::Move, "Calisma alaninda gezinmeyi saglar"
    });
    availableTools.push_back({
        ActiveTool::SELECT_REGION_RECTANGLE, "Kare Secim", "Dikdortgen Secim Araci (ROI)",
        Icon::Select_Region_Rectangle, "Dikdortgen seklinde alan secer"
    });
    availableTools.push_back({
        ActiveTool::SELECT_REGION_FREE, "Serbest Secim", "Serbest Alan Secimi (Lasso)",
        Icon::Select_Region_Free, "Tuvalde serbest sekilde alan secer"
    });
    availableTools.push_back({
        ActiveTool::BRUSH, "Firca", "Serbest Boyama Araci",
        Icon::Brush, "Pikselleri serbestce boyar"
    });
    availableTools.push_back({
        ActiveTool::COLOR, "Renk", "Renk Secici",
        Icon::Color, "Firca ve metin rengini belirler"
    });
    availableTools.push_back({
        ActiveTool::TEXT, "Metin", "Metin Araci",
        Icon::Text, "Tuvale yazi katmani ekler"
    });
}

void QuickLeftToolbox::render(float displayWidth, float displayHeight) {
    const float toolbarWidth = 50.0f;
    const float toolbarHeight = 20.0f + 75.0f * static_cast<float>(availableTools.size());
    const float xPos = 15.0f;
    const float yPos = (displayHeight - toolbarHeight) * 0.5f;

    ImGui::SetCursorPos(ImVec2(xPos, yPos));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.75f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.85f, 0.45f, 0.0f, 0.4f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 15.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);

    ImGui::BeginChild("Sol Arac Kutusu", ImVec2(toolbarWidth, toolbarHeight), true,
                      ToolboxUI::FloatingPanelFlags());

    for (const auto& tool : availableTools) {
        const bool selected = currentTool == tool.id;
        ImGui::PushStyleColor(ImGuiCol_Button, selected
            ? ImVec4(0.85f, 0.45f, 0.0f, 1.0f)
            : ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, selected
            ? ImVec4(1.0f, 0.55f, 0.0f, 1.0f)
            : ImVec4(0.2f, 0.2f, 0.2f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, selected
            ? ImVec4(0.65f, 0.35f, 0.0f, 1.0f)
            : ImVec4(0.3f, 0.3f, 0.3f, 0.6f));

        const ImVec2 iconSize(displayHeight / 32.0f, displayHeight / 32.0f);
        const float buttonWidth = iconSize.x + ImGui::GetStyle().FramePadding.x * 2.0f;
        const float offsetX = (ImGui::GetContentRegionAvail().x - buttonWidth) * 0.5f;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, iconSize)) {
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

QuickLeftToolbox::~QuickLeftToolbox() = default;
