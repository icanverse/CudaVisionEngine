#include "UI/Layers/QuickTopRightToolbox.h"

#include "UI/Layers/ToolboxIconButton.h"
#include "imgui.h"

QuickTopRightToolbox::QuickTopRightToolbox()
    : lastAction(TopAction::NONE) {
    availableTools.push_back({
        TopAction::UNDO, "Geri Al", "Son islemi geri alir",
        Icon::Undo, "Islem gecmisinde bir adim geriye gider"
    });
    availableTools.push_back({
        TopAction::REDO, "Yinele", "Geri alinan islemi yineler",
        Icon::Redo, "Islem gecmisinde bir adim ileri gider"
    });
    availableTools.push_back({
        TopAction::TURN_LEFT, "Sola Dondur", "90 Derece Sola Dondur",
        Icon::Turn_Left, "Tuvali saat yonunun tersine cevirir"
    });
    availableTools.push_back({
        TopAction::TURN_RIGHT, "Saga Dondur", "90 Derece Saga Dondur",
        Icon::Turn_Right, "Tuvali saat yonunde cevirir"
    });
    availableTools.push_back({
        TopAction::ZOOM_IN, "Yakinlastir", "Tuvali Yakinlastir (+)",
        Icon::Zoom_In, "Calisma alanina yaklasir"
    });
    availableTools.push_back({
        TopAction::ZOOM_OUT, "Uzaklastir", "Tuvali Uzaklastir (-)",
        Icon::Zoom_Out, "Calisma alanindan uzaklasir"
    });
    availableTools.push_back({
        TopAction::MIRROR_HORIZONTAL, "Yatay Aynala", "Yatay Eksende Aynala",
        Icon::Mirror_Horizontal, "Goruntuyu yatay eksende cevirir"
    });
    availableTools.push_back({
        TopAction::MIRROR_VERTICAL, "Dikey Aynala", "Dikey Eksende Aynala",
        Icon::Mirror_Vertical, "Goruntuyu dikey eksende cevirir"
    });
}

void QuickTopRightToolbox::render(float displayWidth, float displayHeight) {
    const float iconSide = displayHeight / 32.0f;
    const float padding = 10.0f;
    const float spacing = 5.0f;
    const float buttonWidth = iconSide + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float toolbarWidth = padding * 2.0f +
        static_cast<float>(availableTools.size()) * buttonWidth +
        static_cast<float>(availableTools.size() - 1) * spacing;
    const float toolbarHeight = padding * 2.0f + iconSide +
                                ImGui::GetStyle().FramePadding.y * 2.0f;

    ImGui::SetCursorPos(ImVec2(displayWidth - toolbarWidth - 15.0f, 15.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.75f));
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
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.3f, 0.3f, 0.6f));

        if (ToolboxUI::IconButton(tool.name.c_str(), tool.icon, ImVec2(iconSide, iconSide))) {
            lastAction = tool.id;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());

        ImGui::PopStyleColor(3);
        if (i + 1 < availableTools.size()) ImGui::SameLine(0.0f, spacing);
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}

QuickTopRightToolbox::~QuickTopRightToolbox() = default;
