#include "Layers/QuickLeftToolbox.h"
#include "Layers/ToolboxIconButton.h"
#include "imgui.h"
#include "Data/ToolRegistry.h"

void QuickLeftToolbox::render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight) {
    if (!state) return; // Güvenlik kontrolü

    // YENİ: Araç listesini Registry'den (Kayıt Defterinden) tek satırda çekiyoruz
    const auto& availableTools = UIRegistry::ToolRegistry::GetCanvasTools();

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
        // Seçili durumu doğrudan merkez State'ten okuyoruz
        const bool selected = (state->tools.activeCanvasTool == tool.id);

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
            // Kullanıcı bir butona bastığında doğrudan merkez State verisini güncelle
            state->tools.activeCanvasTool = tool.id;

            // Eğer yeni bir araç seçildiyse, diğer paneldeki (örneğin renk, pozlama) ayar editörlerini kapat
            state->tools.activeAdjustment = Kdata::AdjustmentTool::NONE;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tool.tooltip.c_str());

        ImGui::PopStyleColor(3);
        ImGui::Dummy(ImVec2(0.0f, 5.0f));
    }

    ImGui::EndChild();
    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(2);
}
