#include "w_BackgroundPanel.h"
#include "imgui.h"

void BackgroundPanel::render(float displayWidth, float displayHeight) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();

    // Paneli ana pencereye tamamen hapseder, kendi OS penceresi olmasını engeller
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGui::SetNextWindowPos(viewport->Pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(viewport->Size, ImGuiCond_Always);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGuiWindowFlags bgFlags = ImGuiWindowFlags_NoTitleBar |
                               ImGuiWindowFlags_NoCollapse |
                               ImGuiWindowFlags_NoResize |
                               ImGuiWindowFlags_NoMove |
                               ImGuiWindowFlags_NoBringToFrontOnFocus |
                               ImGuiWindowFlags_NoNavFocus |
                               ImGuiWindowFlags_NoDocking |
                               ImGuiWindowFlags_NoInputs;

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

    ImGui::Begin("background", nullptr, bgFlags);

    const char* altMetin = "Kıvılcım Görüntü Motoru ile destekleniyor";
    ImVec2 textSize = ImGui::CalcTextSize(altMetin);

    float paddingX = 15.0f;
    float paddingY = 15.0f;

    ImGui::SetCursorPos(ImVec2(paddingX, viewport->Size.y - textSize.y - paddingY));
    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "%s", altMetin);

    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(3);
}