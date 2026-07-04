#include "UI/w_RightPanel.h"
#include "imgui.h"

void RightPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 350.0f;
    float panelHeight = displayHeight - 70.0f;
    if (panelHeight < 100.0f) panelHeight= 100.0f;

    float x_poz = displayWidth - panelWidth - 15.0f;
    float y_poz = 50.0f;

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(x_poz, y_poz), ImGuiCond_Always);

    ImGuiWindowFlags sirca_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove;

    ImGui::Begin("SircaKontrol", nullptr, sirca_flags);
    ImGui::Text("Yükle");
    ImGui::Separator();
    ImGui::End();
}