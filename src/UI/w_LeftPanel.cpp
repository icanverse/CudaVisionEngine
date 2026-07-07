#include "../../include/UI/w_LeftPanel.h"
#include <iostream>

#include "imgui.h"
#include "UI/w_TopPanel.h"

void LeftPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 700.0f;
    float topPanelHeight = TopPanel::getPanelHeight();
    float panelHeight = displayHeight - topPanelHeight * 1.3f;

    if (panelHeight < 100.0f) panelHeight= 100.0f;

    float xPos = 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.03f, 0.6f));

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags leftPanel_flags =
                                           ImGuiWindowFlags_NoResize   |
                                           ImGuiWindowFlags_NoCollapse |
                                           ImGuiWindowFlags_NoMove     |
                                           ImGuiWindowFlags_NoTitleBar; // Başlık çubuğunu kaldırdık (Daha temiz)

    ImGui::Begin("Hadi Başlayalım!", nullptr, leftPanel_flags);
    ImGui::SetWindowFontScale(1.8f);
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Hadi Başlayalım!"); // Turuncu vurgulu başlık
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f));


    // 3. Fotoğraf Stack'ini Listele
    // if (!photoStack.empty()) {
    //     for (size_t i = 0; i < photoStack.size(); ++i) {
    //         // Her fotoğraf için Y koordinatını bir miktar aşağı kaydır
    //         float photo_y = y_start + (i * 0.15f);
    //
    //         // Fotoğrafı render etme mantığı (burada texture bind işlemleri olacak)
    //         // drawTexture(photoStack[i], x_start + 0.05f, photo_y);
    //     }
    // }


    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void LeftPanel::addPhotoToStack(const std::string& photoPath) {
    // photoStack.push_back(photoPath);
    // std::cout << "[UI] Stack'e yeni fotoğraf eklendi: " << photoPath << std::endl;
}