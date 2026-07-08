#include "../../include/UI/w_LeftPanel.h"
#include <iostream>

#include "imgui.h"
#include "UI/w_TopPanel.h"

void LeftPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 820.0f;
    float topPanelHeight = TopPanel::getPanelHeight();

    float realScreenHeight = ImGui::GetIO().DisplaySize.y;

    float xPos = 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;

    float panelHeight = realScreenHeight - yPos - 15.0f;

    if (panelHeight < 100.0f) panelHeight = 100.0f;

    // ... [Buradan sonrası eski kodunla birebir aynı devam ediyor] ...
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
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Hadi Baslayalim!");
    ImGui::SetWindowFontScale(1.0f); // Fontu normale döndür
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    // ==========================================
    // PROJE IZGARASI (GRID SYSTEM) - 16:9 FIX
    // ==========================================
    float windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    ImGuiStyle& style = ImGui::GetStyle();

    // 16:9 Oranında yeni boyutlandırma (Genişlik: 256, Yükseklik: 144)
    float tileWidth = 256.0f;
    float tileHeight = 144.0f;

    for (size_t i = 0; i < photoStack.size(); ++i) {
        ImGui::PushID(i);

        // GRUPLAMA BAŞLANGICI: Buton ve yazıyı tek bir element gibi paketle
        ImGui::BeginGroup();

        // 1. Görsel Alanı (16:9 Buton)
        if (ImGui::Button("Gorsel\nYok", ImVec2(tileWidth, tileHeight))) {
            std::cout << "[UI] Proje secildi: " << photoStack[i].name << std::endl;
        }

        // 2. Altındaki İsim (Ortalanmış metin)
        float textWidth = ImGui::CalcTextSize(photoStack[i].name.c_str()).x;
        float textIndent = (tileWidth - textWidth) * 0.5f;
        if (textIndent > 0.0f) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + textIndent);
        }
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", photoStack[i].name.c_str()); // Yazıyı hafif gri yaptık

        ImGui::EndGroup();
        // GRUPLAMA BİTİŞİ

        // 3. Grid (Yan Yana Dizme) Mantığı
        float lastGroupX2 = ImGui::GetItemRectMax().x; // Tüm grubun sağ kenarı
        float nextGroupX2 = lastGroupX2 + style.ItemSpacing.x + tileWidth;

        // Eğer bir sonraki paket ekranı taşmıyorsa yanına koy
        if (i + 1 < photoStack.size() && nextGroupX2 < windowVisibleX2) {
            ImGui::SameLine();
        } else {
            // Taşıyorsa alt satıra geç ve biraz boşluk bırak
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
        }

        ImGui::PopID();
    }

    ImGui::End();

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void LeftPanel::addPhotoToStack(const std::string& photoPath) {
    ProjectItem newItem;
    newItem.name = "Isimsiz " + std::to_string(projectCounter++);
    newItem.imagePath = photoPath;
    // newItem.textureID = null; // Şimdilik boş

    photoStack.push_back(newItem);
    std::cout << "[UI] Yeni proje eklendi: " << newItem.name << " (" << photoPath << ")" << std::endl;
}