#include "../../include/UI/w_LeftPanel.h"
#include <iostream>
#include <algorithm> // std::min için eklendi

#include "imgui.h"
#include "UI/w_TopPanel.h"

// --- RENDER DÖNGÜSÜ ---
void LeftPanel::render(float displayWidth, float displayHeight) {
    float topPanelHeight = TopPanel::getPanelHeight();
    float realScreenHeight = ImGui::GetIO().DisplaySize.y;

    float panelWidth = 840.0f;
    float xPos = 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;
    float panelHeight = realScreenHeight - yPos - 15.0f;

    if (panelHeight < 100.0f) panelHeight = 100.0f;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.03f, 0.6f));

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags leftPanel_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("Hadi Başlayalım!", nullptr, leftPanel_flags);
    ImGui::SetWindowFontScale(1.8f);
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Hadi Baslayalim!");
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    // ==========================================
    // PROJE IZGARASI (16:9)
    // ==========================================
    float windowVisibleX2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    ImGuiStyle& style = ImGui::GetStyle();

    float tileWidth = 256.0f;
    float tileHeight = 144.0f;

    for (size_t i = 0; i < projectStack.size(); ++i) {
        ImGui::PushID((int)i);
        ImGui::BeginGroup();

        ImVec2 startPos = ImGui::GetCursorPos(); // Grubun başlangıç noktası

        // 1. Görsel Alanı (Texture ID 0'dan büyükse başarılı yüklenmiştir)
        if (projectStack[i].textureID > 0) {
            // --- EN-BOY ORANI (ASPECT RATIO) HESAPLAMA ---
            float origW = (float)projectStack[i].size.x;
            float origH = (float)projectStack[i].size.y;

            // Eğer boyut 0 ise hatayı önlemek için varsayılan ver
            if (origW <= 0.0f) origW = tileWidth;
            if (origH <= 0.0f) origH = tileHeight;

            // Görseli 256x144 kutuya sığdırmak için küçültme oranını bul
            float scale = std::min(tileWidth / origW, tileHeight / origH);
            float renderW = origW * scale;
            float renderH = origH * scale;

            // Ortalamak için gereken X ve Y boşluklarını hesapla
            float offsetX = (tileWidth - renderW) * 0.5f;
            float offsetY = (tileHeight - renderH) * 0.5f;

            // İmleci ortalanmış noktaya taşı ve butonu oraya çiz
            ImGui::SetCursorPos(ImVec2(startPos.x + offsetX, startPos.y + offsetY));

            // UV koordinatlarını ekledik: ImVec2(0, 1) ve ImVec2(1, 0) görseli dikeyde aynalar
            if (ImGui::ImageButton(projectStack[i].name.c_str(), (ImTextureID)(intptr_t)projectStack[i].textureID, ImVec2(renderW, renderH), ImVec2(0, 1), ImVec2(1, 0))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
                projectStack[i].isSelected = true;
            }
        } else {
            // Görsel yüklenemediyse normal buton
            if (ImGui::Button("Gorsel\nYok", ImVec2(tileWidth, tileHeight))) {
                std::cout << "[UI] Proje secildi: " << projectStack[i].name << std::endl;
            }
        }

        // Metni her zaman görsel alanının hemen altına yazmak için imleci sabit konuma sıfırla
        ImGui::SetCursorPos(ImVec2(startPos.x, startPos.y + tileHeight + 5.0f));

        // 2. Altındaki İsim
        float textWidth = ImGui::CalcTextSize(projectStack[i].name.c_str()).x;
        float textIndent = (tileWidth - textWidth) * 0.5f;
        if (textIndent > 0.0f) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + textIndent);
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "%s", projectStack[i].name.c_str());

        ImGui::EndGroup();

        // 3. Grid Mantığı
        float lastGroupX2 = startPos.x + tileWidth; // Orijinal genişliği referans al
        float nextGroupX2 = lastGroupX2 + style.ItemSpacing.x + tileWidth;
        if (i + 1 < projectStack.size() && nextGroupX2 < windowVisibleX2) {
            ImGui::SameLine();
        } else {
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
        }

        ImGui::PopID();
    }

    ImGui::End();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

// --- YENİ PROJE EKLEME (TEMİZ VE YÜKSÜZ) ---
void LeftPanel::addProjectToStack(Kivilcim::ProjectData newProject) {
    // 1. Yeni projeye ID ata
    newProject.id = projectCounter++;

    // 2. İsim girilmemişse numaralandır
    if (newProject.name == "İsimsiz-1" || newProject.name.empty()) {
        newProject.name = "İsimsiz Proje " + std::to_string(newProject.id);
    }

    // 3. Stack'in en başına (sola) ekle
    projectStack.insert(projectStack.begin(), newProject);
    std::cout << "[Kivilcim UI] Yeni proje VRAM'e aktarildi: " << newProject.name << std::endl;
}