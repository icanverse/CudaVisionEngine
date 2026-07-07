#include "UI/w_RightPanel.h"
#include "imgui.h"
#include "UI/w_TopPanel.h"

void RightPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 450.0f;
    float topPanelHeight = TopPanel::getPanelHeight();
    float panelHeight = displayHeight - topPanelHeight * 1.3f;

    if (panelHeight < 100.0f) panelHeight= 100.0f;

    float xPos = displayWidth - panelWidth - 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;

    // Stil atamaları
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.06f, 1.0f));

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags rightPanel_flags =
                                        ImGuiWindowFlags_NoResize   |
                                        ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoMove;

    ImGui::Begin("Import Image", nullptr, rightPanel_flags);

    // --- BAŞLIK ALANI ---
    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Görselinizi Buraya Yükleyin");
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f)); // Araya estetik bir boşluk (Padding) bırakıyoruz

    // ==========================================
    // SÜRÜKLE BIRAK TASARIMI
    // ==========================================

    // Alanın boyutlarını belirliyoruz (Panelin mevcut genişliği kadar, 120px yüksekliğinde)
    ImVec2 dropZoneSize = ImVec2(ImGui::GetContentRegionAvail().x, 120.0f);

    // Çizim yapacağımız mutlak başlangıç noktasını kaydediyoruz
    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

    // 1. ALANI REZERVE ET: Görünmez buton hem ImGui'ye bu alanı kullandırır hem de hatayı önler.
    ImGui::InvisibleButton("DropZone", dropZoneSize);

    // Artık manuel matematik yok, doğrudan butonun Hover (üzerine gelme) durumunu soruyoruz
    bool isHovered = ImGui::IsItemHovered();

    ImU32 bgColor = isHovered ? IM_COL32(40, 40, 50, 255) : IM_COL32(25, 25, 30, 255);
    ImU32 borderColor = isHovered ? IM_COL32(255, 165, 0, 255) : IM_COL32(100, 100, 110, 255);

    // 2. KUTUYU FİZİKSEL OLARAK ÇİZ (DrawList Katmanı)
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), bgColor, 8.0f);
    drawList->AddRect(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), borderColor, 8.0f, 0, isHovered ? 2.0f : 1.0f);

    // 3. METNİ ÇİZ (Yine DrawList ile çiziyoruz ki imleci aşağı doğru bozmasın)
    const char* dropText = "Görsel Yükleyin";
    ImVec2 textSize = ImGui::CalcTextSize(dropText);

    ImVec2 textPos = ImVec2(
        cursorScreenPos.x + (dropZoneSize.x - textSize.x) * 0.5f,
        cursorScreenPos.y + (dropZoneSize.y - textSize.y) * 0.5f
    );

    ImU32 textColor = isHovered ? IM_COL32(255, 204, 102, 255) : IM_COL32(150, 150, 150, 255);
    drawList->AddText(textPos, textColor, dropText);

    // Sıradaki arayüz elemanları için kutunun altına biraz boşluk bırak
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    ImGui::End();

    // Stilleri Pop ile temizlemeyi unutmuyoruz
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
}