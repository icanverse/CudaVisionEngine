#include "../../include/UI/w_RightPanel.h"
#include <iostream>
#include "imgui.h"
#include "UI/w_TopPanel.h"
#include <windows.h>
#include <commdlg.h>

// --- DOSYA SEÇİCİ YARDIMCI FONKSİYONU ---
std::string openFileDialog() {
    char filename[MAX_PATH];
    OPENFILENAMEA ofn;
    ZeroMemory(&filename, sizeof(filename));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Gorsel Dosyalari\0*.png;*.jpg;*.jpeg;*.bmp\0Tum Dosyalar\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Kivilcim: Gorsel Yukle";
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return "";
}

// --- CONSTRUCTOR ---
RightPanel::RightPanel() {
    memset(projectNameBuf, 0, sizeof(projectNameBuf));
    docWidth = 1920;
    docHeight = 1080;
    dimMetric = 0; // Piksel
    orientation = 1; // 1 = Yatay başlasın
    resolution = 72; // Standart web/ekran çözünürlüğü
    resMetric = 0; // Piksel/İnç
    bgContentMode = 0; // Beyaz
    bgColor[0] = 1.0f; bgColor[1] = 1.0f; bgColor[2] = 1.0f;
    selectedImagePath = "";
    keepOriginalSize = true; // Varsayılan olarak kilitli gelsin
}

// --- RENDER DÖNGÜSÜ ---
void RightPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 450.0f;
    float topPanelHeight = TopPanel::getPanelHeight();

    float realScreenHeight = ImGui::GetIO().DisplaySize.y;
    float realScreenWidth = ImGui::GetIO().DisplaySize.x;

    float xPos = realScreenWidth - panelWidth - 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;
    float panelHeight = realScreenHeight - yPos - 15.0f;

    if (panelHeight < 100.0f) panelHeight = 100.0f;

    // ==========================================
    // TEMA VE STİL AYARLARI (TURUNCU KONSEPT & CAM EFEKTİ)
    // ==========================================
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.2f);

    // 1. Ana Pencere ve Çerçeve
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.05f, 0.05f, 0.06f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.3f, 0.3f, 0.35f, 0.5f));

    // 2. Input (Kutucuk) Arkaplanları - Mavimsiliği alıp Nötr Koyu Gri yapıyoruz
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.13f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.18f, 0.18f, 0.19f, 1.0f)); // Fareyle üstüne gelince hafif açılsın
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.85f, 0.45f, 0.0f, 0.6f)); // İçine tıklayınca turuncu vurgu

    // 3. Dropdown (Açılır Liste) ve Seçim Renkleri - Mavi seçimleri turuncu yapıyoruz
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.08f, 0.08f, 0.09f, 0.98f)); // Açılır menü arkaplanı (Neredeyse tam siyah)
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.85f, 0.45f, 0.0f, 0.5f)); // Listede seçili olan öğe
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.85f, 0.45f, 0.0f, 0.7f)); // Listede fareyle üstüne gelinen öğe
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.85f, 0.45f, 0.0f, 0.9f)); // Listede tıklanan öğe

    // 4. Diğer Vurgular
    ImGui::PushStyleColor(ImGuiCol_CheckMark, ImVec4(1.0f, 0.6f, 0.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_TextSelectedBg, ImVec4(0.85f, 0.45f, 0.0f, 0.4f)); // Yazı seçme (Highlight) rengi

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags rightPanel_flags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar;

    ImGui::Begin("Proje Hazirlik", nullptr, rightPanel_flags);

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // ==========================================
    // 1. PROJE ŞABLONU (ÜST KISIM)
    // ==========================================
    ImGui::Dummy(ImVec2(0.0f, 5.0f));
    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "PROJE DETAYLARI");
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // Proje Adı
    ImGui::Text("Proje Adı:");
    ImGui::PushItemWidth(panelWidth * 0.9f);
    ImGui::InputTextWithHint("##ProjAdi", "İsimsiz-1", projectNameBuf, IM_ARRAYSIZE(projectNameBuf));
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Orijinal Boyutları Koru Checkbox'ı
    ImGui::Checkbox("Orijinal Çözünürlüğü Koru", &keepOriginalSize);
    ImGui::Dummy(ImVec2(0.0f, 5.0f));

    // EĞER İŞARETLİYSE ALTAKİLERİ KİLİTLE (DISABLE)
    if (keepOriginalSize) ImGui::BeginDisabled();

    // Genişlik ve Yükseklik YAN YANA
    float colWidth = 130.0f;
    const char* dimMetrics[] = { "Piksel", "İnç", "Santimetre" };

    ImGui::Text("Genişlik:");
    ImGui::SameLine(colWidth + 25.0f);
    ImGui::Text("Yükseklik:");

    ImGui::PushItemWidth(colWidth);
    ImGui::InputInt("##Width", &docWidth, 0, 0);
    ImGui::PopItemWidth();

    ImGui::SameLine(colWidth + 25.0f);
    ImGui::PushItemWidth(colWidth);
    ImGui::InputInt("##Height", &docHeight, 0, 0);
    ImGui::PopItemWidth();

    ImGui::SameLine();
    ImGui::PushItemWidth(panelWidth - (colWidth * 2) - 45.0f);
    ImGui::Combo("##DimMetric", &dimMetric, dimMetrics, IM_ARRAYSIZE(dimMetrics));
    ImGui::PopItemWidth();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Oryantasyon (Mavi renkler Turuncu ile değiştirildi)
    ImGui::Text("Oryantasyon:");
    ImGui::SameLine(100.0f);

    if (orientation == 0) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f)); // Turuncu
    else ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);
    if (ImGui::Button(" | ", ImVec2(35, 25))) { orientation = 0; }
    ImGui::PopStyleColor();

    ImGui::SameLine();

    if (orientation == 1) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f)); // Turuncu
    else ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_Button]);
    if (ImGui::Button(" - ", ImVec2(35, 25))) { orientation = 1; }
    ImGui::PopStyleColor();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Çözünürlük (Resolution)
    const char* resMetrics[] = { "Piksel/İnç", "Piksel/cm" };
    ImGui::Text("Çözünürlük:");
    ImGui::PushItemWidth(colWidth);
    ImGui::InputInt("##Res", &resolution, 0, 0);
    ImGui::PopItemWidth();

    ImGui::SameLine(colWidth + 25.0f);
    ImGui::PushItemWidth(colWidth);
    ImGui::Combo("##ResMetric", &resMetric, resMetrics, IM_ARRAYSIZE(resMetrics));
    ImGui::PopItemWidth();

    // KİLİDİ AÇ
    if (keepOriginalSize) ImGui::EndDisabled();

    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Arka Plan İçeriği (Background Contents)
    ImGui::Text("Arka Plan İçeriği:");
    const char* bgContents[] = { "Beyaz", "Siyah", "Şeffaf", "Özel Renk" };

    ImGui::PushItemWidth(colWidth * 1.5f);
    if (ImGui::Combo("##BgContent", &bgContentMode, bgContents, IM_ARRAYSIZE(bgContents))) {
        if (bgContentMode == 0) { bgColor[0] = 1.0f; bgColor[1] = 1.0f; bgColor[2] = 1.0f; }
        if (bgContentMode == 1) { bgColor[0] = 0.0f; bgColor[1] = 0.0f; bgColor[2] = 0.0f; }
    }
    ImGui::PopItemWidth();

    ImGui::SameLine();

    ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_AlphaPreview;
    if (ImGui::ColorEdit3("##ColorBox", bgColor, colorFlags)) {
        bgContentMode = 3;
    }

    ImGui::Dummy(ImVec2(0.0f, 15.0f));
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // ==========================================
    // 2. GÖRSEL YÜKLEME ALANI
    // ==========================================
    ImVec2 dropZoneSize = ImVec2(ImGui::GetContentRegionAvail().x * 0.9f, 120.0f);

    float dropZoneIndent = (ImGui::GetContentRegionAvail().x - dropZoneSize.x) * 0.5f;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dropZoneIndent);

    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

    ImGui::InvisibleButton("DropZone", dropZoneSize);
    bool isHovered = ImGui::IsItemHovered();

    if (ImGui::IsItemClicked()) {
        std::string tempPath = openFileDialog();
        if (!tempPath.empty()) {
            selectedImagePath = tempPath;
            std::cout << "[RightPanel] Görsel seçildi: " << selectedImagePath << std::endl;
        }
    }

    ImU32 bgColorZone = isHovered ? IM_COL32(50, 50, 60, 200) : IM_COL32(30, 30, 35, 180);
    ImU32 borderColor = isHovered ? IM_COL32(255, 165, 0, 255) : IM_COL32(100, 100, 110, 150);

    drawList->AddRectFilled(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), bgColorZone, 8.0f);
    drawList->AddRect(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), borderColor, 8.0f, 0, isHovered ? 2.0f : 1.0f);

    std::string dropText = selectedImagePath.empty() ? "Görsel Seçin (İsteğe Bağlı)" : "Görsel Hazır!\nDeğiştirmek için tıkla.";
    ImVec2 textSize = ImGui::CalcTextSize(dropText.c_str());

    ImVec2 textPos = ImVec2(
        cursorScreenPos.x + (dropZoneSize.x - textSize.x) * 0.5f,
        cursorScreenPos.y + (dropZoneSize.y - textSize.y) * 0.5f
    );

    ImU32 textColor = selectedImagePath.empty() ? (isHovered ? IM_COL32(255, 204, 102, 255) : IM_COL32(180, 180, 180, 255)) : IM_COL32(120, 255, 120, 255);
    drawList->AddText(textPos, textColor, dropText.c_str());

    // ==========================================
    // 3. OLUŞTUR BUTONU (EN ALT KISIM)
    // ==========================================
    float buttonHeight = 45.0f;
    float currentY = ImGui::GetCursorPosY();

    float spaceLeft = panelHeight - currentY - buttonHeight - 35.0f;

    if (spaceLeft > 0) {
        ImGui::Dummy(ImVec2(0.0f, spaceLeft));
    }

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.85f, 0.45f, 0.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.55f, 0.0f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.65f, 0.35f, 0.0f, 1.0f));

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + dropZoneIndent);

    if (ImGui::Button("PROJEYİ OLUŞTUR", ImVec2(dropZoneSize.x, buttonHeight))) {
        if (onImageImported) {
            onImageImported(selectedImagePath);
        }
        selectedImagePath = "";
        memset(projectNameBuf, 0, sizeof(projectNameBuf));
    }

    ImGui::PopStyleColor(3);

    ImGui::End();

    // Yukarıda tanımladığımız 6 renk ve 2 stil değerini temizliyoruz.
    ImGui::PopStyleColor(11);
    ImGui::PopStyleVar(2);
}