#include "UI/w_RightPanel.h"
#include "imgui.h"
#include "UI/w_TopPanel.h"

#include "UI/w_RightPanel.h"
#include "imgui.h"
#include "UI/w_TopPanel.h"
#include <windows.h>
#include <commdlg.h>
#include <string>
#include <iostream>


// --- DOSYA SEÇİCİ YARDIMCI FONKSİYONU ---
std::string openFileDialog() {
    char filename[MAX_PATH];
    OPENFILENAMEA ofn;
    ZeroMemory(&filename, sizeof(filename));
    ZeroMemory(&ofn, sizeof(ofn));
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;  // Uygulamanın ana penceresine bağlanabilir
    ofn.lpstrFilter = "Görsel Dosyaları\0*.png;*.jpg;*.jpeg;*.bmp\0Tüm Dosyalar\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = "Kıvılcım: Görsel Yükle";
    ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

    if (GetOpenFileNameA(&ofn)) {
        return std::string(filename);
    }
    return ""; // İptal edilirse boş döner
}

void RightPanel::render(float displayWidth, float displayHeight) {
    float panelWidth = 450.0f;
    float topPanelHeight = TopPanel::getPanelHeight();
    float panelHeight = displayHeight - topPanelHeight * 1.3f;

    if (panelHeight < 100.0f) panelHeight= 100.0f;

    float xPos = displayWidth - panelWidth - 15.0f;
    float yPos = 50.0f + topPanelHeight * 0.3f;

    // ==========================================
    // "GÖMÜLÜ VE ŞEFFAF" GÖRÜNÜM STİLLERİ
    // ==========================================

    // Köşe yumuşatmayı (Rounding) sıfırlıyoruz ki ekrana tam otursun
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);

    // Sınır çizgilerini (Border) kaldırıyoruz (Tam gömülü hissiyat için)
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

    // ARKA PLAN ŞEFFAFLIĞI (ALPHA AYARI)
    // Son değer (0.3f) Alpha değeridir. Eğer arkadaki 3D likit cam objen her şeyi yapacaksa
    // burayı 0.0f (Tamamen şeffaf) yapabilirsin. Şimdilik hafif karartılı yarı şeffaf bırakıyoruz.
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.02f, 0.02f, 0.03f, 0.0f));

    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(xPos, yPos), ImGuiCond_Always);

    ImGuiWindowFlags rightPanel_flags =
                                        ImGuiWindowFlags_NoResize   |
                                        ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoMove     |
                                        ImGuiWindowFlags_NoTitleBar; // Başlık çubuğunu kaldırdık (Daha temiz)

    ImGui::Begin("İçe Aktar", nullptr, rightPanel_flags);

    // --- BAŞLIK ALANI ---
    // TitleBar'ı kaldırdığımız için kendi başlığımızı metin olarak ekliyoruz
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "GÖRSEL YÜKLEME ALANI"); // Turuncu vurgulu başlık
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    // ==========================================
    // SÜRÜKLE BIRAK TASARIMI (Aynen Kalıyor)
    // ==========================================

    ImVec2 dropZoneSize = ImVec2(ImGui::GetContentRegionAvail().x, 120.0f);
    ImVec2 cursorScreenPos = ImGui::GetCursorScreenPos();

    ImGui::InvisibleButton("DropZone", dropZoneSize);
    bool isHovered = ImGui::IsItemHovered();
    bool isClicked = ImGui::IsItemClicked(); // Tıklanma olayını yakala

    if (isClicked) {
        std::string selectedImagePath = openFileDialog();

        if (!selectedImagePath.empty()) {
            std::cout << "[RightPanel] Secilen Gorsel: " << selectedImagePath << std::endl;

            // TODO: Bu dosya yolunu (path) alıp LeftPanel'in stack'ine ekleyeceğiz
            // VEYA doğrudan CUDA VRAM'e yükleyeceğiz.
        }
    }

    // Dropzone arka planını da biraz şeffaflaştıralım ki arkadaki cam efekti buradan da parlasın
    // IM_COL32(R, G, B, Alpha) -> Son parametre 255'ten 150'ye çekildi.
    ImU32 bgColor = isHovered ? IM_COL32(40, 40, 50, 180) : IM_COL32(25, 25, 30, 120);
    ImU32 borderColor = isHovered ? IM_COL32(255, 165, 0, 255) : IM_COL32(100, 100, 110, 150);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), bgColor, 8.0f);
    drawList->AddRect(cursorScreenPos, ImVec2(cursorScreenPos.x + dropZoneSize.x, cursorScreenPos.y + dropZoneSize.y), borderColor, 8.0f, 0, isHovered ? 2.0f : 1.0f);

    const char* dropText = "Görsel Yükleyin";
    ImVec2 textSize = ImGui::CalcTextSize(dropText);

    ImVec2 textPos = ImVec2(
        cursorScreenPos.x + (dropZoneSize.x - textSize.x) * 0.5f,
        cursorScreenPos.y + (dropZoneSize.y - textSize.y) * 0.5f
    );

    ImU32 textColor = isHovered ? IM_COL32(255, 204, 102, 255) : IM_COL32(150, 150, 150, 255);
    drawList->AddText(textPos, textColor, dropText);

    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    ImGui::End();

    // Sınır ve Köşe yumuşatmayı sıfırladığımız için 2 adet StyleVar poplamamız gerekiyor
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

