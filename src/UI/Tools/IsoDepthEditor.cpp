#include "UI/Tools/IsoDepthEditor.h"

#include <string>

namespace Kivilcim {
namespace Tools {

IsoDepthEditor::IsoDepthEditor() {
    isoLines.clear();
}

void IsoDepthEditor::render(unsigned int baseTextureID, float imageWidth, float imageHeight) {
    if (!isOpen) return;

    // Kapatma butonu olan bağımsız bir pencere olarak açılır
    ImGui::SetNextWindowSize(ImVec2(1000, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("İzohips Derinlik Editörü", &isOpen)) {
        ImGui::End();
        return;
    }

    // Ekranı iki sütuna ayır: %75 Tuval, %25 Özellikler Paneli
    ImGui::Columns(2, "IsoDepthColumns");
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.75f);

    // ==========================================
    // SOL PANEL: ÇİZİM TUVALİ (CANVAS)
    // ==========================================
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    
    canvasMin = ImGui::GetCursorScreenPos();
    canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 50.0f) canvasSize.x = 50.0f;
    if (canvasSize.y < 50.0f) canvasSize.y = 50.0f;
    canvasMax = ImVec2(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);

    // Tuval arka planı (Koyu)
    drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(20, 20, 25, 255));

    // Orijinal görseli tuvale sığdırarak çiz (Aspect Ratio korunabilir ileride)
    if (baseTextureID != 0) {
        drawList->AddImage((ImTextureID)(intptr_t)baseTextureID, canvasMin, canvasMax);
    }

    // Görünmez buton ile fare girdilerini yakala
    ImGui::InvisibleButton("CanvasInput", canvasSize, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool isCanvasHovered = ImGui::IsItemHovered();
    ImVec2 mousePos = ImGui::GetIO().MousePos;

    // Farenin UV (0.0 - 1.0) koordinatı
    ImVec2 mouseUV = ImVec2((mousePos.x - canvasMin.x) / canvasSize.x, (mousePos.y - canvasMin.y) / canvasSize.y);

    // Sol Tık: Seçili çizgiye yeni nokta ekle
    if (isCanvasHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (isoLines.empty() || selectedLineIndex < 0 || selectedLineIndex >= isoLines.size()) {
            IsoLine newLine;
            isoLines.push_back(newLine);
            selectedLineIndex = (int)isoLines.size() - 1;
        }
        isoLines[selectedLineIndex].points.push_back({mouseUV});
    }

    // Çizgileri ve Noktaları Ekrana Çiz
    for (size_t l = 0; l < isoLines.size(); ++l) {
        auto& line = isoLines[l];
        bool isSelected = (l == selectedLineIndex);
        float thickness = isSelected ? 3.0f : 1.5f;
        ImU32 col = ImGui::ColorConvertFloat4ToU32(line.color);

        for (size_t p = 0; p < line.points.size(); ++p) {
            ImVec2 screenPt = ImVec2(canvasMin.x + line.points[p].uv.x * canvasSize.x, canvasMin.y + line.points[p].uv.y * canvasSize.y);

            // Çizgi bağlantıları
            if (p > 0) {
                ImVec2 prevPt = ImVec2(canvasMin.x + line.points[p - 1].uv.x * canvasSize.x, canvasMin.y + line.points[p - 1].uv.y * canvasSize.y);
                drawList->AddLine(prevPt, screenPt, col, thickness);
            }

            // Kontrol Noktaları
            drawList->AddCircleFilled(screenPt, isSelected ? 5.0f : 3.0f, isSelected ? IM_COL32(255, 255, 255, 255) : col);
        }
    }

    ImGui::NextColumn();

    // ==========================================
    // SAĞ PANEL: ÖZELLİKLER VE LİSTE
    // ==========================================
    ImGui::Text("İzohips Katmanları");
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    if (ImGui::Button("Yeni Hat Ekle", ImVec2(-1, 30))) {
        IsoLine newLine;
        isoLines.push_back(newLine);
        selectedLineIndex = (int)isoLines.size() - 1;
    }
    
    ImGui::Dummy(ImVec2(0.0f, 10.0f));

    // Liste kutusu (Mevcut hatları listele)
    if (ImGui::BeginListBox("##IsoLinesList", ImVec2(-1, 200))) {
        for (int i = 0; i < (int)isoLines.size(); ++i) {
            std::string label = "Hat " + std::to_string(i + 1) + " (Nokta: " + std::to_string(isoLines[i].points.size()) + ")";
            const bool isSelected = (selectedLineIndex == i);
            if (ImGui::Selectable(label.c_str(), isSelected)) {
                selectedLineIndex = i;
            }
        }
        ImGui::EndListBox();
    }

    ImGui::Dummy(ImVec2(0.0f, 15.0f));

    // Seçili hattın ayarları
    if (selectedLineIndex >= 0 && selectedLineIndex < isoLines.size()) {
        ImGui::Text("Seçili Hat Ayarları");
        ImGui::Separator();
        
        IsoLine& selLine = isoLines[selectedLineIndex];
        
        ImGui::Text("Derinlik (Z-Depth):");
        ImGui::SliderFloat("##DepthSlider", &selLine.depthValue, 0.0f, 1.0f, "%.2f");
        
        ImGui::ColorEdit3("Arayüz Rengi", (float*)&selLine.color, ImGuiColorEditFlags_NoInputs);
        
        ImGui::Dummy(ImVec2(0.0f, 10.0f));
        if (ImGui::Button("Hattı Sil", ImVec2(-1, 0))) {
            isoLines.erase(isoLines.begin() + selectedLineIndex);
            selectedLineIndex = -1;
        }
    }

    ImGui::Columns(1);
    ImGui::End();
}

} // namespace Tools
} // namespace Kivilcim