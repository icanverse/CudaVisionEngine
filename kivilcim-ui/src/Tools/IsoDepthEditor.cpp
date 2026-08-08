#include "Tools/IsoDepthEditor.h"

#include <imgui.h>
#include <string>
#include <cstdint>

namespace Kivilcim {
    namespace Tools {

        IsoDepthEditor::IsoDepthEditor() {
            isoLines.clear();
        }

void IsoDepthEditor::render(unsigned int baseTextureID, float imageWidth, float imageHeight) {
    if (!isOpen) return;

    ImGui::SetNextWindowSize(ImVec2(1100, 750), ImGuiCond_FirstUseEver);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));

    if (!ImGui::Begin("İzohips Derinlik Editörü", &isOpen, ImGuiWindowFlags_NoCollapse)) {
        ImGui::PopStyleVar();
        ImGui::End();
        return;
    }
    ImGui::PopStyleVar();

    float propertiesPanelWidth = 320.0f;
    float canvasWidth = ImGui::GetContentRegionAvail().x - propertiesPanelWidth;

    // ==========================================
    // SOL PANEL: ÇİZİM TUVALİ (CANVAS)
    // ==========================================
    ImGui::BeginChild("CanvasRegion", ImVec2(canvasWidth, 0), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    ImVec2 canvasMin = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 50.0f) canvasSize.x = 50.0f;
    if (canvasSize.y < 50.0f) canvasSize.y = 50.0f;
    ImVec2 canvasMax = ImVec2(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);

    // 1. Tuvalin tamamını koyu arka planla doldur
    drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(25, 25, 30, 255));

    // ==========================================
    // ASPECT RATIO (EN/BOY ORANI) VE ORTALAMA
    // ==========================================
    ImVec2 imageMin = canvasMin;
    ImVec2 imageMax = canvasMax;
    ImVec2 drawSize = canvasSize;

    if (imageWidth > 0 && imageHeight > 0) {
        float imageAspect = imageWidth / imageHeight;
        float canvasAspect = canvasSize.x / canvasSize.y;

        if (canvasAspect > imageAspect) {
            // Tuval resimden daha geniş -> Yüksekliği tam oturt, genişliği daralt
            drawSize.y = canvasSize.y;
            drawSize.x = canvasSize.y * imageAspect;
        } else {
            // Tuval resimden daha yüksek -> Genişliği tam oturt, yüksekliği daralt
            drawSize.x = canvasSize.x;
            drawSize.y = canvasSize.x / imageAspect;
        }

        // Resmi tuvalin tam ortasına konumlandırmak için ofset hesapla
        float offsetX = (canvasSize.x - drawSize.x) * 0.5f;
        float offsetY = (canvasSize.y - drawSize.y) * 0.5f;

        imageMin = ImVec2(canvasMin.x + offsetX, canvasMin.y + offsetY);
        imageMax = ImVec2(imageMin.x + drawSize.x, imageMin.y + drawSize.y);
    }

    // 2. Resmi deforme etmeden ve ters-Y hatası çözülmüş şekilde çiz
    if (baseTextureID != 0) {
        drawList->AddImage(
            (ImTextureID)(intptr_t)baseTextureID,
            imageMin,
            imageMax,
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f)
        );
    }

    // Görünmez buton tuvalin tamamını kaplar, tıklamaları yakalar
    ImGui::InvisibleButton("CanvasInput", canvasSize, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool isCanvasHovered = ImGui::IsItemHovered();
    ImVec2 mousePos = ImGui::GetIO().MousePos;

    // UV koordinatını artık tuvale (canvas) göre değil, içteki resme (imageMin, drawSize) göre hesapla
    ImVec2 mouseUV = ImVec2((mousePos.x - imageMin.x) / drawSize.x, (mousePos.y - imageMin.y) / drawSize.y);

    // Farenin resmin sınırları içinde olup olmadığını kontrol et (Siyah boşluklara tıklanmasın)
    bool isMouseOverImage = (mouseUV.x >= 0.0f && mouseUV.x <= 1.0f && mouseUV.y >= 0.0f && mouseUV.y <= 1.0f);

    // Sol Tık: Sadece resmin üzerine tıklandığında yeni nokta ekle
    if (isCanvasHovered && isMouseOverImage && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        if (isoLines.empty() || selectedLineIndex < 0 || selectedLineIndex >= isoLines.size()) {
            isoLines.push_back(IsoLine());
            selectedLineIndex = (int)isoLines.size() - 1;
        }
        isoLines[selectedLineIndex].points.push_back({mouseUV});
    }

    // 3. Çizgileri ve Noktaları Ekrana Çiz (Resmin oranlarına sadık kalarak)
    for (size_t l = 0; l < isoLines.size(); ++l) {
        const auto& line = isoLines[l];
        bool isSelected = (l == selectedLineIndex);
        float thickness = isSelected ? 3.5f : 2.0f;
        ImU32 col = ImGui::ColorConvertFloat4ToU32(line.color);

        for (size_t p = 0; p < line.points.size(); ++p) {
            // Noktaların ekrandaki koordinatları resmin pozisyonuna ve boyutuna göre ölçeklenir
            ImVec2 screenPt = ImVec2(imageMin.x + line.points[p].uv.x * drawSize.x, imageMin.y + line.points[p].uv.y * drawSize.y);

            // Çizgi bağlantıları
            if (p > 0) {
                ImVec2 prevPt = ImVec2(imageMin.x + line.points[p - 1].uv.x * drawSize.x, imageMin.y + line.points[p - 1].uv.y * drawSize.y);
                drawList->AddLine(prevPt, screenPt, col, thickness);
            }

            // Kontrol Noktaları
            drawList->AddCircleFilled(screenPt, isSelected ? 5.0f : 3.5f, isSelected ? IM_COL32(255, 255, 255, 255) : col);
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // ==========================================
    // SAĞ PANEL: ÖZELLİKLER VE LİSTE
    // ==========================================
    ImGui::BeginChild("PropertiesRegion", ImVec2(0, 0), true, 0); // Hata veren padding bayrağı '0' yapıldı

    ImGui::Spacing();
    ImGui::TextDisabled("KIVILCIM İZOHİPS ARAÇLARI");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Button(" Yeni Hat Ekle ", ImVec2(-1, 35))) {
        isoLines.push_back(IsoLine());
        selectedLineIndex = (int)isoLines.size() - 1;
    }

    ImGui::Spacing();

    if (ImGui::BeginListBox("##IsoLinesList", ImVec2(-1, 220))) {
        for (int i = 0; i < (int)isoLines.size(); ++i) {
            std::string label = "Hat " + std::to_string(i + 1) + "  [" + std::to_string(isoLines[i].points.size()) + " Nokta]";
            const bool isSelected = (selectedLineIndex == i);
            if (ImGui::Selectable(label.c_str(), isSelected)) {
                selectedLineIndex = i;
            }
        }
        ImGui::EndListBox();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (selectedLineIndex >= 0 && selectedLineIndex < isoLines.size()) {
        IsoLine& selLine = isoLines[selectedLineIndex];

        ImGui::Text("Hat Ayarları");
        ImGui::Spacing();

        ImGui::Text("Derinlik (Z-Depth):");
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##DepthSlider", &selLine.depthValue, 0.0f, 1.0f, "%.3f");

        ImGui::Spacing();
        ImGui::Text("Arayüz Rengi:");
        ImGui::SetNextItemWidth(-1);
        ImGui::ColorEdit3("##Color", (float*)&selLine.color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_Float);

        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(180, 40, 40, 255));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, IM_COL32(220, 60, 60, 255));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, IM_COL32(255, 80, 80, 255));
        if (ImGui::Button("Hattı Sil", ImVec2(-1, 30))) {
            isoLines.erase(isoLines.begin() + selectedLineIndex);
            selectedLineIndex = -1;
        }
        ImGui::PopStyleColor(3);
    } else {
        ImGui::TextDisabled("Düzenlemek için bir hat seçin.");
    }

    ImGui::EndChild();
    ImGui::End();
}

} // namespace Tools
} // namespace Kivilcim