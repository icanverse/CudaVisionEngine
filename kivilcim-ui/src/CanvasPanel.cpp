#include "CanvasPanel.h"
#include "Data/WorkspaceStateData.h" // Kdata namespace'i ve struct'ları için

#include <cstdint>
#include <cmath>

CanvasPanel::CanvasPanel() = default;
CanvasPanel::~CanvasPanel() = default;

void CanvasPanel::render(
    Kdata::WorkspaceStateData* state,
    unsigned int compositeTextureId,
    float displayWidth,
    float displayHeight,
    float leftInset,
    float rightInset,
    float topInset,
    float bottomInset
) {
    if (!state) return; // Güvenlik kontrolü

    const float requestedWidth = displayWidth - leftInset - rightInset;
    const float requestedHeight = displayHeight - topInset - bottomInset;
    const float panelWidth = requestedWidth > 240.0f ? requestedWidth : 240.0f;
    const float panelHeight = requestedHeight > 180.0f ? requestedHeight : 180.0f;

    ImGui::SetCursorPos(ImVec2(leftInset, topInset));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.08f, 0.09f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.15f, 0.15f, 0.16f, 1.0f));

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 2.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.0f, 8.0f));

    ImGui::BeginChild(
        "CanvasPanel",
        ImVec2(panelWidth, panelHeight),
        true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse
    );

    // Başlığı doğrudan state üzerinden okuyoruz
    std::string displayTitle = state->project.name.empty() ? "İsimsiz Proje" : state->project.name;

    ImGui::TextColored(ImVec4(0.90f, 0.50f, 0.15f, 1.0f), "%s", displayTitle.c_str());
    ImGui::SameLine();
    // Zoom bilgisini Viewport'tan okuyoruz
    ImGui::TextDisabled("  %.0f%%", state->viewport.zoomLevel * 100.0f);
    ImGui::SameLine();
    ImGui::TextDisabled("  |  Orta tus: kaydir  |  Tekerlek: zoom  |  Cift tik: sifirla");

    ImGui::PushStyleColor(ImGuiCol_Separator, ImVec4(0.18f, 0.18f, 0.20f, 1.0f));
    ImGui::Separator();
    ImGui::PopStyleColor();

    ImVec2 canvasSize = ImGui::GetContentRegionAvail();
    if (canvasSize.x < 1.0f) canvasSize.x = 1.0f;
    if (canvasSize.y < 1.0f) canvasSize.y = 1.0f;

    const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
    const ImVec2 canvasMax(
        canvasMin.x + canvasSize.x,
        canvasMin.y + canvasSize.y
    );

    ImGui::InvisibleButton("CanvasInteraction", canvasSize);
    const bool hovered = ImGui::IsItemHovered();

    if (hovered) {
        ImGuiIO& io = ImGui::GetIO();

        // Etkileşimler doğrudan state->viewport verilerini günceller
        if (io.MouseWheel != 0.0f) {
            state->viewport.zoomLevel *= io.MouseWheel > 0.0f ? 1.10f : 0.90f;
            if (state->viewport.zoomLevel < 0.10f) state->viewport.zoomLevel = 0.10f;
            if (state->viewport.zoomLevel > 16.0f) state->viewport.zoomLevel = 16.0f;
        }

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f)) {
            state->viewport.cameraPosX += io.MouseDelta.x;
            state->viewport.cameraPosY += io.MouseDelta.y;
        }

        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            state->viewport.zoomLevel = 1.0f;
            state->viewport.cameraPosX = 0.0f;
            state->viewport.cameraPosY = 0.0f;
        }
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(canvasMin, canvasMax, true);

    // Arkadaki deseni çizen fonksiyon çağrısı (Pan verilerini state'den geçiriyoruz)
    drawCheckerboard(drawList, canvasMin, canvasMax, state->viewport.cameraPosX, state->viewport.cameraPosY);

    // Eğer projenin bir genişliği/yüksekliği varsa ve motor bize bir doku (texture) verdiyse çiz
    if (compositeTextureId != 0 && state->project.projectWidth > 0 && state->project.projectHeight > 0) {
        const float fitX = canvasSize.x / static_cast<float>(state->project.projectWidth);
        const float fitY = canvasSize.y / static_cast<float>(state->project.projectHeight);
        const float fitScale = fitX < fitY ? fitX : fitY;
        const float finalScale = fitScale * 0.92f * state->viewport.zoomLevel;

        const ImVec2 imageSize(
            static_cast<float>(state->project.projectWidth) * finalScale,
            static_cast<float>(state->project.projectHeight) * finalScale
        );
        const ImVec2 imageMin(
            canvasMin.x + (canvasSize.x - imageSize.x) * 0.5f + state->viewport.cameraPosX,
            canvasMin.y + (canvasSize.y - imageSize.y) * 0.5f + state->viewport.cameraPosY
        );
        const ImVec2 imageMax(
            imageMin.x + imageSize.x,
            imageMin.y + imageSize.y
        );

        drawList->AddImage(
            (ImTextureID)(intptr_t)compositeTextureId,
            imageMin,
            imageMax,
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f)
        );

    } else {
        const char* emptyText = "Tuval Bos Veya Gorsel Yuklenmedi";
        const ImVec2 textSize = ImGui::CalcTextSize(emptyText);
        drawList->AddText(
            ImVec2(
                canvasMin.x + (canvasSize.x - textSize.x) * 0.5f,
                canvasMin.y + (canvasSize.y - textSize.y) * 0.5f
            ),
            IM_COL32(110, 110, 115, 255),
            emptyText
        );
    }

    drawList->PopClipRect();
    ImGui::EndChild();

    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(2);
}

void CanvasPanel::drawCheckerboard(
    ImDrawList* drawList,
    const ImVec2& min,
    const ImVec2& max,
    float panX,
    float panY
) const {
    drawList->AddRectFilled(min, max, IM_COL32(16, 16, 18, 255));

    constexpr float gridSize = 32.0f;
    const ImU32 gridColor = IM_COL32(32, 32, 36, 255);

    float offsetX = std::fmod(panX, gridSize);
    float offsetY = std::fmod(panY, gridSize);
    if (offsetX < 0) offsetX += gridSize;
    if (offsetY < 0) offsetY += gridSize;

    for (float x = min.x + offsetX; x < max.x; x += gridSize) {
        drawList->AddLine(ImVec2(x, min.y), ImVec2(x, max.y), gridColor, 1.0f);
    }
    for (float y = min.y + offsetY; y < max.y; y += gridSize) {
        drawList->AddLine(ImVec2(min.x, y), ImVec2(max.x, y), gridColor, 1.0f);
    }
}