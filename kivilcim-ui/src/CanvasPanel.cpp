#include "CanvasPanel.h"
#include "Data/ProjectData.h"
#include "TextureUtility/CudaDynamicTexture.cuh"

#include <cstdint>
#include <cmath> // Izgara kaydırma (fmodf) matematiği için eklendi

CanvasPanel::CanvasPanel() = default;
CanvasPanel::~CanvasPanel() = default;

void CanvasPanel::render(
    float displayWidth,
    float displayHeight,
    float leftInset,
    float rightInset,
    float topInset,
    float bottomInset
) {
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

    ImGui::TextColored(ImVec4(0.90f, 0.50f, 0.15f, 1.0f), "%s", title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("  %.0f%%", zoom * 100.0f);
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

        if (io.MouseWheel != 0.0f) {
            zoom *= io.MouseWheel > 0.0f ? 1.10f : 0.90f;
            if (zoom < 0.10f) zoom = 0.10f;
            if (zoom > 16.0f) zoom = 16.0f;
        }

        if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f)) {
            pan.x += io.MouseDelta.x;
            pan.y += io.MouseDelta.y;
        }

        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) resetView();
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(canvasMin, canvasMax, true);

    // Arkadaki deseni çizen fonksiyon çağrısı
    drawCheckerboard(drawList, canvasMin, canvasMax);

    if (textureId != 0 && imageWidth > 0 && imageHeight > 0) {
        const float fitX = canvasSize.x / static_cast<float>(imageWidth);
        const float fitY = canvasSize.y / static_cast<float>(imageHeight);
        const float fitScale = fitX < fitY ? fitX : fitY;
        const float finalScale = fitScale * 0.92f * zoom;
        const ImVec2 imageSize(
            static_cast<float>(imageWidth) * finalScale,
            static_cast<float>(imageHeight) * finalScale
        );
        const ImVec2 imageMin(
            canvasMin.x + (canvasSize.x - imageSize.x) * 0.5f + pan.x,
            canvasMin.y + (canvasSize.y - imageSize.y) * 0.5f + pan.y
        );
        const ImVec2 imageMax(
            imageMin.x + imageSize.x,
            imageMin.y + imageSize.y
        );

        drawList->AddImage(
            (ImTextureID)(intptr_t)textureId,
            imageMin,
            imageMax,
            ImVec2(0.0f, 1.0f),
            ImVec2(1.0f, 0.0f)
        );

    } else {
        const char* emptyText = "Gorsel texture'i henuz CanvasPanel'e baglanmadi";
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

void CanvasPanel::setImage(
    unsigned int newTextureId,
    int newImageWidth,
    int newImageHeight,
    const std::string& imageName
) {
    const bool imageChanged =
        textureId != newTextureId ||
        imageWidth != newImageWidth ||
        imageHeight != newImageHeight;

    textureId = newTextureId;
    imageWidth = newImageWidth;
    imageHeight = newImageHeight;
    if (!imageName.empty()) title = imageName;
    if (imageChanged) resetView();
}

void CanvasPanel::setProject(const Kivilcim::ProjectData* project) {
    if (project == nullptr) {
        clearImage();
        return;
    }

    const int sourceWidth = project->size.x > 0
        ? project->size.x
        : project->projectWidth;
    const int sourceHeight = project->size.y > 0
        ? project->size.y
        : project->projectHeight;

    if (
        project->isLoadedToGPU &&
        project->d_imageData != nullptr &&
        sourceWidth > 0 &&
        sourceHeight > 0 &&
        (project->channels == 1 || project->channels == 3 || project->channels == 4)
    ) {
        const bool textureMustBeCreated =
            dynamicTexture == nullptr ||
            dynamicTexture->getWidth() != sourceWidth ||
            dynamicTexture->getHeight() != sourceHeight;

        if (textureMustBeCreated) {
            dynamicTexture = std::make_unique<CudaDynamicTexture>(
                sourceWidth,
                sourceHeight
            );
        }

        if (dynamicTexture->updateFromDeviceData(
                project->d_imageData,
                project->channels,
                1.0f / 255.0f
            )) {
            setImage(
                dynamicTexture->getTextureID(),
                sourceWidth,
                sourceHeight,
                project->name
            );
            return;
        }
    }

    dynamicTexture.reset();
    setImage(
        project->textureID,
        sourceWidth,
        sourceHeight,
        project->name
    );
}

void CanvasPanel::clearImage() {
    dynamicTexture.reset();
    textureId = 0;
    imageWidth = 0;
    imageHeight = 0;
    resetView();
}

void CanvasPanel::resetView() {
    zoom = 1.0f;
    pan = ImVec2(0.0f, 0.0f);
}

// İsim hata vermesin diye drawCheckerboard kaldı ancak modern bir grid (ızgara) çiziyor
void CanvasPanel::drawCheckerboard(
    ImDrawList* drawList,
    const ImVec2& min,
    const ImVec2& max
) const {
    // 1. Zemin rengi: Tamamen koyu, dikkat dağıtmayan mat antrasit
    drawList->AddRectFilled(min, max, IM_COL32(16, 16, 18, 255));

    // 2. Modern ızgara çizgileri (Blueprint/Node editor stili)
    constexpr float gridSize = 32.0f;
    const ImU32 gridColor = IM_COL32(32, 32, 36, 255); // Çok hafif belli olan ince referans çizgileri

    // Kamerayı kaydırdığında (pan) ızgaranın da senkronize hareket etmesini sağlayan matematik
    float offsetX = std::fmod(pan.x, gridSize);
    float offsetY = std::fmod(pan.y, gridSize);
    if (offsetX < 0) offsetX += gridSize;
    if (offsetY < 0) offsetY += gridSize;

    // Yatay ve dikey çizgileri çizerek sonsuz bir doku oluştur
    for (float x = min.x + offsetX; x < max.x; x += gridSize) {
        drawList->AddLine(ImVec2(x, min.y), ImVec2(x, max.y), gridColor, 1.0f);
    }
    for (float y = min.y + offsetY; y < max.y; y += gridSize) {
        drawList->AddLine(ImVec2(min.x, y), ImVec2(max.x, y), gridColor, 1.0f);
    }
}