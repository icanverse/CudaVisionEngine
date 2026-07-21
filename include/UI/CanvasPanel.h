#pragma once

#include "imgui.h"

#include <memory>
#include <string>

class CudaDynamicTexture;

namespace Kivilcim {
    struct ProjectData;
}

class CanvasPanel {
public:
    CanvasPanel();
    ~CanvasPanel();

    CanvasPanel(const CanvasPanel&) = delete;
    CanvasPanel& operator=(const CanvasPanel&) = delete;
    CanvasPanel(CanvasPanel&&) = delete;
    CanvasPanel& operator=(CanvasPanel&&) = delete;

    void render(
        float displayWidth,
        float displayHeight,
        float leftInset,
        float rightInset,
        float topInset,
        float bottomInset
    );

    void setImage(
        unsigned int textureId,
        int imageWidth,
        int imageHeight,
        const std::string& imageName = {}
    );

    void setProject(const Kivilcim::ProjectData* project);
    void clearImage();
    void resetView();

    void setTitle(const std::string& newTitle) { title = newTitle; }
    float getZoom() const { return zoom; }
    unsigned int getDisplayTextureId() const { return textureId; }

private:
    void drawCheckerboard(
        ImDrawList* drawList,
        const ImVec2& min,
        const ImVec2& max
    ) const;

    unsigned int textureId = 0;
    int imageWidth = 0;
    int imageHeight = 0;
    std::string title = "Calisma Tuvali";

    float zoom = 1.0f;
    ImVec2 pan = ImVec2(0.0f, 0.0f);

    std::unique_ptr<CudaDynamicTexture> dynamicTexture;
};