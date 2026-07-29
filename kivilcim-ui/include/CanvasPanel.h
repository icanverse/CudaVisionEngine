#pragma once

#include "imgui.h"

namespace Kdata {
    struct WorkspaceStateData;
}

class CanvasPanel {
public:
    CanvasPanel();
    ~CanvasPanel();

    CanvasPanel(const CanvasPanel&) = delete;
    CanvasPanel& operator=(const CanvasPanel&) = delete;
    CanvasPanel(CanvasPanel&&) = delete;
    CanvasPanel& operator=(CanvasPanel&&) = delete;

    // Render artık doğrudan "State" (Veri) ve motorun birleştirdiği nihai dokuyu alıyor
    void render(
        Kdata::WorkspaceStateData* state,
        unsigned int compositeTextureId, // Motorun katmanları birleştirip ürettiği son doku
        float displayWidth,
        float displayHeight,
        float leftInset,
        float rightInset,
        float topInset,
        float bottomInset
    );

private:
    void drawCheckerboard(
        ImDrawList* drawList,
        const ImVec2& min,
        const ImVec2& max,
        float panX,
        float panY
    ) const;
};