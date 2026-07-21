#include "UI/Layers/WorkspaceToolboxes.h"

void WorkspaceToolboxes::render(float displayWidth, float displayHeight) {
    const float layersWidth = QuickLayersToolbox::getPanelWidth();

    // Canvas once cizilir; yüzen toolbox'lar onun uzerinde kalir.
    canvasPanel.render(
        displayWidth,
        displayHeight,
        82.0f,
        layersWidth + 32.0f,
        112.0f,
        22.0f
    );

    topCenterToolbox.render(displayWidth, displayHeight);

    // Kontrast/sicaklik kutusunu genis Layers panelinin soluna al.
    rightToolbox.render(displayWidth - layersWidth - 16.0f, displayHeight);
    layersToolbox.render(displayWidth, displayHeight);
}