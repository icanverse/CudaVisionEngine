#include "Layers/WorkspaceToolboxes.h"

void WorkspaceToolboxes::render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight) {
    if (!state) return;

    const float layersWidth = QuickLayersToolbox::getPanelWidth();

    // DÜZELTME: 8 parametreyi tamamlamak için state->project.textureID (veya ilgili texture ID) eklendi.
    canvasPanel.render(
        state,
        state->project.textureID, // Eksik olan 2. parametre burasıydı!
        displayWidth,
        displayHeight,
        82.0f,
        layersWidth + 32.0f,
        112.0f,
        22.0f
    );

    topCenterToolbox.render(state, displayWidth, displayHeight);

    rightToolbox.render(state, displayWidth - layersWidth - 16.0f, displayHeight);
    layersToolbox.render(state, displayWidth, displayHeight);
}