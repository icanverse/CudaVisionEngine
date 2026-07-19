#include "UI/Layers/WorkspaceToolboxes.h"

void WorkspaceToolboxes::render(float displayWidth, float displayHeight) {
    topCenterToolbox.render(displayWidth, displayHeight);
    rightToolbox.render(displayWidth, displayHeight);
    layersToolbox.render(displayWidth, displayHeight);
}
