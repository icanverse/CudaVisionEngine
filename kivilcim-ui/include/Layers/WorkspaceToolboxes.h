#pragma once

#include "CanvasPanel.h"
#include "Layers/QuickLayersToolbox.h"
#include "Layers/QuickRightToolbox.h"
#include "Layers/QuickTopCenterToolbox.h"

class WorkspaceToolboxes {
public:
    void render(float displayWidth, float displayHeight);

    CanvasPanel& canvas() { return canvasPanel; }
    QuickTopCenterToolbox& topCenter() { return topCenterToolbox; }
    QuickRightToolbox& right() { return rightToolbox; }
    QuickLayersToolbox& layers() { return layersToolbox; }

private:
    CanvasPanel canvasPanel;
    QuickTopCenterToolbox topCenterToolbox;
    QuickRightToolbox rightToolbox;
    QuickLayersToolbox layersToolbox;
};