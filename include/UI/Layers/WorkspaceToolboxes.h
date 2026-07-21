#pragma once

#include "UI/CanvasPanel.h"
#include "UI/Layers/QuickLayersToolbox.h"
#include "UI/Layers/QuickRightToolbox.h"
#include "UI/Layers/QuickTopCenterToolbox.h"

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