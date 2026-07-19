#pragma once

#include "UI/Layers/QuickLayersToolbox.h"
#include "UI/Layers/QuickRightToolbox.h"
#include "UI/Layers/QuickTopCenterToolbox.h"

class WorkspaceToolboxes {
public:
    void render(float displayWidth, float displayHeight);

    QuickTopCenterToolbox& topCenter() { return topCenterToolbox; }
    QuickRightToolbox& right() { return rightToolbox; }
    QuickLayersToolbox& layers() { return layersToolbox; }

private:
    QuickTopCenterToolbox topCenterToolbox;
    QuickRightToolbox rightToolbox;
    QuickLayersToolbox layersToolbox;
};
