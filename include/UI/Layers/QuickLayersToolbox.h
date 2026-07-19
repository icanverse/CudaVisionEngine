#pragma once

#include "io/AssetsManager/IconManager.h"

#include <string>
#include <vector>

enum class LayerToolAction {
    NONE,
    MOVE_DOWN,
    OPEN_LAYERS,
    ADD_LAYER,
    LOCK,
    UNION_LAYERS,
    TOGGLE_VISIBLE
};

class QuickLayersToolbox {
public:
    QuickLayersToolbox();
    ~QuickLayersToolbox();

    void render(float displayWidth, float displayHeight);
    LayerToolAction consumeLastAction();

private:
    struct Tool {
        LayerToolAction id;
        std::string name;
        Icon icon;
        std::string tooltip;
    };

    LayerToolAction lastAction;
    std::vector<Tool> availableTools;
};
