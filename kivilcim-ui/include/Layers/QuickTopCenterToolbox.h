#pragma once

#include "AssetsManager/IconManager.h"

#include <string>
#include <vector>

enum class CenterToolAction {
    NONE,
    CIRCLE,
    LINE,
    SQUARE,
    VECTOR,
    BRUSH,
    ERASE,
    COLOR,
    TEXT,
    TEXT_SIZE,
    TEXT_COLOR
};

class QuickTopCenterToolbox {
public:
    QuickTopCenterToolbox();
    ~QuickTopCenterToolbox();

    void render(float displayWidth, float displayHeight);
    CenterToolAction getCurrentTool() const { return currentTool; }

private:
    struct Tool {
        CenterToolAction id;
        std::string name;
        Icon icon;
        std::string tooltip;
    };

    CenterToolAction currentTool;
    std::vector<Tool> availableTools;
};
