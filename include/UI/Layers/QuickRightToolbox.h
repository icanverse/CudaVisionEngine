#pragma once

#include "io/AssetsManager/IconManager.h"

#include <string>
#include <vector>

enum class RightToolAction {
    NONE,
    CONTRAST,
    TEMPERATURE
};

class QuickRightToolbox {
public:
    QuickRightToolbox();
    ~QuickRightToolbox();

    void render(float displayWidth, float displayHeight);
    RightToolAction getCurrentTool() const { return currentTool; }

private:
    struct Tool {
        RightToolAction id;
        std::string name;
        Icon icon;
        std::string tooltip;
    };

    RightToolAction currentTool;
    std::vector<Tool> availableTools;
};
