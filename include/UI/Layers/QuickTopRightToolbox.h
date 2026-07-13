#pragma once
#include <vector>
#include <string>
#include "io/AssetsManager/IconManager.h"

// Üst paneldeki araçlar birer durum değil, anlık "eylem" belirtir
enum class TopAction {
    NONE,
    UNDO,
    REDO,
    TURN_LEFT,
    TURN_RIGHT,
    ZOOM_IN,
    ZOOM_OUT,
    MIRROR_HORIZONTAL,
    MIRROR_VERTICAL
};

// Üst araç çubuğu için veri yapısı
struct TopToolUI_Info {
    TopAction id;
    std::string name;
    std::string tooltip;
    Icon icon;
    std::string definition;
};

class QuickTopRightToolbox {
public:
    QuickTopRightToolbox();
    ~QuickTopRightToolbox();

    void render(float displayWidth, float displayHeight);
    TopAction getLastAction() const { return lastAction; }
    void clearLastAction() { lastAction = TopAction::NONE; }

private:
    TopAction lastAction;
    std::vector<TopToolUI_Info> availableTools;
};