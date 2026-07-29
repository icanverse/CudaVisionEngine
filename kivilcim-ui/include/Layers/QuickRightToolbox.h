#pragma once

#include "Data/WorkspaceStateData.h"

class QuickRightToolbox {
public:
    QuickRightToolbox() = default;
    ~QuickRightToolbox() = default;

    // YENİ: Sadece state referansı alarak çalışır
    void render(Kdata::WorkspaceStateData* state, float displayWidth, float displayHeight);
};